"""Phase 3B Sequence Preprocessor and Tensor Ingestion Pipeline.

Implements isolated and controlled transformations for Phase 3B experiments:
- M0: Frozen baseline replication (34 channels, linear z-score, legacy zeroed c_object_type, no dt).
- M1: Categorical correction ONLY (37 channels, one-hot binary c_object_type, linear z-score, no dt).
- M2: Covariance log10 transform ONLY (34 channels, monotonic log10 on sigmas & mahalanobis, linear z-score on others).
- M3: Explicit Delta-t ONLY (35 channels, inter-arrival time dt from time_to_tca, linear z-score).
- M4: Combination M1 + M2 (37 channels, categorical one-hot + covariance log10).
- M5: Full Combination M1 + M2 + M3 (38 channels, categorical one-hot + covariance log10 + Delta-t).

Enforces strict zero-leakage constraints: preprocessor is fitted strictly on training events.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from orvexa.event_builder import DIRECT_FEATURE_COLUMNS
from orvexa.phase3b_config import (
    COVARIANCE_LOG10_FEATURE_COLUMNS,
    ExperimentID,
    OBJECT_TYPE_ONE_HOT_CHANNELS,
    Phase3BExperimentConfig,
    RAW_OBJECT_TYPE_CATEGORIES,
    get_experiment_config,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def encode_c_object_type_one_hot(raw_val: Any) -> List[float]:
    """Deterministically encode c_object_type into 4 binary one-hot channels.
    
    Category Mapping:
    - 'DEBRIS' -> [1.0, 0.0, 0.0, 0.0]
    - 'PAYLOAD' -> [0.0, 1.0, 0.0, 0.0]
    - 'ROCKET BODY' / 'ROCKET_BODY' -> [0.0, 0.0, 1.0, 0.0]
    - 'UNKNOWN' / 'TBA' / '' / None / other -> [0.0, 0.0, 0.0, 1.0]
    
    Returns:
        Exact 4-element binary float list summing to 1.0.
    """
    if raw_val is None:
        norm_val = "UNKNOWN"
    else:
        norm_val = str(raw_val).strip().upper()
        if not norm_val or norm_val in ("NONE", "NAN", "NULL"):
            norm_val = "UNKNOWN"

    if norm_val.startswith("DEBRIS"):
        return [1.0, 0.0, 0.0, 0.0]
    elif norm_val.startswith("PAYLOAD"):
        return [0.0, 1.0, 0.0, 0.0]
    elif norm_val in ("ROCKET BODY", "ROCKET_BODY", "ROCKETBODY"):
        return [0.0, 0.0, 1.0, 0.0]
    else:
        # UNKNOWN, TBA, or other unclassified
        return [0.0, 0.0, 0.0, 1.0]


class Phase3BSequencePreprocessor:
    """Channel-wise normalizer and sequence tensor pipeline for Phase 3B experiments.
    
    Strict zero-leakage: normalizer statistics are fit strictly on training observations.
    """

    def __init__(self, config: Union[Phase3BExperimentConfig, ExperimentID, str]) -> None:
        if isinstance(config, (ExperimentID, str)):
            self.config = get_experiment_config(config)
        else:
            self.config = config

        self.feature_columns = list(self.config.feature_columns)
        self.channel_stats_: Dict[str, Dict[str, Any]] = {}
        self.is_fitted_: bool = False

    def prepare_sequence_tensors(
        self,
        events: Dict[str, List[Dict[str, Any]]],
        horizon_cutoff: Optional[float] = None,
    ) -> Tuple[Any, Any, List[float], List[str]]:
        """Extract and construct 3D sequence tensors for Phase 3B experiments.
        
        Applies feature-level transformations (one-hot encoding, log10, dt extraction)
        before normalizer fitting.
        
        Output format:
        - X_tensor: [batch, n_channels, max_seq_len] (left-padded with zeros)
        - mask_tensor: [batch, max_seq_len] (1.0 for valid timesteps, 0.0 for padding)
        - target_risks: List of final target log-risk values
        - event_ids: List of valid event IDs
        """
        all_x: List[List[List[float]]] = []
        all_masks: List[List[float]] = []
        target_risks: List[float] = []
        valid_ids: List[str] = []

        max_seq_len = self.config.max_seq_len
        n_channels = len(self.feature_columns)

        for ev_id, cdms in events.items():
            if not cdms:
                continue

            # Filter qualifying CDMs (time_to_tca >= horizon_cutoff)
            qualifying = []
            for cdm in cdms:
                if horizon_cutoff is not None:
                    try:
                        tt = float(cdm.get("time_to_tca", -1.0))
                        if tt < horizon_cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue
                qualifying.append(cdm)

            if not qualifying:
                continue

            # Target is the risk of the final CDM of the entire event
            final_cdm = cdms[-1]
            try:
                risk_target = float(final_cdm["risk"])
            except (KeyError, ValueError, TypeError):
                continue

            # Truncate to max_seq_len if longer (keep newest qualifying CDMs)
            if len(qualifying) > max_seq_len:
                qualifying = qualifying[-max_seq_len:]

            seq_len = len(qualifying)
            pad_len = max_seq_len - seq_len

            # Feature matrix: [n_channels, max_seq_len]
            feat_matrix = [[0.0] * max_seq_len for _ in range(n_channels)]
            mask_row = [0.0] * pad_len + [1.0] * seq_len

            # Compute dt array if required (qualifying CDMs ordered oldest to newest)
            dt_values: List[float] = []
            if self.config.include_delta_t:
                for i, cdm in enumerate(qualifying):
                    if i == 0:
                        dt_values.append(0.0)
                    else:
                        try:
                            t_prev = float(qualifying[i - 1].get("time_to_tca", 0.0))
                            t_curr = float(cdm.get("time_to_tca", 0.0))
                            dt = max(t_prev - t_curr, 0.0)
                        except (ValueError, TypeError):
                            dt = 0.0
                        dt_values.append(dt)

            for t_idx, cdm in enumerate(qualifying):
                target_t = pad_len + t_idx
                ch_idx = 0

                # 1. Channel time_to_tca
                raw_t = cdm.get("time_to_tca")
                try:
                    fval_t = float(raw_t) if raw_t is not None else 0.0
                    feat_matrix[ch_idx][target_t] = 0.0 if math.isnan(fval_t) else fval_t
                except (ValueError, TypeError):
                    feat_matrix[ch_idx][target_t] = 0.0
                ch_idx += 1

                # 2. Optional Delta-t Channel
                if self.config.include_delta_t:
                    feat_matrix[ch_idx][target_t] = dt_values[t_idx]
                    ch_idx += 1

                # 3. Categorical Channel(s)
                if self.config.include_categorical_one_hot:
                    raw_obj = cdm.get("c_object_type")
                    one_hot = encode_c_object_type_one_hot(raw_obj)
                    for cat_val in one_hot:
                        feat_matrix[ch_idx][target_t] = cat_val
                        ch_idx += 1
                else:
                    # Legacy / broken c_object_type handling matching M0
                    raw_obj = cdm.get("c_object_type")
                    if raw_obj is not None:
                        try:
                            fval_obj = float(raw_obj)
                            feat_matrix[ch_idx][target_t] = 0.0 if math.isnan(fval_obj) else fval_obj
                        except (ValueError, TypeError):
                            feat_matrix[ch_idx][target_t] = 0.0
                    else:
                        feat_matrix[ch_idx][target_t] = 0.0
                    ch_idx += 1

                # 4. Remaining Numeric Features
                remaining_direct = [
                    c for c in DIRECT_FEATURE_COLUMNS
                    if c not in ("time_to_tca", "c_object_type")
                ]
                for col in remaining_direct:
                    raw_val = cdm.get(col)
                    if raw_val is not None:
                        try:
                            fval = float(raw_val)
                            if math.isnan(fval) or math.isinf(fval):
                                fval = 0.0
                            else:
                                if self.config.include_covariance_log10 and col in COVARIANCE_LOG10_FEATURE_COLUMNS:
                                    # Monotonic log10 transformation on positive features
                                    if fval > 0:
                                        fval = math.log10(fval)
                                    else:
                                        fval = 0.0
                            feat_matrix[ch_idx][target_t] = fval
                        except (ValueError, TypeError):
                            feat_matrix[ch_idx][target_t] = 0.0
                    else:
                        feat_matrix[ch_idx][target_t] = 0.0
                    ch_idx += 1

            all_x.append(feat_matrix)
            all_masks.append(mask_row)
            target_risks.append(risk_target)
            valid_ids.append(ev_id)

        if TORCH_AVAILABLE:
            X_tensor = torch.tensor(all_x, dtype=torch.float32)
            mask_tensor = torch.tensor(all_masks, dtype=torch.float32)
            return X_tensor, mask_tensor, target_risks, valid_ids

        return all_x, all_masks, target_risks, valid_ids

    def fit(self, X_train: Any, mask_train: Any) -> "Phase3BSequencePreprocessor":
        """Compute channel-wise mean and std strictly on valid observation timesteps of training split."""
        if hasattr(X_train, "cpu"):
            X_np = X_train.cpu().numpy()
            mask_np = mask_train.cpu().numpy()
        else:
            X_np = np.asarray(X_train, dtype=np.float64)
            mask_np = np.asarray(mask_train, dtype=np.float64)

        n_samples, n_channels, time_steps = X_np.shape
        valid_mask = mask_np > 0.5  # [N, T]

        self.channel_stats_ = {}
        for c in range(n_channels):
            feat_name = self.feature_columns[c] if c < len(self.feature_columns) else f"feat_{c}"
            channel_vals = X_np[:, c, :][valid_mask]
            finite_vals = channel_vals[np.isfinite(channel_vals)]

            is_one_hot = feat_name in OBJECT_TYPE_ONE_HOT_CHANNELS

            if is_one_hot:
                # One-hot binary indicators remain in {0.0, 1.0} on valid timesteps
                mean_val = 0.0
                std_val = 1.0
                median_val = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
                min_val = float(np.min(finite_vals)) if len(finite_vals) > 0 else 0.0
                max_val = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1.0
            else:
                if len(finite_vals) > 0:
                    mean_val = float(np.mean(finite_vals))
                    std_val = float(np.std(finite_vals))
                    std_val = max(std_val, 1e-4)
                    median_val = float(np.median(finite_vals))
                    min_val = float(np.min(finite_vals))
                    max_val = float(np.max(finite_vals))
                else:
                    mean_val = 0.0
                    std_val = 1.0
                    median_val = 0.0
                    min_val = 0.0
                    max_val = 1.0

            self.channel_stats_[feat_name] = {
                "channel_idx": c,
                "name": feat_name,
                "is_one_hot": is_one_hot,
                "is_log10": self.config.include_covariance_log10 and feat_name in COVARIANCE_LOG10_FEATURE_COLUMNS,
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "min": min_val,
                "max": max_val,
                "valid_samples_count": len(finite_vals),
            }

        self.is_fitted_ = True
        return self

    def transform(self, X: Any, mask: Any) -> Any:
        """Apply train-fitted normalization and zero out padding timesteps."""
        if not self.is_fitted_:
            raise RuntimeError("Phase3BSequencePreprocessor must be fitted before transforming.")

        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X_norm = X.clone()
            n_channels = X.size(1)
            for c in range(n_channels):
                feat_name = self.feature_columns[c] if c < len(self.feature_columns) else f"feat_{c}"
                stat = self.channel_stats_.get(feat_name, {"mean": 0.0, "std": 1.0})
                m = stat["mean"]
                s = stat["std"]
                X_norm[:, c, :] = (X_norm[:, c, :] - m) / s

            # Zero out padding positions (where mask == 0.0)
            mask_expanded = mask.unsqueeze(1).expand_as(X_norm)
            X_norm = X_norm * mask_expanded
            return X_norm

        X_np = np.array(X, copy=True, dtype=np.float64)
        mask_np = np.asarray(mask, dtype=np.float64)
        n_channels = X_np.shape[1]

        for c in range(n_channels):
            feat_name = self.feature_columns[c] if c < len(self.feature_columns) else f"feat_{c}"
            stat = self.channel_stats_.get(feat_name, {"mean": 0.0, "std": 1.0})
            m = stat["mean"]
            s = stat["std"]
            X_np[:, c, :] = (X_np[:, c, :] - m) / s

        mask_expanded = np.expand_dims(mask_np, axis=1)
        X_np = X_np * mask_expanded
        return X_np

    def fit_transform(self, X_train: Any, mask_train: Any) -> Any:
        """Fit on training sequence tensors and transform in a single step."""
        return self.fit(X_train, mask_train).transform(X_train, mask_train)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessor state and complete metadata manifest."""
        return {
            "experiment_id": self.config.experiment_id.value,
            "experiment_config": self.config.to_dict(),
            "n_channels": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "channel_stats": self.channel_stats_,
            "is_fitted": self.is_fitted_,
            "category_mapping": {
                "raw_categories": RAW_OBJECT_TYPE_CATEGORIES,
                "encoded_channels": OBJECT_TYPE_ONE_HOT_CHANNELS,
                "encoding_type": "binary_one_hot",
            },
            "covariance_transformation": {
                "type": "log10" if self.config.include_covariance_log10 else "none",
                "transformed_features": COVARIANCE_LOG10_FEATURE_COLUMNS if self.config.include_covariance_log10 else [],
            },
            "delta_t_transformation": {
                "included": self.config.include_delta_t,
                "source_feature": "time_to_tca",
                "physical_unit": "days",
                "initial_step_dt": 0.0,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase3BSequencePreprocessor":
        """Reconstruct preprocessor from dictionary manifest."""
        config = Phase3BExperimentConfig.from_dict(data["experiment_config"])
        instance = cls(config=config)
        instance.channel_stats_ = data["channel_stats"]
        instance.is_fitted_ = data["is_fitted"]
        instance.feature_columns = data["feature_columns"]
        return instance

    def save(self, path: Union[Path, str]) -> None:
        """Save preprocessor manifest to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "Phase3BSequencePreprocessor":
        """Load preprocessor state from JSON manifest file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
