"""Train-fitted preprocessing pipelines: imputation, scaling, encoding, and missingness tracking.

Enforces strict zero-leakage constraints: transformers are fit solely on training partitions.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ColumnStats:
    """Train-fitted statistics for a single feature column."""

    name: str
    dtype: str  # 'numeric' or 'categorical'
    median: float = 0.0
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    categories: List[str] = field(default_factory=list)
    missing_count: int = 0
    total_count: int = 0


class TrainFittedPreprocessor:
    """Preprocesses features fitted strictly on training events.
    
    Guarantees:
    1. Zero leakage: No validation or test data ever updates preprocessor statistics.
    2. Missingness indicators: Explicit binary indicators for imputed values.
    3. Deterministic output: Reproducible feature ordering and normalization.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        add_missing_indicators: bool = True,
        scale_numeric: bool = True,
    ) -> None:
        self.numeric_features = list(numeric_features or [])
        self.categorical_features = list(categorical_features or [])
        self.add_missing_indicators = add_missing_indicators
        self.scale_numeric = scale_numeric
        self.stats_: Dict[str, ColumnStats] = {}
        self.output_feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, records: List[Dict[str, Any]]) -> "TrainFittedPreprocessor":
        """Compute statistics strictly on training data records."""
        if not records:
            raise ValueError("Cannot fit preprocessor on an empty record list.")

        self.stats_ = {}
        n_records = len(records)

        # 1. Fit Categorical Features
        for cat_col in self.categorical_features:
            cat_counts: Dict[str, int] = {}
            for rec in records:
                val = str(rec.get(cat_col, "UNKNOWN")).strip().upper()
                if not val or val == "NONE" or val == "NAN":
                    val = "UNKNOWN"
                cat_counts[val] = cat_counts.get(val, 0) + 1
            
            # Sort categories deterministically
            sorted_cats = sorted(cat_counts.keys())
            if "UNKNOWN" not in sorted_cats:
                sorted_cats.append("UNKNOWN")

            self.stats_[cat_col] = ColumnStats(
                name=cat_col,
                dtype="categorical",
                categories=sorted_cats,
                total_count=n_records,
            )

        # 2. Fit Numeric Features
        for num_col in self.numeric_features:
            values: List[float] = []
            missing_count = 0
            for rec in records:
                raw_val = rec.get(num_col)
                if raw_val is None:
                    missing_count += 1
                    continue
                try:
                    fval = float(raw_val)
                    if math.isnan(fval) or math.isinf(fval):
                        missing_count += 1
                    else:
                        values.append(fval)
                except (ValueError, TypeError):
                    missing_count += 1

            if not values:
                # All values missing in training set - default fallback
                median_val = 0.0
                mean_val = 0.0
                std_val = 1.0
                min_v = 0.0
                max_v = 1.0
            else:
                values.sort()
                n_v = len(values)
                if n_v % 2 == 1:
                    median_val = values[n_v // 2]
                else:
                    median_val = (values[n_v // 2 - 1] + values[n_v // 2]) / 2.0

                mean_val = sum(values) / n_v
                variance = sum((x - mean_val) ** 2 for x in values) / max(n_v - 1, 1)
                std_val = math.sqrt(variance)
                if std_val < 1e-8:
                    std_val = 1.0
                min_v = values[0]
                max_v = values[-1]

            self.stats_[num_col] = ColumnStats(
                name=num_col,
                dtype="numeric",
                median=median_val,
                mean=mean_val,
                std=std_val,
                min_val=min_v,
                max_val=max_v,
                missing_count=missing_count,
                total_count=n_records,
            )

        # Build output feature schema
        output_names: List[str] = []
        for num_col in self.numeric_features:
            output_names.append(num_col)
            if self.add_missing_indicators:
                output_names.append(f"{num_col}_is_missing")

        for cat_col in self.categorical_features:
            for cat in self.stats_[cat_col].categories:
                output_names.append(f"{cat_col}__{cat}")

        self.output_feature_names_ = output_names
        self.is_fitted_ = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> List[List[float]]:
        """Transform records into a 2D float feature matrix using train-fitted statistics."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fit before transforming data.")

        matrix: List[List[float]] = []
        for rec in records:
            row: List[float] = []

            # Process Numeric Columns
            for num_col in self.numeric_features:
                stats = self.stats_[num_col]
                raw_val = rec.get(num_col)
                is_missing = 0.0

                if raw_val is None:
                    val = stats.median
                    is_missing = 1.0
                else:
                    try:
                        fval = float(raw_val)
                        if math.isnan(fval) or math.isinf(fval):
                            val = stats.median
                            is_missing = 1.0
                        else:
                            val = fval
                    except (ValueError, TypeError):
                        val = stats.median
                        is_missing = 1.0

                if self.scale_numeric:
                    normalized = (val - stats.mean) / stats.std
                else:
                    normalized = val

                row.append(normalized)
                if self.add_missing_indicators:
                    row.append(is_missing)

            # Process Categorical Columns (One-Hot Encoded)
            for cat_col in self.categorical_features:
                stats = self.stats_[cat_col]
                raw_val = str(rec.get(cat_col, "UNKNOWN")).strip().upper()
                if not raw_val or raw_val == "NONE" or raw_val == "NAN":
                    raw_val = "UNKNOWN"

                for cat in stats.categories:
                    row.append(1.0 if raw_val == cat else 0.0)

            matrix.append(row)

        return matrix

    def fit_transform(self, records: List[Dict[str, Any]]) -> List[List[float]]:
        """Fit on records and transform in a single step."""
        return self.fit(records).transform(records)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted preprocessor state to dictionary."""
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "add_missing_indicators": self.add_missing_indicators,
            "scale_numeric": self.scale_numeric,
            "is_fitted": self.is_fitted_,
            "output_feature_names": self.output_feature_names_,
            "stats": {
                k: {
                    "name": s.name,
                    "dtype": s.dtype,
                    "median": s.median,
                    "mean": s.mean,
                    "std": s.std,
                    "min_val": s.min_val,
                    "max_val": s.max_val,
                    "categories": s.categories,
                    "missing_count": s.missing_count,
                    "total_count": s.total_count,
                }
                for k, s in self.stats_.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainFittedPreprocessor":
        """Reconstruct preprocessor from dictionary state."""
        instance = cls(
            numeric_features=data["numeric_features"],
            categorical_features=data["categorical_features"],
            add_missing_indicators=data["add_missing_indicators"],
            scale_numeric=data["scale_numeric"],
        )
        instance.is_fitted_ = data["is_fitted"]
        instance.output_feature_names_ = data["output_feature_names"]
        instance.stats_ = {}
        for k, v in data.get("stats", {}).items():
            instance.stats_[k] = ColumnStats(
                name=v["name"],
                dtype=v["dtype"],
                median=v.get("median", 0.0),
                mean=v.get("mean", 0.0),
                std=v.get("std", 1.0),
                min_val=v.get("min_val", 0.0),
                max_val=v.get("max_val", 1.0),
                categories=v.get("categories", []),
                missing_count=v.get("missing_count", 0),
                total_count=v.get("total_count", 0),
            )
        return instance

    def save(self, path: Union[Path, str]) -> None:
        """Save preprocessor state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "TrainFittedPreprocessor":
        """Load preprocessor state from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class TrainFittedSequencePreprocessor:
    """Channel-wise normalizer for 3D sequence tensors fitted strictly on training observations."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)
        self.channel_stats_: Dict[str, Dict[str, float]] = {}
        self.is_fitted_: bool = False

    def fit(self, X_train: Any, mask_train: Any) -> "TrainFittedSequencePreprocessor":
        """Compute channel-wise mean and std on valid observation timesteps of training split."""
        import numpy as np

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
            feat_name = self.feature_names[c] if c < len(self.feature_names) else f"feat_{c}"
            channel_vals = X_np[:, c, :][valid_mask]
            # Remove NaNs if any
            finite_vals = channel_vals[np.isfinite(channel_vals)]
            if len(finite_vals) > 0:
                mean_val = float(np.mean(finite_vals))
                std_val = float(np.std(finite_vals))
                std_val = max(std_val, 1e-4)
            else:
                mean_val = 0.0
                std_val = 1.0

            self.channel_stats_[feat_name] = {
                "channel_idx": c,
                "mean": mean_val,
                "std": std_val,
            }

        self.is_fitted_ = True
        return self

    def transform(self, X: Any, mask: Any) -> Any:
        """Apply train-fitted mean/std normalization and zero out padding timesteps."""
        if not self.is_fitted_:
            raise RuntimeError("TrainFittedSequencePreprocessor must be fitted before transforming.")

        try:
            import torch

            if isinstance(X, torch.Tensor):
                X_norm = X.clone()
                n_channels = X.size(1)
                for c in range(n_channels):
                    feat_name = self.feature_names[c] if c < len(self.feature_names) else f"feat_{c}"
                    stat = self.channel_stats_.get(feat_name, {"mean": 0.0, "std": 1.0})
                    m = stat["mean"]
                    s = stat["std"]
                    X_norm[:, c, :] = (X_norm[:, c, :] - m) / s

                # Zero out padding positions (where mask == 0.0)
                mask_expanded = mask.unsqueeze(1).expand_as(X_norm)
                X_norm = X_norm * mask_expanded
                return X_norm
        except ImportError:
            pass

        import numpy as np

        X_np = np.array(X, copy=True, dtype=np.float64)
        mask_np = np.asarray(mask, dtype=np.float64)
        n_channels = X_np.shape[1]

        for c in range(n_channels):
            feat_name = self.feature_names[c] if c < len(self.feature_names) else f"feat_{c}"
            stat = self.channel_stats_.get(feat_name, {"mean": 0.0, "std": 1.0})
            m = stat["mean"]
            s = stat["std"]
            X_np[:, c, :] = (X_np[:, c, :] - m) / s

        mask_expanded = np.expand_dims(mask_np, axis=1)
        X_np = X_np * mask_expanded
        return X_np

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "channel_stats": self.channel_stats_,
            "is_fitted": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainFittedSequencePreprocessor":
        instance = cls(feature_names=data["feature_names"])
        instance.channel_stats_ = data["channel_stats"]
        instance.is_fitted_ = data["is_fitted"]
        return instance

    def save(self, path: Union[Path, str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "TrainFittedSequencePreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

