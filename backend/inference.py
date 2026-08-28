"""Inference engine for ORVEXA Candidate C (Quantile M4 + CQR).

Strict Zero-Modification Governance:
- Frozen models, preprocessors, and calibrators are loaded strictly in read-only mode.
- Evaluates torch in no_grad() / eval() mode.
- Preprocessors execute transform() only (no fitting).
- Calibrators execute predict_intervals() only (no fitting or score alteration).
- Verifies SHA-256 hashes against artifacts/models/phase5/candidate_freeze_manifest.json.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from backend.data_service import data_service
from backend.schemas import (
    CQRIntervalOutput,
    InferenceResponse,
    ModelArtifactInfo,
    MultiHorizonInferenceResponse,
    QuantilesOutput,
    RiskClassificationOutput,
)
from orvexa.conformal import SplitConformalCalibrator
from orvexa.models_probabilistic import (
    DEFAULT_QUANTILES,
    QuantileTCNRiskModel,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor


def compute_file_sha256(file_path: Path) -> str:
    """Deterministically compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


class CandidateCInferenceEngine:
    """Read-only inference dispatcher for Phase 5 Candidate C."""

    HORIZON_MAP = {
        "H2": {"days": 2.0, "hours": 48.0, "file_suffix": "h2"},
        "H3": {"days": 3.0, "hours": 72.0, "file_suffix": "h3"},
        "H5": {"days": 5.0, "hours": 120.0, "file_suffix": "h5"},
        "H6": {"days": 6.0, "hours": 144.0, "file_suffix": "h6"},
    }

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent
        self.manifest_path = (
            self.workspace_root / "artifacts" / "models" / "phase5" / "candidate_freeze_manifest.json"
        )

        self.models: Dict[str, QuantileTCNRiskModel] = {}
        self.preprocessors: Dict[str, Phase3BSequencePreprocessor] = {}
        self.calibrators: Dict[str, SplitConformalCalibrator] = {}
        self.artifact_hashes: Dict[str, Dict[str, str]] = {}
        self._is_loaded: bool = False

    def load_artifacts(self) -> None:
        """Load and verify frozen candidate artifacts for H2, H3, H5, H6."""
        if self._is_loaded:
            return

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing freeze manifest: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        frozen = manifest.get("frozen_artifacts", {})

        for h_key, cfg in self.HORIZON_MAP.items():
            if h_key not in frozen:
                raise KeyError(f"Horizon '{h_key}' not in candidate freeze manifest.")

            h_manifest = frozen[h_key]

            # Model weights & config
            model_base = self.workspace_root / f"artifacts/models/phase5/tcn_quantile_M4_{cfg['file_suffix']}"
            pt_path = model_base.with_suffix(".pt")
            json_path = model_base.with_suffix(".json")

            if not pt_path.exists() or not json_path.exists():
                raise FileNotFoundError(f"Missing model files for {h_key}: {model_base}")

            # Calibrator & Preprocessor paths
            cal_path = self.workspace_root / h_manifest["cqr_calibrator"]
            preproc_path = self.workspace_root / h_manifest["preprocessor"]

            if not cal_path.exists():
                raise FileNotFoundError(f"Missing CQR calibrator for {h_key}: {cal_path}")
            if not preproc_path.exists():
                raise FileNotFoundError(f"Missing preprocessor for {h_key}: {preproc_path}")

            # Verify SHA-256 hashes
            weights_sha = compute_file_sha256(pt_path)
            cal_sha = compute_file_sha256(cal_path)
            preproc_sha = compute_file_sha256(preproc_path)

            if weights_sha != h_manifest["model_weights_sha256"]:
                raise ValueError(
                    f"Integrity error: {h_key} weights SHA-256 mismatch! "
                    f"Expected {h_manifest['model_weights_sha256']}, got {weights_sha}"
                )
            if cal_sha != h_manifest["cqr_calibrator_sha256"]:
                raise ValueError(f"Integrity error: {h_key} CQR calibrator SHA-256 mismatch!")
            if preproc_sha != h_manifest["preprocessor_sha256"]:
                raise ValueError(f"Integrity error: {h_key} preprocessor SHA-256 mismatch!")

            self.artifact_hashes[h_key] = {
                "weights": weights_sha,
                "calibrator": cal_sha,
                "preprocessor": preproc_sha,
            }

            # Load model in eval mode
            model = QuantileTCNRiskModel.load(str(model_base))
            if model.network_ is not None:
                model.network_.eval()
            self.models[h_key] = model

            # Load preprocessor (transform only)
            preproc = Phase3BSequencePreprocessor.load(str(preproc_path))
            self.preprocessors[h_key] = preproc

            # Load CQR calibrator (read-only)
            calibrator = SplitConformalCalibrator.load(str(cal_path))
            self.calibrators[h_key] = calibrator

        self._is_loaded = True

    def classify_risk(self, q50: float) -> RiskClassificationOutput:
        """Classify predicted log10 risk according to established aerospace research thresholds."""
        if q50 >= -4.0:
            return RiskClassificationOutput(
                level="CRITICAL / HIGH RISK",
                description=(
                    "Predicted collision risk log10(Pc) >= -4.0 (Pc >= 1e-4). "
                    "Exceeds standard space-agency threshold for operational collision avoidance consideration."
                ),
                log10_threshold="log10(Pc) >= -4.0",
            )
        elif q50 >= -6.0:
            return RiskClassificationOutput(
                level="MODERATE RISK",
                description=(
                    "Predicted collision risk -6.0 <= log10(Pc) < -4.0 (1e-6 <= Pc < 1e-4). "
                    "Heightened surveillance zone requiring tracking prioritization."
                ),
                log10_threshold="-6.0 <= log10(Pc) < -4.0",
            )
        elif q50 >= -15.0:
            return RiskClassificationOutput(
                level="LOW RISK",
                description=(
                    "Predicted collision risk -15.0 <= log10(Pc) < -6.0. "
                    "Conjunction geometry is well clear of high-probability collision envelope."
                ),
                log10_threshold="-15.0 <= log10(Pc) < -6.0",
            )
        else:
            return RiskClassificationOutput(
                level="NEGLIGIBLE / SAFE",
                description=(
                    "Predicted collision risk log10(Pc) < -15.0. "
                    "Nominal orbital passage with negligible statistical collision probability."
                ),
                log10_threshold="log10(Pc) < -15.0",
            )

    def predict_event(
        self,
        event_id: str,
        horizon: str = "H2",
        alpha: float = 0.10,
    ) -> InferenceResponse:
        """Perform real Candidate C inference on an allowed demo conjunction event."""
        if not self._is_loaded:
            self.load_artifacts()

        h_key = horizon.upper()
        if h_key not in self.HORIZON_MAP:
            raise ValueError(f"Invalid horizon '{horizon}'. Must be one of H2, H3, H5, H6.")

        cfg = self.HORIZON_MAP[h_key]
        cutoff_days = cfg["days"]

        # Fetch CDMs (strictly enforces quarantine on test IDs)
        cdms = data_service.get_event_cdms(event_id)
        total_cdms = len(cdms)

        preprocessor = self.preprocessors[h_key]
        model = self.models[h_key]
        calibrator = self.calibrators[h_key]

        # Prepare sequence tensor (filtering CDMs with time_to_tca >= cutoff_days)
        events_dict = {str(event_id): cdms}
        X_raw, mask, targets, valid_ids = preprocessor.prepare_sequence_tensors(
            events_dict, horizon_cutoff=cutoff_days
        )

        if not valid_ids or len(valid_ids) == 0:
            raise ValueError(
                f"Conjunction Event '{event_id}' has no observations prior to warning horizon {h_key} "
                f"({cutoff_days:.1f} days / {cfg['hours']:.0f} hours to TCA)."
            )

        # Count qualifying CDMs
        qualifying_cdms = int(sum(mask[0]))

        # Normalize sequence using training statistics (transform-only)
        X_norm = preprocessor.transform(X_raw, mask)

        # Run model inference in eval mode
        with torch.no_grad():
            quant_preds = model.predict_quantiles(X_norm, mask)  # Shape: [1, 7]

        q_arr = quant_preds[0]  # [q05, q10, q25, q50, q75, q90, q95]

        # Monotonicity check
        for i in range(len(q_arr) - 1):
            if q_arr[i] > q_arr[i + 1] + 1e-5:
                raise RuntimeError(
                    f"Quantile monotonicity violation: q[{i}] ({q_arr[i]}) > q[{i+1}] ({q_arr[i+1]})"
                )

        quantiles_out = QuantilesOutput(
            q05=round(float(q_arr[0]), 4),
            q10=round(float(q_arr[1]), 4),
            q25=round(float(q_arr[2]), 4),
            q50=round(float(q_arr[3]), 4),
            q75=round(float(q_arr[4]), 4),
            q90=round(float(q_arr[5]), 4),
            q95=round(float(q_arr[6]), 4),
        )

        # Apply CQR calibrator: interval [q05 - q_hat, q95 + q_hat]
        q_low = np.array([q_arr[0]])
        q_high = np.array([q_arr[6]])
        lower_bounds, upper_bounds = calibrator.predict_intervals((q_low, q_high), alpha=alpha)

        q_hat = calibrator.get_conformal_quantile(alpha)
        cqr_low = float(lower_bounds[0])
        cqr_high = float(upper_bounds[0])
        cqr_width = cqr_high - cqr_low

        cqr_out = CQRIntervalOutput(
            lower=round(cqr_low, 4),
            upper=round(cqr_high, 4),
            width=round(cqr_width, 4),
            confidence_level=round(1.0 - alpha, 4),
            target_alpha=round(alpha, 4),
            conformal_shift_qhat=round(float(q_hat), 4),
        )

        risk_out = self.classify_risk(quantiles_out.q50)

        hashes = self.artifact_hashes[h_key]
        model_info = ModelArtifactInfo(
            candidate_id="Candidate_C_QuantileM4_CQR",
            architecture="QuantileMaskedCausalTCN (37-channel M4)",
            horizon=h_key,
            lead_time_hours=cfg["hours"],
            model_weights_sha256=hashes["weights"],
            preprocessor_sha256=hashes["preprocessor"],
            cqr_calibrator_sha256=hashes["calibrator"],
        )

        return InferenceResponse(
            event_id=str(event_id),
            horizon=h_key,
            horizon_days=cutoff_days,
            lead_time_hours=cfg["hours"],
            qualifying_cdms_count=qualifying_cdms,
            total_cdms_count=total_cdms,
            quantiles=quantiles_out,
            cqr_interval=cqr_out,
            risk_assessment=risk_out,
            model_info=model_info,
        )

    def predict_multi_horizon(
        self,
        event_id: str,
        alpha: float = 0.10,
    ) -> MultiHorizonInferenceResponse:
        """Run inference across all 4 operational horizons for side-by-side comparison."""
        results: Dict[str, Optional[InferenceResponse]] = {}
        for h_key in ["H2", "H3", "H5", "H6"]:
            try:
                res = self.predict_event(event_id, horizon=h_key, alpha=alpha)
                results[h_key] = res
            except Exception:
                results[h_key] = None

        return MultiHorizonInferenceResponse(
            event_id=str(event_id),
            horizons=results,
        )


# Global singleton instance
inference_engine = CandidateCInferenceEngine()
