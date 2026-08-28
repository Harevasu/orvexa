"""Inductive Split Conformal Prediction Calibrator for Conjunction Risk Intervals.

Implements distribution-free, finite-sample calibrated prediction intervals
guaranteed under exchangeability.

References:
- Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World", Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  "Distribution-free Predictive Inference for Regression", JASA.
- Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction", arXiv:2107.07511.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class SplitConformalCalibrator:
    """Inductive Split Conformal Prediction Calibrator for continuous risk regression.
    
    Provides exact finite-sample coverage guarantees P(Y in C(X)) >= 1 - alpha.
    """

    def __init__(
        self,
        default_alpha: float = 0.10,
        score_type: str = "absolute_residual",
    ) -> None:
        """Initialize split conformal calibrator.
        
        Args:
            default_alpha: Default miscoverage rate in (0, 1) (e.g. 0.10 for 90% coverage).
            score_type: Nonconformity score type:
                - 'absolute_residual': s_i = |y_i - y_hat_i| (standard split conformal)
                - 'cqr': s_i = max(q_low_i - y_i, y_i - q_high_i) (conformalized quantile regression)
        """
        if not (0.0 < default_alpha < 1.0):
            raise ValueError(f"default_alpha must be in (0, 1), got {default_alpha}")
        if score_type not in ("absolute_residual", "cqr"):
            raise ValueError(f"Unknown score_type: {score_type}. Must be 'absolute_residual' or 'cqr'.")

        self.default_alpha = default_alpha
        self.score_type = score_type

        self.calibration_scores_: Optional[np.ndarray] = None
        self.n_calibration_samples_: int = 0
        self.is_calibrated_: bool = False

    def compute_nonconformity_scores(
        self,
        y_true: Union[List[float], np.ndarray],
        y_pred: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Compute nonconformity scores for calibration or test samples.
        
        Args:
            y_true: Array of true target values [N].
            y_pred: Predicted point estimates [N] or (q_low, q_high) tuple for CQR.
            
        Returns:
            1D array of nonconformity scores [N].
        """
        y_arr = np.asarray(y_true, dtype=float).ravel()

        if self.score_type == "absolute_residual":
            pred_arr = np.asarray(y_pred, dtype=float).ravel()
            if len(y_arr) != len(pred_arr):
                raise ValueError(f"Shape mismatch: y_true ({len(y_arr)}) vs y_pred ({len(pred_arr)})")
            scores = np.abs(y_arr - pred_arr)
        elif self.score_type == "cqr":
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                q_low, q_high = np.asarray(y_pred[0], dtype=float).ravel(), np.asarray(y_pred[1], dtype=float).ravel()
            elif isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] >= 2:
                q_low, q_high = y_pred[:, 0], y_pred[:, -1]
            else:
                raise ValueError("CQR requires a tuple of (q_low, q_high) or 2D array [N, 2].")
            
            under_error = q_low - y_arr
            over_error = y_arr - q_high
            scores = np.maximum(under_error, over_error)
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")

        return scores

    def fit(
        self,
        y_calibration: Union[List[float], np.ndarray],
        y_pred_calibration: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ) -> "SplitConformalCalibrator":
        """Calibrate nonconformity scores strictly on Phase 5 Calibration partition.
        
        Args:
            y_calibration: True risk targets from calibration partition.
            y_pred_calibration: Model predictions on calibration partition.
            
        Returns:
            Self (calibrated instance).
        """
        scores = self.compute_nonconformity_scores(y_calibration, y_pred_calibration)
        self.calibration_scores_ = np.sort(scores)
        self.n_calibration_samples_ = len(self.calibration_scores_)

        if self.n_calibration_samples_ == 0:
            raise ValueError("Calibration set cannot be empty.")

        self.is_calibrated_ = True
        return self

    def get_conformal_quantile(self, alpha: Optional[float] = None) -> float:
        """Compute the exact finite-sample conformal critical value q_hat at level 1 - alpha.
        
        Formula:
            k = ceil((n_cal + 1) * (1 - alpha))
            q_hat = sorted_scores[k - 1] (capped at n_cal)
            
        Args:
            alpha: Miscoverage rate in (0, 1). If None, uses self.default_alpha.
            
        Returns:
            Conformal quantile critical value.
        """
        if not self.is_calibrated_ or self.calibration_scores_ is None:
            raise RuntimeError("Calibrator has not been calibrated. Call fit() first.")

        a = alpha if alpha is not None else self.default_alpha
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {a}")

        n = self.n_calibration_samples_
        # Exact finite sample index rule
        k = int(math.ceil((n + 1) * (1.0 - a)))

        if k > n:
            # When (n+1)(1-a) exceeds n, use maximum score
            return float(self.calibration_scores_[-1])
        elif k <= 0:
            return float(self.calibration_scores_[0])
        else:
            return float(self.calibration_scores_[k - 1])

    def predict_intervals(
        self,
        y_pred: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate calibrated prediction intervals [lower, upper] at level 1 - alpha.
        
        Args:
            y_pred: Point estimates or (q_low, q_high) bounds.
            alpha: Miscoverage rate in (0, 1).
            
        Returns:
            Tuple of (lower_bounds, upper_bounds) numpy arrays.
        """
        q_conf = self.get_conformal_quantile(alpha)

        if self.score_type == "absolute_residual":
            pred_arr = np.asarray(y_pred, dtype=float).ravel()
            lower = pred_arr - q_conf
            upper = pred_arr + q_conf
        elif self.score_type == "cqr":
            if isinstance(y_pred, tuple):
                q_low = np.asarray(y_pred[0], dtype=float).ravel()
                q_high = np.asarray(y_pred[1], dtype=float).ravel()
            elif isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
                q_low = y_pred[:, 0]
                q_high = y_pred[:, -1]
            else:
                raise ValueError("CQR requires a tuple of (q_low, q_high) or 2D array.")
            lower = q_low - q_conf
            upper = q_high + q_conf
        else:
            raise ValueError(f"Unsupported score_type: {self.score_type}")

        return lower, upper

    def evaluate_coverage(
        self,
        y_true: Union[List[float], np.ndarray],
        y_pred: Union[List[float], np.ndarray, Tuple[np.ndarray, np.ndarray]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate empirical coverage and interval sharpness.
        
        Args:
            y_true: True continuous targets [N].
            y_pred: Model predictions [N] or bounds.
            alpha: Target miscoverage rate.
            
        Returns:
            Dictionary with coverage statistics.
        """
        a = alpha if alpha is not None else self.default_alpha
        lower, upper = self.predict_intervals(y_pred, alpha=a)
        y_arr = np.asarray(y_true, dtype=float).ravel()

        covered = (y_arr >= lower) & (y_arr <= upper)
        empirical_coverage = float(np.mean(covered))
        widths = upper - lower

        # Tail evaluation on critical events (y >= -5.0)
        crit_mask = y_arr >= -5.0
        crit_count = int(np.sum(crit_mask))
        tail_coverage = None
        tail_mean_width = None
        if crit_count > 0:
            tail_coverage = float(np.mean(covered[crit_mask]))
            tail_mean_width = float(np.mean(widths[crit_mask]))

        q_conf = self.get_conformal_quantile(a)

        return {
            "nominal_confidence_level": round(1.0 - a, 4),
            "target_alpha": round(a, 4),
            "n_calibration_samples": self.n_calibration_samples_,
            "n_evaluation_samples": len(y_arr),
            "conformal_quantile_q_hat": round(q_conf, 5),
            "empirical_coverage": round(empirical_coverage, 5),
            "mean_interval_width": round(float(np.mean(widths)), 5),
            "median_interval_width": round(float(np.median(widths)), 5),
            "min_interval_width": round(float(np.min(widths)), 5),
            "max_interval_width": round(float(np.max(widths)), 5),
            "critical_events_count": crit_count,
            "critical_events_coverage": round(tail_coverage, 5) if tail_coverage is not None else None,
            "critical_events_mean_width": round(tail_mean_width, 5) if tail_mean_width is not None else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator configuration and score summaries."""
        return {
            "default_alpha": self.default_alpha,
            "score_type": self.score_type,
            "n_calibration_samples": self.n_calibration_samples_,
            "is_calibrated": self.is_calibrated_,
            "calibration_score_summary": {
                "min": round(float(np.min(self.calibration_scores_)), 6) if self.is_calibrated_ else None,
                "median": round(float(np.median(self.calibration_scores_)), 6) if self.is_calibrated_ else None,
                "p90": round(self.get_conformal_quantile(0.10), 6) if self.is_calibrated_ else None,
                "p95": round(self.get_conformal_quantile(0.05), 6) if self.is_calibrated_ else None,
                "max": round(float(np.max(self.calibration_scores_)), 6) if self.is_calibrated_ else None,
            } if self.is_calibrated_ else {},
        }

    def save(self, path: Union[Path, str]) -> None:
        """Save calibrator state to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if self.is_calibrated_ and self.calibration_scores_ is not None:
            data["calibration_scores"] = [round(float(s), 6) for s in self.calibration_scores_]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "SplitConformalCalibrator":
        """Load calibrator state from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        calibrator = cls(
            default_alpha=data.get("default_alpha", 0.10),
            score_type=data.get("score_type", "absolute_residual"),
        )
        if data.get("is_calibrated", False) and "calibration_scores" in data:
            calibrator.calibration_scores_ = np.array(data["calibration_scores"], dtype=float)
            calibrator.n_calibration_samples_ = len(calibrator.calibration_scores_)
            calibrator.is_calibrated_ = True

        return calibrator
