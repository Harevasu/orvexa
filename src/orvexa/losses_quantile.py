"""Loss functions and metrics for quantile regression and uncertainty evaluation.

References:
- Koenker, R., & Bassett, G. (1978). "Regression Quantiles", Econometrica.
- Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile Regression", NeurIPS.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def pinball_loss_scalar(y_true: float, y_pred: float, tau: float) -> float:
    """Compute scalar pinball (quantile) loss for a single observation and target quantile tau.
    
    Formula:
        L_tau(y, q) = max(tau * (y - q), (1 - tau) * (q - y))
        
    Args:
        y_true: True continuous target value.
        y_pred: Predicted quantile value q_tau.
        tau: Quantile level in (0, 1).
        
    Returns:
        Non-negative pinball loss value.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"Quantile tau must be in (0, 1), got {tau}")
    diff = y_true - y_pred
    return max(tau * diff, (tau - 1.0) * diff)


def pinball_loss_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Compute elementwise pinball loss for numpy arrays."""
    if not (0.0 < tau < 1.0):
        raise ValueError(f"Quantile tau must be in (0, 1), got {tau}")
    diff = y_true - y_pred
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def multi_quantile_pinball_loss_numpy(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: List[float],
) -> Tuple[float, Dict[float, float]]:
    """Compute mean pinball loss across multiple quantiles using numpy.
    
    Args:
        y_true: 1D array of true targets shape [N].
        y_pred_quantiles: 2D array of predictions shape [N, K].
        quantiles: List of K quantile levels in increasing order.
        
    Returns:
        Tuple of (mean_pinball_loss, dict_of_loss_per_quantile).
    """
    y_true_1d = np.asarray(y_true, dtype=float).ravel()
    y_pred_2d = np.asarray(y_pred_quantiles, dtype=float)
    if y_pred_2d.ndim == 1:
        y_pred_2d = y_pred_2d.reshape(-1, len(quantiles))
    
    losses_by_tau: Dict[float, float] = {}
    for k, tau in enumerate(quantiles):
        q_k = y_pred_2d[:, k]
        losses_by_tau[tau] = float(np.mean(pinball_loss_numpy(y_true_1d, q_k, tau)))
    
    mean_loss = float(np.mean(list(losses_by_tau.values())))
    return mean_loss, losses_by_tau


if TORCH_AVAILABLE:

    class MultiQuantileLoss(nn.Module):
        """PyTorch Module for Multi-Quantile Pinball Loss."""

        def __init__(self, quantiles: List[float]) -> None:
            super().__init__()
            for q in quantiles:
                if not (0.0 < q < 1.0):
                    raise ValueError(f"Invalid quantile level {q}; must be in (0, 1).")
            self.quantiles = sorted(quantiles)
            self.register_buffer(
                "tau_tensor",
                torch.tensor(self.quantiles, dtype=torch.float32).view(1, -1),
            )

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """Compute average pinball loss across all quantiles.
            
            Args:
                y_pred: Tensor of shape [batch_size, n_quantiles].
                y_true: Tensor of shape [batch_size, 1] or [batch_size].
                
            Returns:
                Scalar tensor representing the mean pinball loss.
            """
            if y_true.ndim == 1:
                y_true = y_true.unsqueeze(-1)
            # Expand true target to match predicted quantiles: [batch, K]
            diff = y_true - y_pred
            # tau * diff vs (tau - 1) * diff
            loss_k = torch.maximum(self.tau_tensor * diff, (self.tau_tensor - 1.0) * diff)
            return torch.mean(loss_k)


def compute_quantile_evaluation_metrics(
    y_true: Union[List[float], np.ndarray],
    y_pred_quantiles: Union[List[List[float]], np.ndarray],
    quantiles: List[float],
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic metrics for multi-quantile risk predictions.
    
    Calculates:
    - Pinball loss per quantile and average pinball loss
    - Quantile crossing count and rate
    - Empirical coverage and width for intervals (e.g. 90%, 80%, 50%)
    - Tail metrics on critical events (y >= -5.0)
    
    Args:
        y_true: True continuous risk values (length N).
        y_pred_quantiles: Predicted quantiles matrix (shape [N, K]).
        quantiles: Quantile levels (length K).
        
    Returns:
        Structured dictionary of evaluation metrics.
    """
    y_arr = np.asarray(y_true, dtype=float).ravel()
    q_arr = np.asarray(y_pred_quantiles, dtype=float)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(-1, len(quantiles))
    
    n_samples, n_quantiles = q_arr.shape
    if len(quantiles) != n_quantiles:
        raise ValueError(f"Quantiles length ({len(quantiles)}) != prediction shape ({q_arr.shape})")
    
    # 1. Pinball losses
    mean_pinball, pinball_by_q = multi_quantile_pinball_loss_numpy(y_arr, q_arr, quantiles)
    
    # 2. Quantile Crossing Analysis
    crossing_violations = 0
    total_adjacent_pairs = n_samples * (n_quantiles - 1)
    for k in range(n_quantiles - 1):
        diff_k = q_arr[:, k + 1] - q_arr[:, k]
        crossing_violations += int(np.sum(diff_k < -1e-6))
    
    crossing_rate = float(crossing_violations / max(total_adjacent_pairs, 1))
    
    # 3. Interval Coverage & Sharpness
    q_map = {round(q, 4): k for k, q in enumerate(quantiles)}
    
    intervals: Dict[str, Any] = {}
    standard_intervals = [
        ("90pct_interval", 0.05, 0.95),
        ("80pct_interval", 0.10, 0.90),
        ("50pct_interval", 0.25, 0.75),
    ]
    for name, q_low, q_high in standard_intervals:
        if q_low in q_map and q_high in q_map:
            idx_l = q_map[q_low]
            idx_h = q_map[q_high]
            low_vals = q_arr[:, idx_l]
            high_vals = q_arr[:, idx_h]
            
            covered = (y_arr >= low_vals) & (y_arr <= high_vals)
            empirical_coverage = float(np.mean(covered))
            widths = high_vals - low_vals
            
            intervals[name] = {
                "nominal_coverage": round(q_high - q_low, 4),
                "empirical_coverage": round(empirical_coverage, 5),
                "mean_width": round(float(np.mean(widths)), 5),
                "median_width": round(float(np.median(widths)), 5),
                "min_width": round(float(np.min(widths)), 5),
                "max_width": round(float(np.max(widths)), 5),
            }
    
    # 4. Critical-tail diagnostics (y >= -5.0)
    crit_mask = y_arr >= -5.0
    crit_count = int(np.sum(crit_mask))
    tail_diagnostics: Dict[str, Any] = {
        "critical_events_count": crit_count,
    }
    if crit_count > 0:
        y_crit = y_arr[crit_mask]
        q_crit = q_arr[crit_mask]
        
        # Median quantile (q=0.50) residual on tail
        if 0.50 in q_map:
            idx_med = q_map[0.50]
            med_pred = q_crit[:, idx_med]
            residuals = med_pred - y_crit
            tail_diagnostics["tail_q50_mean_residual"] = round(float(np.mean(residuals)), 5)
            tail_diagnostics["tail_q50_median_residual"] = round(float(np.median(residuals)), 5)
            tail_diagnostics["tail_q50_mae"] = round(float(np.mean(np.abs(residuals))), 5)
        
        # Tail coverage for 90% interval
        if 0.05 in q_map and 0.95 in q_map:
            l_crit = q_crit[:, q_map[0.05]]
            h_crit = q_crit[:, q_map[0.95]]
            tail_cov_90 = float(np.mean((y_crit >= l_crit) & (y_crit <= h_crit)))
            tail_diagnostics["tail_coverage_90pct"] = round(tail_cov_90, 5)
            tail_diagnostics["tail_mean_width_90pct"] = round(float(np.mean(h_crit - l_crit)), 5)

    return {
        "n_samples": n_samples,
        "n_quantiles": n_quantiles,
        "quantiles": quantiles,
        "mean_pinball_loss": round(mean_pinball, 6),
        "pinball_loss_per_quantile": {str(q): round(v, 6) for q, v in pinball_by_q.items()},
        "quantile_crossing_violations": crossing_violations,
        "quantile_crossing_rate": round(crossing_rate, 6),
        "intervals": intervals,
        "tail_diagnostics": tail_diagnostics,
    }
