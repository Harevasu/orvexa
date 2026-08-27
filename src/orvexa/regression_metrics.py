"""Continuous regression metrics: MAE, RMSE, Median Absolute Error, and Spearman correlation."""

import math
from typing import Any, Dict, List


def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient between two continuous series."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(vals: List[float]) -> List[float]:
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_indices[j]] == vals[sorted_indices[j + 1]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-indexed
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom <= 1e-12:
        return 0.0
    return cov / denom


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two continuous series."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    denom = math.sqrt(var_x * var_y)
    if denom <= 1e-12:
        return 0.0
    return cov / denom


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
    """Compute MAE, RMSE, R^2, Pearson, Spearman rank correlation, and Median AE.
    
    Args:
        y_true: True continuous log-risk values.
        y_pred: Predicted continuous log-risk values.
        
    Returns:
        Dictionary of regression evaluation metrics.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}.")

    n = len(y_true)
    abs_errors = [abs(yt - yp) for yt, yp in zip(y_true, y_pred)]
    sq_errors = [(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)]

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)

    # R^2 calculation: 1 - (SS_res / SS_tot)
    mean_y_true = sum(y_true) / n
    ss_tot = sum((yt - mean_y_true) ** 2 for yt in y_true)
    ss_res = sum(sq_errors)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    sorted_abs = sorted(abs_errors)
    if n % 2 == 1:
        med_ae = sorted_abs[n // 2]
    else:
        med_ae = (sorted_abs[n // 2 - 1] + sorted_abs[n // 2]) / 2.0

    spearman_corr = _spearman_rank_correlation(y_true, y_pred)
    pearson_corr = _pearson_correlation(y_true, y_pred)

    return {
        "n_samples": n,
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 5),
        "pearson_correlation": round(pearson_corr, 5),
        "spearman_correlation": round(spearman_corr, 5),
        "median_absolute_error": round(med_ae, 5),
    }

