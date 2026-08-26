"""Event-level bootstrap confidence interval estimation (2,000 iterations, seed 42).

Resampling is performed strictly at the EVENT level (not individual CDM rows) to preserve
intra-event correlation structures and prevent bootstrap optimism.
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional


def compute_event_bootstrap_ci(
    event_ids: List[str],
    y_true: List[float],
    y_pred: List[float],
    metric_fn: Callable[[List[float], List[float]], float],
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute percentile bootstrap confidence intervals by resampling conjunction events.
    
    Args:
        event_ids: List of unique event identifiers corresponding to y_true and y_pred.
        y_true: True target values.
        y_pred: Predicted values.
        metric_fn: Evaluation metric callable taking (y_true_sample, y_pred_sample) and returning float.
        n_iterations: Number of bootstrap iterations (default: 2000).
        confidence_level: Confidence level interval (default: 0.95 for 95% CI).
        seed: Random generator seed for reproducibility.
        
    Returns:
        Dictionary with point estimate, lower CI, upper CI, and bootstrap standard error.
    """
    if len(event_ids) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between event_ids, y_true, and y_pred.")

    n_events = len(event_ids)
    if n_events == 0:
        raise ValueError("Cannot bootstrap on empty data.")

    # Point estimate on full sample
    point_estimate = float(metric_fn(y_true, y_pred))

    rng = random.Random(seed)
    bootstrap_estimates: List[float] = []

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    # Index mapping for fast lookup
    for _ in range(n_iterations):
        # Sample event indices with replacement
        sample_indices = [rng.randrange(n_events) for _ in range(n_events)]
        sample_true = [y_true[idx] for idx in sample_indices]
        sample_pred = [y_pred[idx] for idx in sample_indices]

        try:
            val = float(metric_fn(sample_true, sample_pred))
            if not (math.isnan(val) or math.isinf(val)):
                bootstrap_estimates.append(val)
        except Exception:
            continue

    if not bootstrap_estimates:
        return {
            "point_estimate": point_estimate,
            "ci_lower": point_estimate,
            "ci_upper": point_estimate,
            "std_error": 0.0,
            "n_iterations": 0,
        }

    bootstrap_estimates.sort()
    n_valid = len(bootstrap_estimates)

    # Percentile method
    idx_lower = int(math.floor((lower_pct / 100.0) * n_valid))
    idx_upper = int(math.ceil((upper_pct / 100.0) * n_valid)) - 1
    idx_lower = min(max(idx_lower, 0), n_valid - 1)
    idx_upper = min(max(idx_upper, 0), n_valid - 1)

    ci_lower = bootstrap_estimates[idx_lower]
    ci_upper = bootstrap_estimates[idx_upper]

    mean_b = sum(bootstrap_estimates) / n_valid
    variance_b = sum((x - mean_b) ** 2 for x in bootstrap_estimates) / max(n_valid - 1, 1)
    std_error = math.sqrt(variance_b)

    return {
        "point_estimate": round(point_estimate, 5),
        "ci_lower": round(ci_lower, 5),
        "ci_upper": round(ci_upper, 5),
        "std_error": round(std_error, 5),
        "n_iterations": n_valid,
        "confidence_level": confidence_level,
    }
