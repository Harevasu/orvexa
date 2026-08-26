"""Input perturbation and data degradation evaluation (masking, noise, sequence reduction)."""

import math
import random
from typing import Any, Callable, Dict, List, Optional

from orvexa.regression_metrics import compute_regression_metrics


def inject_gaussian_noise(
    X: List[List[float]], noise_scale: float = 0.10, seed: int = 42
) -> List[List[float]]:
    """Add zero-mean Gaussian noise proportional to column standard deviation."""
    if not X:
        return []
    n_samples = len(X)
    n_features = len(X[0])
    rng = random.Random(seed)

    # Compute column standard deviations
    col_stds: List[float] = []
    for j in range(n_features):
        vals = [X[i][j] for i in range(n_samples)]
        mean_v = sum(vals) / n_samples
        var_v = sum((v - mean_v) ** 2 for v in vals) / max(n_samples - 1, 1)
        col_stds.append(math.sqrt(var_v) if var_v > 1e-12 else 1.0)

    perturbed: List[List[float]] = []
    for i in range(n_samples):
        row: List[float] = []
        for j in range(n_features):
            # Box-Muller normal approximation
            u1 = max(rng.random(), 1e-10)
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            val = X[i][j] + z * noise_scale * col_stds[j]
            row.append(val)
        perturbed.append(row)

    return perturbed


def inject_feature_dropout(
    X: List[List[float]], drop_rate: float = 0.20, default_fill: float = 0.0, seed: int = 42
) -> List[List[float]]:
    """Randomly mask out feature entries to simulate sensor/telemetry dropouts."""
    if not X:
        return []
    rng = random.Random(seed)
    dropped: List[List[float]] = []
    for row in X:
        new_row = [default_fill if rng.random() < drop_rate else v for v in row]
        dropped.append(new_row)
    return dropped


def evaluate_robustness_perturbations(
    predict_fn: Callable[[List[List[float]]], List[float]],
    X_test: List[List[float]],
    y_test: List[float],
    noise_levels: Optional[List[float]] = None,
    dropout_rates: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate model prediction degradation under synthetic telemetry corruption.
    
    Args:
        predict_fn: Callable taking feature matrix and returning predicted risks.
        X_test: Clean test feature matrix.
        y_test: True test target risk values.
        noise_levels: List of Gaussian noise scales (default: [0.05, 0.10, 0.20]).
        dropout_rates: List of feature missingness drop rates (default: [0.10, 0.25]).
        seed: Random seed.
        
    Returns:
        Dictionary detailing baseline metrics and degraded metrics under each perturbation.
    """
    noise_levels = noise_levels or [0.05, 0.10, 0.20]
    dropout_rates = dropout_rates or [0.10, 0.25]

    # Baseline performance
    clean_preds = predict_fn(X_test)
    baseline_metrics = compute_regression_metrics(y_test, clean_preds)

    results: Dict[str, Any] = {
        "baseline": baseline_metrics,
        "gaussian_noise": {},
        "feature_dropout": {},
    }

    # Noise experiments
    for lvl in noise_levels:
        x_noisy = inject_gaussian_noise(X_test, noise_scale=lvl, seed=seed)
        noisy_preds = predict_fn(x_noisy)
        metrics = compute_regression_metrics(y_test, noisy_preds)
        results["gaussian_noise"][f"noise_{int(lvl*100)}pct"] = {
            "scale": lvl,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mae_delta": round(metrics["mae"] - baseline_metrics["mae"], 5),
        }

    # Feature dropout experiments
    for rate in dropout_rates:
        x_dropped = inject_feature_dropout(X_test, drop_rate=rate, seed=seed)
        drop_preds = predict_fn(x_dropped)
        metrics = compute_regression_metrics(y_test, drop_preds)
        results["feature_dropout"][f"drop_{int(rate*100)}pct"] = {
            "rate": rate,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mae_delta": round(metrics["mae"] - baseline_metrics["mae"], 5),
        }

    return results
