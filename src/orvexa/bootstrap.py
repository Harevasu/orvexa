"""Event-level bootstrap confidence interval estimation (2,000 iterations, seed 42)."""

from typing import Any, Callable, Dict


def compute_event_bootstrap_ci(
    event_ids: Any,
    y_true: Any,
    y_pred: Any,
    metric_fn: Callable[[Any, Any], float],
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute percentile confidence intervals by resampling unique conjunction event IDs."""
    raise NotImplementedError("Bootstrap estimation not yet implemented.")
