"""Warning horizon definitions and cutoff filtering logic."""

from typing import Any, List

DEFAULT_HORIZONS: List[float] = [2.0, 3.0, 5.0, 7.0]


def filter_by_horizon(df: Any, horizon_days: float, cutoff_col: str = "time_to_tca") -> Any:
    """Filter rows satisfying cutoff_col >= horizon_days."""
    raise NotImplementedError("Horizon filtering not yet implemented.")
