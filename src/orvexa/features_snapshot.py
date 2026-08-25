"""Snapshot feature extraction from the latest qualifying CDM before cutoff."""

from typing import Any, List


def extract_snapshot_features(event_df: Any, feature_cols: List[str]) -> Any:
    """Extract features from the single most recent CDM satisfying time_to_tca >= H."""
    raise NotImplementedError("Snapshot feature extraction not yet implemented.")
