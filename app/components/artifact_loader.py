"""Artifact loading utilities for precomputed results with actionable empty states."""

from pathlib import Path
from typing import Any, Dict


def load_metrics_artifact(artifact_path: str | Path) -> Dict[str, Any] | None:
    """Load JSON metrics artifact or return None if absent."""
    path = Path(artifact_path)
    if not path.exists():
        return None
    raise NotImplementedError("Artifact loading not yet implemented.")
