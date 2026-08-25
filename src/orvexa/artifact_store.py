"""Deterministic artifact storage and loading for models, preprocessors, and evaluation outputs."""

from pathlib import Path
from typing import Any, Dict


class ArtifactStore:
    """Manages deterministic file paths and loading for models, preprocessors, metrics, and figures."""

    def __init__(self, root_dir: Path | str = "artifacts") -> None:
        self.root_dir = Path(root_dir)

    def get_model_path(self, experiment_id: str, horizon: float) -> Path:
        """Return standardized path for model checkpoint."""
        return self.root_dir / "models" / f"{experiment_id}_h{horizon:.1f}" / "model.pt"

    def get_metric_path(self, experiment_id: str, horizon: float, split: str = "test") -> Path:
        """Return standardized path for metric JSON."""
        return self.root_dir / "metrics" / f"{experiment_id}_h{horizon:.1f}_{split}.json"
