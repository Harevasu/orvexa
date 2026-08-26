"""Deterministic artifact storage and loading for models, preprocessors, metrics, and figures."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ArtifactStore:
    """Manages deterministic file paths, serialization, and registry for experiment outputs."""

    def __init__(self, root_dir: Union[Path, str] = "artifacts") -> None:
        self.root_dir = Path(root_dir)

    def get_model_dir(self, model_name: str, horizon: float) -> Path:
        """Standard directory for model artifacts."""
        p = self.root_dir / "models" / f"{model_name}_h{horizon:.1f}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_preprocessor_path(self, horizon: float) -> Path:
        """Standard path for fitted preprocessor JSON."""
        p = self.root_dir / "preprocessors"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"preprocessor_h{horizon:.1f}.json"

    def get_metrics_path(self, model_name: str, horizon: float, split: str = "test") -> Path:
        """Standard path for evaluation metrics JSON."""
        p = self.root_dir / "metrics"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{model_name}_h{horizon:.1f}_{split}.json"

    def get_splits_path(self) -> Path:
        """Standard path for split manifest JSON."""
        p = self.root_dir / "splits"
        p.mkdir(parents=True, exist_ok=True)
        return p / "split_manifest.json"

    def save_json(self, data: Dict[str, Any], path: Union[Path, str]) -> None:
        """Save dictionary to standardized JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: Union[Path, str]) -> Dict[str, Any]:
        """Load dictionary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
