"""Reporting utilities, metric summaries, and thesis figure generation."""

from pathlib import Path
from typing import Any, Dict


def generate_experiment_summary(metrics_dir: Path | str, output_path: Path | str) -> Any:
    """Collate metric JSON artifacts into structured summary tables."""
    raise NotImplementedError("Reporting not yet implemented.")


def plot_reliability_diagram(y_true: Any, y_prob: Any, output_path: Path | str) -> None:
    """Generate and save publication-quality reliability curve."""
    raise NotImplementedError("Plotting not yet implemented.")
