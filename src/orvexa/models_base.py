"""Abstract model protocols and base interfaces for ORVEXA predictors."""

from pathlib import Path
from typing import Any, Protocol


class RiskModel(Protocol):
    """Common protocol for all ORVEXA conjunction risk models."""

    def fit(self, X_train: Any, y_train: Any, X_valid: Any = None, y_valid: Any = None) -> "RiskModel":
        """Fit model on training split with optional early stopping on validation split."""
        ...

    def predict_risk(self, X: Any) -> Any:
        """Predict continuous log-risk score for each event."""
        ...

    def predict_probability(self, X: Any, threshold_log10: float) -> Any:
        """Predict probability that risk exceeds threshold_log10."""
        ...

    def save(self, path: Path | str) -> None:
        """Serialize model artifact to disk."""
        ...

    @classmethod
    def load(cls, path: Path | str) -> "RiskModel":
        """Load serialized model artifact from disk."""
        ...
