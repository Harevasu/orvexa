"""Physics-derived baseline using ESA max_risk_estimate without machine learning."""

from pathlib import Path
from typing import Any


class PhysicsMaxRiskModel:
    """Baseline ranker and risk estimator derived directly from ESA max_risk_estimate."""

    def __init__(self, risk_col: str = "max_risk_estimate") -> None:
        self.risk_col = risk_col

    def fit(self, X_train: Any, y_train: Any, X_valid: Any = None, y_valid: Any = None) -> "PhysicsMaxRiskModel":
        """Physics baseline does not train."""
        return self

    def predict_risk(self, X: Any) -> Any:
        """Return raw max_risk_estimate as continuous risk predictor."""
        raise NotImplementedError("Physics risk prediction not yet implemented.")

    def save(self, path: Path | str) -> None:
        raise NotImplementedError("Model serialization not yet implemented.")

    @classmethod
    def load(cls, path: Path | str) -> "PhysicsMaxRiskModel":
        raise NotImplementedError("Model loading not yet implemented.")
