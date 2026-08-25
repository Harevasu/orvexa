"""Regularized Ridge and Logistic Regression baseline models."""

from pathlib import Path
from typing import Any, Dict


class LinearRiskModel:
    """Ridge regression for continuous log-risk and Logistic regression for thresholded classification."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.regressor_ = None
        self.classifier_ = None

    def fit(self, X_train: Any, y_train: Any, X_valid: Any = None, y_valid: Any = None) -> "LinearRiskModel":
        raise NotImplementedError("Linear model fitting not yet implemented.")

    def predict_risk(self, X: Any) -> Any:
        raise NotImplementedError("Linear model risk prediction not yet implemented.")

    def save(self, path: Path | str) -> None:
        raise NotImplementedError("Model serialization not yet implemented.")

    @classmethod
    def load(cls, path: Path | str) -> "LinearRiskModel":
        raise NotImplementedError("Model loading not yet implemented.")
