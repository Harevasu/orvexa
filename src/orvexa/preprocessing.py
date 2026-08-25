"""Train-fitted preprocessing pipelines: imputation, scaling, encoding, and missingness tracking."""

from typing import Any, Dict


class TrainFittedPreprocessor:
    """Preprocesses features fitted strictly on training data."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.fitted_ = False

    def fit(self, X_train: Any) -> "TrainFittedPreprocessor":
        """Fit transformers strictly on training events."""
        raise NotImplementedError("Fitting not yet implemented.")

    def transform(self, X: Any) -> Any:
        """Transform data using train-fitted transformers."""
        raise NotImplementedError("Transform not yet implemented.")
