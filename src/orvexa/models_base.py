"""Abstract model protocols and base interfaces for ORVEXA predictors."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union


@dataclass
class RiskPredictionResult:
    """Standardized prediction container for ORVEXA model outputs."""

    event_ids: List[str]
    risk_scores: List[float]
    model_name: str
    horizon_days: float
    probabilities: Dict[float, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to serializable dictionary."""
        return {
            "event_ids": self.event_ids,
            "risk_scores": self.risk_scores,
            "model_name": self.model_name,
            "horizon_days": self.horizon_days,
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskPredictionResult":
        """Reconstruct prediction result from dictionary."""
        probs = {float(k): v for k, v in data.get("probabilities", {}).items()}
        return cls(
            event_ids=data["event_ids"],
            risk_scores=data["risk_scores"],
            model_name=data["model_name"],
            horizon_days=float(data["horizon_days"]),
            probabilities=probs,
            metadata=data.get("metadata", {}),
        )


class RiskModel(Protocol):
    """Common protocol for all ORVEXA conjunction risk models."""

    model_name: str

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Optional[Any] = None,
        y_valid: Optional[Any] = None,
    ) -> "RiskModel":
        """Fit model on training split with optional early stopping on validation split."""
        ...

    def predict_risk(self, X: Any) -> List[float]:
        """Predict continuous log-risk score for each event."""
        ...

    def predict_probability(self, X: Any, threshold_log10: float) -> List[float]:
        """Predict probability that risk exceeds threshold_log10."""
        ...

    def save(self, path: Union[Path, str]) -> None:
        """Serialize model artifact to disk."""
        ...

    @classmethod
    def load(cls, path: Union[Path, str]) -> "RiskModel":
        """Load serialized model artifact from disk."""
        ...
