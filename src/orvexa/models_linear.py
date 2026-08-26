"""Regularized Ridge Regression and Logistic Regression baseline models."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _solve_ridge_analytical(
    X: List[List[float]], y: List[float], alpha: float = 1.0
) -> Tuple[List[float], float]:
    """Solve Ridge Regression w = (X^T X + alpha*I)^-1 X^T y with intercept."""
    n_samples = len(X)
    n_features = len(X[0])

    # Add bias column of 1.0s
    # Center y and X for numerical stability
    y_mean = sum(y) / n_samples
    y_centered = [yi - y_mean for yi in y]

    x_means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    X_centered = [[X[i][j] - x_means[j] for j in range(n_features)] for i in range(n_samples)]

    # Compute XtX (n_features x n_features)
    XtX = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(i, n_features):
            val = sum(X_centered[k][i] * X_centered[k][j] for k in range(n_samples))
            XtX[i][j] = val
            XtX[j][i] = val
        # Add ridge penalty to diagonal
        XtX[i][i] += alpha

    # Compute Xty (n_features x 1)
    Xty = [sum(X_centered[k][j] * y_centered[k] for k in range(n_samples)) for j in range(n_features)]

    # Solve linear system XtX * w = Xty using Gaussian elimination with partial pivoting
    # Augmented matrix
    aug = [XtX[i] + [Xty[i]] for i in range(n_features)]

    for i in range(n_features):
        # Pivot
        max_row = i
        for k in range(i + 1, n_features):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]

        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            pivot = 1e-12

        for j in range(i, n_features + 1):
            aug[i][j] /= pivot

        for k in range(n_features):
            if k != i:
                factor = aug[k][i]
                for j in range(i, n_features + 1):
                    aug[k][j] -= factor * aug[i][j]

    weights = [aug[i][n_features] for i in range(n_features)]
    intercept = y_mean - sum(weights[j] * x_means[j] for j in range(n_features))

    return weights, intercept


class LinearRiskModel:
    """Ridge regression baseline for continuous log-risk and thresholded risk probability."""

    model_name: str = "linear_ridge"

    def __init__(
        self,
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.alpha = alpha
        self.feature_names = feature_names or []
        self.config = config or {}
        self.weights_: List[float] = []
        self.intercept_: float = -20.0
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[float],
        X_valid: Optional[List[List[float]]] = None,
        y_valid: Optional[List[float]] = None,
    ) -> "LinearRiskModel":
        """Fit regularized linear model on training features."""
        if not X_train or not y_train:
            raise ValueError("Training data cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature count ({len(X_train)}) does not match target count ({len(y_train)}).")

        # Try scikit-learn Ridge if installed, fallback to self-contained solver
        try:
            from sklearn.linear_model import Ridge
            sk_model = Ridge(alpha=self.alpha, fit_intercept=True)
            sk_model.fit(X_train, y_train)
            self.weights_ = [float(w) for w in sk_model.coef_]
            self.intercept_ = float(sk_model.intercept_)
        except (ImportError, Exception):
            self.weights_, self.intercept_ = _solve_ridge_analytical(
                X_train, y_train, alpha=self.alpha
            )

        self.is_fitted_ = True
        return self

    def predict_risk(self, X: List[List[float]]) -> List[float]:
        """Predict continuous log-risk score for each feature vector."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions.")

        predictions: List[float] = []
        for row in X:
            val = self.intercept_ + sum(w * x for w, x in zip(self.weights_, row))
            predictions.append(val)
        return predictions

    def predict_probability(self, X: List[List[float]], threshold_log10: float) -> List[float]:
        """Predict probability that risk exceeds threshold_log10 using logistic scaling."""
        continuous_scores = self.predict_risk(X)
        probs: List[float] = []
        for s in continuous_scores:
            diff = s - threshold_log10
            # Logistic sigmoid with calibrated temperature
            try:
                p = 1.0 / (1.0 + math.exp(-2.0 * diff))
            except OverflowError:
                p = 1.0 if diff > 0 else 0.0
            probs.append(p)
        return probs

    def save(self, path: Union[Path, str]) -> None:
        """Save model weights and metadata to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "alpha": self.alpha,
            "feature_names": self.feature_names,
            "weights": self.weights_,
            "intercept": self.intercept_,
            "is_fitted": self.is_fitted_,
            "config": self.config,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "LinearRiskModel":
        """Load serialized model from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(
            alpha=data.get("alpha", 1.0),
            feature_names=data.get("feature_names", []),
            config=data.get("config", {}),
        )
        instance.weights_ = data.get("weights", [])
        instance.intercept_ = data.get("intercept", -20.0)
        instance.is_fitted_ = data.get("is_fitted", True)
        return instance
