"""Probability calibration using Platt scaling and Isotonic regression fitted strictly on validation data.

References:
- Platt (1999): "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods".
- Zadrozny & Elkan (2002): "Transforming Classifier Scores into Accurate Multiclass Probability Estimates".
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _isotonic_pava(y: List[float], w: Optional[List[float]] = None) -> List[float]:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression."""
    n = len(y)
    if n == 0:
        return []

    weights = [1.0] * n if w is None else list(w)
    values = [float(val) for val in y]

    # Block representation: (start, end, weight, value)
    blocks: List[List[float]] = [[float(v), float(w), 1.0] for v, w in zip(values, weights)]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            # Violation: merge block i and i+1
            w1 = blocks[i][1]
            w2 = blocks[i + 1][1]
            w_total = w1 + w2
            v_merged = (blocks[i][0] * w1 + blocks[i + 1][0] * w2) / w_total
            count_merged = blocks[i][2] + blocks[i + 1][2]

            blocks[i] = [v_merged, w_total, count_merged]
            del blocks[i + 1]

            # Step back to check if previous blocks are now violating
            if i > 0:
                i -= 1
        else:
            i += 1

    # Unpack fitted values
    result: List[float] = []
    for b in blocks:
        count = int(b[2])
        result.extend([b[0]] * count)

    return result


class ProbabilityCalibrator:
    """Calibrates continuous model risk scores into reliable probabilities [0, 1].
    
    Methods:
    - 'platt': Sigmoid logistic calibration: P(y=1|s) = 1 / (1 + exp(A*s + B))
    - 'isotonic': Non-parametric isotonic regression via PAVA
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown calibration method: '{method}'. Must be 'isotonic' or 'platt'.")
        self.method = method
        self.a_: float = -1.0
        self.b_: float = 0.0
        self.iso_x_: List[float] = []
        self.iso_y_: List[float] = []
        self.is_fitted_: bool = False

    def fit(self, y_pred_val: List[float], y_true_binary: List[int]) -> "ProbabilityCalibrator":
        """Fit calibration curve strictly on validation set predictions and binary labels."""
        if not y_pred_val or not y_true_binary:
            raise ValueError("Validation inputs cannot be empty.")
        if len(y_pred_val) != len(y_true_binary):
            raise ValueError("Length mismatch between predictions and targets.")

        if self.method == "platt":
            # Simple gradient descent for logistic parameters A, B: P = 1 / (1 + exp(A*s + B))
            a = -1.0
            b = 0.0
            lr = 0.05
            for _ in range(200):
                grad_a = 0.0
                grad_b = 0.0
                for s, y in zip(y_pred_val, y_true_binary):
                    z = a * s + b
                    p = 1.0 / (1.0 + math.exp(min(max(z, -30.0), 30.0)))
                    err = p - y
                    grad_a += err * s
                    grad_b += err
                a -= lr * (grad_a / len(y_pred_val))
                b -= lr * (grad_b / len(y_pred_val))
            self.a_ = a
            self.b_ = b

        elif self.method == "isotonic":
            # Sort by predicted score
            paired = sorted(zip(y_pred_val, y_true_binary), key=lambda x: x[0])
            sorted_x = [p[0] for p in paired]
            sorted_y = [float(p[1]) for p in paired]

            fitted_y = _isotonic_pava(sorted_y)
            self.iso_x_ = sorted_x
            self.iso_y_ = fitted_y

        self.is_fitted_ = True
        return self

    def calibrate(self, y_pred: List[float]) -> List[float]:
        """Transform uncalibrated risk predictions into calibrated probability estimates."""
        if not self.is_fitted_:
            raise RuntimeError("Calibrator must be fitted before transforming.")

        calibrated: List[float] = []

        if self.method == "platt":
            for s in y_pred:
                z = self.a_ * s + self.b_
                p = 1.0 / (1.0 + math.exp(min(max(z, -30.0), 30.0)))
                calibrated.append(min(max(p, 0.0), 1.0))

        elif self.method == "isotonic":
            for s in y_pred:
                if not self.iso_x_:
                    calibrated.append(0.5)
                    continue
                if s <= self.iso_x_[0]:
                    calibrated.append(self.iso_y_[0])
                elif s >= self.iso_x_[-1]:
                    calibrated.append(self.iso_y_[-1])
                else:
                    # Linear interpolation
                    idx = 0
                    while idx < len(self.iso_x_) - 1 and self.iso_x_[idx + 1] < s:
                        idx += 1
                    x0, x1 = self.iso_x_[idx], self.iso_x_[idx + 1]
                    y0, y1 = self.iso_y_[idx], self.iso_y_[idx + 1]
                    if abs(x1 - x0) < 1e-9:
                        p = y0
                    else:
                        t = (s - x0) / (x1 - x0)
                        p = y0 + t * (y1 - y0)
                    calibrated.append(min(max(p, 0.0), 1.0))

        return calibrated

    def save(self, path: Union[Path, str]) -> None:
        """Save calibrator state to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": self.method,
            "a": self.a_,
            "b": self.b_,
            "iso_x": self.iso_x_,
            "iso_y": self.iso_y_,
            "is_fitted": self.is_fitted_,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "ProbabilityCalibrator":
        """Load calibrator from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = cls(method=data["method"])
        inst.a_ = data.get("a", -1.0)
        inst.b_ = data.get("b", 0.0)
        inst.iso_x_ = data.get("iso_x", [])
        inst.iso_y_ = data.get("iso_y", [])
        inst.is_fitted_ = data.get("is_fitted", True)
        return inst
