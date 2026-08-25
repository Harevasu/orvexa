"""Probability calibration using Platt scaling and Isotonic regression fitted on validation data."""

from typing import Any, Dict


class ProbabilityCalibrator:
    """Calibrates continuous or uncalibrated risk predictions to reliable probability estimates."""

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.calibrator_ = None

    def fit(self, y_pred_val: Any, y_true_val: Any) -> "ProbabilityCalibrator":
        raise NotImplementedError("Calibration fitting not yet implemented.")

    def calibrate(self, y_pred: Any) -> Any:
        raise NotImplementedError("Calibration transformation not yet implemented.")
