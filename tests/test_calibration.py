"""Tests for probability calibrators (Platt, Isotonic) and ECE calculation."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.calibration import ProbabilityCalibrator
from orvexa.classification_metrics import compute_expected_calibration_error


class TestCalibration(unittest.TestCase):
    """Test suite for post-hoc probability calibrators and calibration error metrics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Synthetic monotonic relationship between predicted score and positive rate
        self.val_scores = [-25.0, -20.0, -15.0, -10.0, -5.0, -3.0]
        self.val_labels = [0, 0, 0, 1, 1, 1]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_isotonic_calibration_monotonicity(self):
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(self.val_scores, self.val_labels)
        probs = calibrator.calibrate(self.val_scores)

        # Probabilities must be in [0, 1] and non-decreasing
        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        for i in range(len(probs) - 1):
            self.assertLessEqual(probs[i], probs[i + 1] + 1e-6)

    def test_platt_calibration_sigmoid(self):
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(self.val_scores, self.val_labels)
        probs = calibrator.calibrate(self.val_scores)

        for p in probs:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_calibrator_serialization(self):
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(self.val_scores, self.val_labels)
        save_path = os.path.join(self.temp_dir, "calibrator.json")
        calibrator.save(save_path)

        loaded = ProbabilityCalibrator.load(save_path)
        self.assertEqual(loaded.method, "isotonic")
        self.assertEqual(calibrator.calibrate(self.val_scores), loaded.calibrate(self.val_scores))

    def test_expected_calibration_error(self):
        # Perfect calibration: predicted 1.0 on positive, 0.0 on negative
        y_true = [0, 0, 1, 1]
        y_prob_perfect = [0.0, 0.0, 1.0, 1.0]
        ece_perfect = compute_expected_calibration_error(y_true, y_prob_perfect, n_bins=5)
        self.assertAlmostEqual(ece_perfect, 0.0)

        # Severely uncalibrated: predicted 1.0 on all 0s
        y_prob_poor = [1.0, 1.0, 0.0, 0.0]
        ece_poor = compute_expected_calibration_error(y_true, y_prob_poor, n_bins=5)
        self.assertGreater(ece_poor, 0.5)


if __name__ == "__main__":
    unittest.main()
