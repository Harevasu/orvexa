"""Tests for event-level bootstrap confidence interval estimation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.bootstrap import compute_event_bootstrap_ci


class TestBootstrap(unittest.TestCase):
    """Test suite for event-level non-parametric bootstrap resampling."""

    def test_bootstrap_mean_estimation(self):
        event_ids = [f"ev_{i}" for i in range(50)]
        y_true = [float(i) for i in range(50)]
        y_pred = [float(i) + 1.0 for i in range(50)]

        def mae_metric(yt, yp):
            return sum(abs(t - p) for t, p in zip(yt, yp)) / len(yt)

        ci_res = compute_event_bootstrap_ci(
            event_ids, y_true, y_pred, metric_fn=mae_metric, n_iterations=200, seed=42
        )
        self.assertAlmostEqual(ci_res["point_estimate"], 1.0)
        self.assertAlmostEqual(ci_res["ci_lower"], 1.0)
        self.assertAlmostEqual(ci_res["ci_upper"], 1.0)
        self.assertAlmostEqual(ci_res["std_error"], 0.0)

    def test_bootstrap_confidence_bounds(self):
        event_ids = [f"ev_{i}" for i in range(100)]
        # True values 0.0 or 10.0
        y_true = [0.0] * 50 + [10.0] * 50
        y_pred = [0.0] * 100

        def mean_true_metric(yt, yp):
            return sum(yt) / len(yt)

        ci_res = compute_event_bootstrap_ci(
            event_ids, y_true, y_pred, metric_fn=mean_true_metric, n_iterations=500, seed=42
        )
        self.assertAlmostEqual(ci_res["point_estimate"], 5.0)
        self.assertLess(ci_res["ci_lower"], 5.0)
        self.assertGreater(ci_res["ci_upper"], 5.0)
        self.assertGreater(ci_res["std_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
