"""Tests for data perturbation and robustness functions."""

import unittest

from orvexa.robustness import (
    evaluate_robustness_perturbations,
    inject_feature_dropout,
    inject_gaussian_noise,
)


class TestRobustness(unittest.TestCase):
    """Test suite for perturbation injection and robustness evaluation routines."""

    def setUp(self):
        self.X_test = [
            [10.0, 20.0],
            [15.0, 25.0],
            [20.0, 30.0],
            [25.0, 35.0],
        ]
        self.y_test = [-20.0, -18.0, -16.0, -14.0]

    def test_gaussian_noise_injection(self):
        noisy = inject_gaussian_noise(self.X_test, noise_scale=0.1, seed=42)
        self.assertEqual(len(noisy), len(self.X_test))
        self.assertEqual(len(noisy[0]), len(self.X_test[0]))
        # Values should be perturbed but near original
        for i in range(len(self.X_test)):
            for j in range(len(self.X_test[0])):
                self.assertNotEqual(noisy[i][j], self.X_test[i][j])
                self.assertAlmostEqual(noisy[i][j], self.X_test[i][j], delta=10.0)

    def test_feature_dropout_injection(self):
        # 100% dropout should set all to default_fill
        all_dropped = inject_feature_dropout(self.X_test, drop_rate=1.0, default_fill=0.0)
        for row in all_dropped:
            self.assertEqual(row, [0.0, 0.0])

    def test_evaluate_robustness_perturbations(self):
        def dummy_predict(X):
            # Model: y = -30.0 + x1
            return [-30.0 + row[0] for row in X]

        res = evaluate_robustness_perturbations(
            dummy_predict, self.X_test, self.y_test, noise_levels=[0.05], dropout_rates=[0.10]
        )
        self.assertIn("baseline", res)
        self.assertIn("gaussian_noise", res)
        self.assertIn("feature_dropout", res)
        self.assertIn("noise_5pct", res["gaussian_noise"])
        self.assertIn("drop_10pct", res["feature_dropout"])


if __name__ == "__main__":
    unittest.main()
