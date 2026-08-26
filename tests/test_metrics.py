"""Tests for evaluation metrics: Recall@K, PR-AUC, MAE, and edge cases."""

import unittest

from orvexa.classification_metrics import compute_classification_metrics, compute_pr_auc
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics


class TestMetrics(unittest.TestCase):
    """Test suite for operational ranking, probabilistic classification, and regression metrics."""

    def test_ranking_metrics_perfect_ranking(self):
        # 100 events: 5 high risk (true risk >= -5.0) placed at top of predictions
        y_true = [-20.0] * 95 + [-4.0, -3.5, -2.0, -1.0, -0.5]
        y_pred = list(range(100))  # Ascending, so highest values are at end

        results = compute_ranking_metrics(
            y_true, y_pred, alert_budgets=[0.05, 0.10], threshold_log10=-5.0
        )
        self.assertEqual(results["high_risk_events_count"], 5)
        # Top 5% (5 events) captures all 5 true high-risk events -> Recall@5% = 1.0, Missed = 0
        b5 = results["budget_pct_5"]
        self.assertEqual(b5["recall"], 1.0)
        self.assertEqual(b5["precision"], 1.0)
        self.assertEqual(b5["missed_high_risk"], 0)
        self.assertEqual(b5["ndcg"], 1.0)

    def test_ranking_metrics_inverted_ranking(self):
        # All 5 true high risk events placed at the very bottom
        y_true = [-4.0, -3.5, -2.0, -1.0, -0.5] + [-20.0] * 95
        y_pred = list(range(100))  # High-risk items get lowest predicted scores 0..4

        results = compute_ranking_metrics(
            y_true, y_pred, alert_budgets=[0.05], threshold_log10=-5.0
        )
        b5 = results["budget_pct_5"]
        self.assertEqual(b5["recall"], 0.0)
        self.assertEqual(b5["missed_high_risk"], 5)

    def test_regression_metrics_calculation(self):
        y_true = [-20.0, -15.0, -10.0, -5.0]
        y_pred = [-19.0, -15.0, -12.0, -4.0]
        # Errors: [1.0, 0.0, 2.0, 1.0] -> MAE = 1.0, RMSE = sqrt((1+0+4+1)/4) = sqrt(1.5) approx 1.2247
        metrics = compute_regression_metrics(y_true, y_pred)
        self.assertEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.22474, places=4)
        self.assertEqual(metrics["spearman_correlation"], 1.0)

    def test_classification_metrics_pr_auc(self):
        # 2 positives, 2 negatives
        y_true = [1, 1, 0, 0]
        y_prob = [0.9, 0.8, 0.2, 0.1]
        auc = compute_pr_auc(y_true, y_prob)
        self.assertAlmostEqual(auc, 1.0)

        metrics = compute_classification_metrics(y_true, y_prob)
        self.assertIn("brier_score", metrics)
        self.assertIn("expected_calibration_error", metrics)
        self.assertIn("pr_auc", metrics)


if __name__ == "__main__":
    unittest.main()
