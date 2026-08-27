"""Unit tests for Phase 1 Baseline validation, prediction alignment, and leakage prevention."""

import csv
import json
import math
import os
import unittest
from pathlib import Path

from orvexa.models_linear import LinearRiskModel
from orvexa.models_physics import PhysicsMaxRiskModel
from orvexa.models_xgb import XGBoostRiskModel
from orvexa.preprocessing import TrainFittedPreprocessor
from orvexa.splitting import SplitManifest


class TestPhase1Baselines(unittest.TestCase):
    """Test suite validating Phase 1 baseline artifacts, predictions, and leakage constraints."""

    @classmethod
    def setUpClass(cls):
        cls.horizons = [2, 3, 5]
        cls.expected_counts = {2: 1799, 3: 1700, 5: 1437}
        cls.models = ["esa", "ridge", "xgboost"]

        # Load Master Split
        manifest_path = "artifacts/splits/master_split_manifest.json"
        if os.path.exists(manifest_path):
            cls.master_split = SplitManifest.load(manifest_path)
            cls.train_set = set(cls.master_split.train_event_ids)
            cls.val_set = set(cls.master_split.val_event_ids)
            cls.test_set = set(cls.master_split.test_event_ids)
        else:
            cls.master_split = None

    def test_prediction_row_counts_and_event_ids(self):
        """Verify prediction files have exact row counts and matching event IDs."""
        for h in self.horizons:
            expected_n = self.expected_counts[h]
            h_event_csv = f"data/processed/events/events_H{h}.csv"
            with open(h_event_csv, "r", encoding="utf-8") as f:
                h_events = list(csv.DictReader(f))

            expected_test_events = [r["event_id"] for r in h_events if r["event_id"] in self.test_set]
            self.assertEqual(len(expected_test_events), expected_n)

            for m in self.models:
                pred_path = f"data/processed/predictions/{m}_H{h}_predictions.csv"
                self.assertTrue(os.path.exists(pred_path), f"Missing prediction file: {pred_path}")

                with open(pred_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                self.assertEqual(len(rows), expected_n, f"Row count mismatch in {pred_path}")
                pred_event_ids = [r["event_id"] for r in rows]
                self.assertEqual(pred_event_ids, expected_test_events, f"Event ID mismatch in {pred_path}")

    def test_target_alignment_and_no_nan_predictions(self):
        """Verify predicted target values match final_risk and contain no NaN or Inf."""
        for h in self.horizons:
            h_event_csv = f"data/processed/events/events_H{h}.csv"
            with open(h_event_csv, "r", encoding="utf-8") as f:
                event_target_map = {r["event_id"]: float(r["final_risk"]) for r in csv.DictReader(f)}

            for m in self.models:
                pred_path = f"data/processed/predictions/{m}_H{h}_predictions.csv"
                with open(pred_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        ev_id = row["event_id"]
                        true_risk = float(row["final_risk"])
                        pred_risk = float(row["predicted_risk"])

                        # Assert target matches ground truth
                        self.assertAlmostEqual(
                            true_risk,
                            event_target_map[ev_id],
                            places=5,
                            msg=f"Target drift in {pred_path} for event {ev_id}",
                        )
                        # Assert prediction is finite
                        self.assertFalse(math.isnan(pred_risk), f"NaN prediction in {pred_path} for event {ev_id}")
                        self.assertFalse(math.isinf(pred_risk), f"Inf prediction in {pred_path} for event {ev_id}")

    def test_model_and_preprocessor_artifacts_load_and_predict(self):
        """Verify saved model and preprocessor artifacts exist and can be deserialized."""
        for h in self.horizons:
            prep_path = f"artifacts/preprocessors/preprocessor_h{h:.1f}.json"
            self.assertTrue(os.path.exists(prep_path), f"Preprocessor missing: {prep_path}")
            prep = TrainFittedPreprocessor.load(prep_path)
            self.assertTrue(prep.is_fitted_)
            self.assertEqual(len(prep.output_feature_names_), 71)

            # Ridge model
            ridge_path = f"artifacts/models/ridge_h{h:.1f}.json"
            self.assertTrue(os.path.exists(ridge_path), f"Ridge artifact missing: {ridge_path}")
            ridge = LinearRiskModel.load(ridge_path)
            self.assertTrue(ridge.is_fitted_)
            self.assertEqual(len(ridge.weights_), 71)

            # XGBoost model
            xgb_path = f"artifacts/models/xgboost_h{h:.1f}.json"
            self.assertTrue(os.path.exists(xgb_path), f"XGBoost artifact missing: {xgb_path}")
            xgb = XGBoostRiskModel.load(xgb_path)
            self.assertTrue(xgb.is_fitted_)
            self.assertEqual(len(xgb.trees_), 100)

            # ESA model
            esa_path = f"artifacts/models/esa_max_risk_h{h:.1f}.json"
            self.assertTrue(os.path.exists(esa_path), f"ESA artifact missing: {esa_path}")
            esa = PhysicsMaxRiskModel.load(esa_path)
            self.assertEqual(esa.risk_col, "max_risk_estimate")

    def test_baseline_metrics_csv_structure(self):
        """Verify reports/baseline_metrics.csv exists, contains 9 rows, and expected headers."""
        metrics_path = "reports/baseline_metrics.csv"
        self.assertTrue(os.path.exists(metrics_path), "reports/baseline_metrics.csv is missing")

        with open(metrics_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 9, f"Expected 9 metric rows (3 models x 3 horizons), got {len(rows)}")

        required_cols = [
            "Model", "Horizon", "N", "MAE", "RMSE", "R2", "Pearson", "Spearman",
            "Recall@1%", "Recall@5%", "Recall@10%", "Precision@1%", "Precision@5%", "Precision@10%",
            "Missed@1%", "Missed@5%", "Missed@10%"
        ]
        for col in required_cols:
            self.assertIn(col, rows[0], f"Missing column: {col}")

    def test_split_disjointness_and_cross_horizon_leakage(self):
        """Verify strict zero leakage between train/val/test and across horizons."""
        self.assertIsNotNone(self.master_split)
        self.assertTrue(self.train_set.isdisjoint(self.val_set))
        self.assertTrue(self.train_set.isdisjoint(self.test_set))
        self.assertTrue(self.val_set.isdisjoint(self.test_set))

        # Check cross-horizon intersection
        h_splits = {}
        for h in self.horizons:
            with open(f"data/processed/events/events_H{h}.csv", "r", encoding="utf-8") as f:
                h_events = {r["event_id"] for r in csv.DictReader(f)}
            h_splits[h] = {
                "train": h_events.intersection(self.train_set),
                "val": h_events.intersection(self.val_set),
                "test": h_events.intersection(self.test_set),
            }

        # Train(H2) must never leak into Val(H3/H5) or Test(H3/H5)
        self.assertTrue(h_splits[2]["train"].isdisjoint(h_splits[3]["val"]))
        self.assertTrue(h_splits[2]["train"].isdisjoint(h_splits[3]["test"]))
        self.assertTrue(h_splits[2]["train"].isdisjoint(h_splits[5]["val"]))
        self.assertTrue(h_splits[2]["train"].isdisjoint(h_splits[5]["test"]))


if __name__ == "__main__":
    unittest.main()
