"""Unit tests for Phase 2B Temporal Model Training, Artifacts, and Predictions."""

import csv
import json
import math
import os
import sys
import unittest
from pathlib import Path

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.models_tcn import TCNRiskModel
from orvexa.preprocessing import TrainFittedPreprocessor, TrainFittedSequencePreprocessor
from orvexa.splitting import SplitManifest


class TestPhase2BTemporalTraining(unittest.TestCase):
    """Test suite validating Phase 2B temporal model artifacts, predictions, and metrics."""

    @classmethod
    def setUpClass(cls):
        cls.horizons = [2, 3, 5]
        cls.expected_counts = {2: 1799, 3: 1700, 5: 1437}
        cls.temporal_models = ["temporal_xgboost", "tcn"]

        manifest_path = "artifacts/splits/master_split_manifest.json"
        if os.path.exists(manifest_path):
            cls.master_split = SplitManifest.load(manifest_path)
            cls.train_set = set(cls.master_split.train_event_ids)
            cls.val_set = set(cls.master_split.val_event_ids)
            cls.test_set = set(cls.master_split.test_event_ids)
        else:
            cls.master_split = None

    def test_metrics_csv_structure_and_completeness(self):
        """Verify reports/phase2b_temporal_metrics.csv exists, has 15 rows, and valid columns."""
        csv_path = "reports/phase2b_temporal_metrics.csv"
        self.assertTrue(os.path.exists(csv_path), f"Metrics CSV missing: {csv_path}")

        expected_columns = [
            "Model",
            "Horizon",
            "N",
            "MAE",
            "RMSE",
            "R2",
            "Pearson",
            "Spearman",
            "Recall@1%",
            "Recall@5%",
            "Recall@10%",
            "Precision@1%",
            "Precision@5%",
            "Precision@10%",
            "Missed@1%",
            "Missed@5%",
            "Missed@10%",
        ]

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected_columns)
            rows = list(reader)

        self.assertEqual(len(rows), 15, f"Expected 15 metric rows, found {len(rows)}")

        models_found = {r["Model"] for r in rows}
        expected_models = {
            "ESA max_risk_estimate",
            "Ridge",
            "XGBoost",
            "Temporal XGBoost",
            "Masked Causal TCN",
        }
        self.assertEqual(models_found, expected_models)

        # Check numeric validity
        for r in rows:
            self.assertFalse(math.isnan(float(r["MAE"])))
            self.assertFalse(math.isnan(float(r["RMSE"])))
            self.assertFalse(math.isnan(float(r["R2"])))
            self.assertFalse(math.isnan(float(r["Pearson"])))
            self.assertFalse(math.isnan(float(r["Spearman"])))

    def test_temporal_prediction_files_integrity_and_alignment(self):
        """Verify temporal prediction files exist, have exact row counts, and match test event IDs."""
        for h in self.horizons:
            expected_n = self.expected_counts[h]
            h_event_csv = f"data/processed/events/events_H{h}.csv"
            with open(h_event_csv, "r", encoding="utf-8") as f:
                h_events = list(csv.DictReader(f))

            expected_test_events = [
                eid for eid in self.master_split.test_event_ids
                if any(r["event_id"] == eid for r in h_events)
            ]
            self.assertEqual(len(expected_test_events), expected_n)

            for m in self.temporal_models:
                pred_path = f"data/processed/predictions/{m}_H{h}_predictions.csv"
                self.assertTrue(os.path.exists(pred_path), f"Missing prediction file: {pred_path}")

                with open(pred_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                self.assertEqual(len(rows), expected_n, f"Row count mismatch in {pred_path}")
                pred_event_ids = [r["event_id"] for r in rows]
                self.assertEqual(pred_event_ids, expected_test_events, f"Event ID order mismatch in {pred_path}")

                # Check for NaNs or Infs
                for r in rows:
                    y_true = float(r["final_risk"])
                    y_pred = float(r["predicted_risk"])
                    self.assertFalse(math.isnan(y_true))
                    self.assertFalse(math.isinf(y_true))
                    self.assertFalse(math.isnan(y_pred), f"NaN prediction in {pred_path}")
                    self.assertFalse(math.isinf(y_pred), f"Inf prediction in {pred_path}")

    def test_saved_model_artifacts_and_deserialization(self):
        """Verify model weights, checkpoint JSONs, and preprocessors can be deserialized."""
        for h in self.horizons:
            # 1. Temporal XGBoost
            txgb_path = f"artifacts/models/temporal_xgboost_h{h:.1f}.json"
            self.assertTrue(os.path.exists(txgb_path), f"Missing Temporal XGBoost model: {txgb_path}")

            txgb_prep_path = f"artifacts/preprocessors/preprocessor_temporal_xgb_h{h:.1f}.json"
            self.assertTrue(os.path.exists(txgb_prep_path), f"Missing Temporal XGBoost preprocessor: {txgb_prep_path}")
            prep_txgb = TrainFittedPreprocessor.load(txgb_prep_path)
            self.assertTrue(prep_txgb.is_fitted_)

            # 2. TCN Best Checkpoint
            tcn_pt_path = f"artifacts/models/tcn_best_h{h:.1f}.pt"
            tcn_json_path = f"artifacts/models/tcn_best_h{h:.1f}.json"
            self.assertTrue(os.path.exists(tcn_pt_path), f"Missing TCN weights: {tcn_pt_path}")
            self.assertTrue(os.path.exists(tcn_json_path), f"Missing TCN meta: {tcn_json_path}")

            tcn_model = TCNRiskModel.load(f"artifacts/models/tcn_best_h{h:.1f}")
            self.assertTrue(tcn_model.is_fitted_)

            # 3. TCN Sequence Preprocessor
            tcn_prep_path = f"artifacts/preprocessors/preprocessor_tcn_h{h:.1f}.json"
            self.assertTrue(os.path.exists(tcn_prep_path), f"Missing TCN preprocessor: {tcn_prep_path}")
            tcn_seq_prep = TrainFittedSequencePreprocessor.load(tcn_prep_path)
            self.assertTrue(tcn_seq_prep.is_fitted_)


if __name__ == "__main__":
    unittest.main()
