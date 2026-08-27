"""Unit tests for XGBoost feature ablation study validation."""

import csv
import json
import math
import os
import unittest

from orvexa.models_xgb import XGBoostRiskModel
from orvexa.preprocessing import TrainFittedPreprocessor
from orvexa.splitting import SplitManifest


class TestAblationStudy(unittest.TestCase):
    """Test suite validating the XGBoost feature ablation study."""

    @classmethod
    def setUpClass(cls):
        cls.horizons = [2, 3, 5]
        cls.expected_counts = {2: 1799, 3: 1700, 5: 1437}

        manifest_path = "artifacts/splits/master_split_manifest.json"
        if os.path.exists(manifest_path):
            cls.master_split = SplitManifest.load(manifest_path)
            cls.test_set = set(cls.master_split.test_event_ids)
        else:
            cls.master_split = None
            cls.test_set = set()

    def test_ablation_preprocessor_excludes_max_risk_estimate(self):
        """Verify preprocessors have 69 channels and max_risk_estimate is strictly excluded."""
        for h in self.horizons:
            prep_path = f"artifacts/preprocessors/preprocessor_no_max_risk_h{h:.1f}.json"
            self.assertTrue(os.path.exists(prep_path), f"Missing preprocessor: {prep_path}")

            prep = TrainFittedPreprocessor.load(prep_path)
            self.assertTrue(prep.is_fitted_)
            self.assertEqual(len(prep.output_feature_names_), 69)
            self.assertNotIn("max_risk_estimate", prep.output_feature_names_)
            self.assertNotIn("max_risk_estimate_is_missing", prep.output_feature_names_)
            self.assertNotIn("final_risk", prep.output_feature_names_)
            self.assertNotIn("risk", prep.output_feature_names_)
            self.assertNotIn("event_id", prep.output_feature_names_)
            self.assertNotIn("mission_id", prep.output_feature_names_)

    def test_ablation_model_artifacts(self):
        """Verify ablation model artifacts load and have 33 features / 69 channels."""
        for h in self.horizons:
            model_path = f"artifacts/models/xgboost_no_max_risk_h{h:.1f}.json"
            self.assertTrue(os.path.exists(model_path), f"Missing model: {model_path}")

            model = XGBoostRiskModel.load(model_path)
            self.assertTrue(model.is_fitted_)
            self.assertEqual(len(model.trees_), 100)
            self.assertEqual(len(model.feature_names), 69)
            self.assertNotIn("max_risk_estimate", model.feature_names)

    def test_ablation_prediction_row_counts_and_validity(self):
        """Verify prediction CSVs have exact row counts and finite values."""
        for h in self.horizons:
            expected_n = self.expected_counts[h]
            pred_path = f"data/processed/predictions/xgboost_no_max_risk_H{h}_predictions.csv"
            self.assertTrue(os.path.exists(pred_path), f"Missing prediction file: {pred_path}")

            with open(pred_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), expected_n, f"Row count mismatch in {pred_path}")
            for r in rows:
                p = float(r["predicted_risk"])
                self.assertFalse(math.isnan(p), f"NaN in {pred_path}")
                self.assertFalse(math.isinf(p), f"Inf in {pred_path}")

    def test_original_phase1_artifacts_unmodified(self):
        """Verify that original Phase 1 artifacts remain completely untouched."""
        orig_artifacts = [
            "artifacts/models/xgboost_h2.0.json",
            "artifacts/models/xgboost_h3.0.json",
            "artifacts/models/xgboost_h5.0.json",
            "reports/baseline_metrics.csv",
            "reports/BASELINE_TRAINING_REPORT.md",
        ]
        for art in orig_artifacts:
            self.assertTrue(os.path.exists(art), f"Original artifact missing: {art}")

    def test_ablation_metrics_csv_structure(self):
        """Verify reports/xgboost_ablation_metrics.csv exists and has 3 rows with 33 features."""
        csv_path = "reports/xgboost_ablation_metrics.csv"
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(int(r["Features"]), 33)


if __name__ == "__main__":
    unittest.main()
