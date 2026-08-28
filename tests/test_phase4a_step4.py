"""Automated test suite for ORVEXA Phase 4A Step 4 (Final Blind H6 Test Evaluation).

Verifies:
1. Blind test prediction file exists in data/processed/predictions/phase4a/blind_test/.
2. Test prediction row count matches exact H6 qualifying test count (1,279 events).
3. All evaluated event IDs strictly belong to the master test split partition (1,974 events).
4. Predictions contain no NaN, Inf, or null values.
5. Blind test metrics CSV exists and matches stored prediction metrics with zero discrepancy.
6. Frozen M4 checkpoint, preprocessor, and canonical datasets maintain 100% cryptographic SHA-256 integrity.
"""

import csv
import json
import math
import os
from pathlib import Path
import unittest

from orvexa.event_builder import compute_file_sha256
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


class TestPhase4AStep4(unittest.TestCase):
    """Test suite for Phase 4A Step 4 blind test evaluation artifacts and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.test_event_ids = set(cls.split_manifest.test_event_ids)
        cls.val_event_ids = set(cls.split_manifest.val_event_ids)
        cls.train_event_ids = set(cls.split_manifest.train_event_ids)

        cls.expected_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
            "artifacts/models/phase4a/tcn_best_M4_h6.0.pt": "9bb5f10b990be67336dd0902ab3943c28b20b21d18855f2ab4e9b4ca31844d30",
            "artifacts/models/phase4a/tcn_best_M4_h6.0.json": "99fc8138f877a26567e2e01388293709f8a1f8364ad76c3342274b7b5d25784b",
            "artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json": "e2c5e17769efb93358bfe9308591aefbf30f2b10045cbd970ae0b2c5b0fc5a62",
            "data/processed/events/events_H6.csv": "bbc4a34ebbc900f6344d3380dc14faa1b691c2180e1ad875586eb031b5d7cee9",
            "data/processed/events/sequences_H6.csv": "ad31bc8e99ec8cf720fd4645fb571d2d906d2e9bd1fb961613c99dee514c8817",
        }

    def test_frozen_artifacts_integrity(self):
        """Verify all baseline files and frozen candidate checkpoints maintain SHA-256 integrity."""
        for path, exp_hash in self.expected_hashes.items():
            self.assertTrue(os.path.exists(path), f"Missing file: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch on {path}")

    def test_blind_test_predictions_exist_and_strict_id_membership(self):
        """Verify test prediction CSV exists, has 1,279 rows, and strictly belongs to test partition."""
        csv_path = "data/processed/predictions/phase4a/blind_test/tcn_M4_h6.0_test_predictions.csv"
        self.assertTrue(os.path.exists(csv_path), f"Missing test prediction file: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1279, "Unexpected test row count for H6")

        for r in rows:
            eid = r["event_id"]
            yt = float(r["final_risk"])
            yp = float(r["predicted_risk"])
            sl = int(r["sequence_length"])

            # Strict test set membership
            self.assertIn(eid, self.test_event_ids, f"Event {eid} does not belong to master test partition!")
            self.assertNotIn(eid, self.train_event_ids, f"Train event {eid} leaked into test predictions!")
            self.assertNotIn(eid, self.val_event_ids, f"Val event {eid} leaked into test predictions!")
            self.assertTrue(math.isfinite(yt))
            self.assertTrue(math.isfinite(yp))
            self.assertGreaterEqual(sl, 1)

    def test_blind_test_metrics_csv_consistency(self):
        """Verify blind test metrics CSV exists and matches recomputed prediction metrics."""
        csv_path = "reports/phase4a_step4_blind_test_metrics.csv"
        pred_csv = "data/processed/predictions/phase4a/blind_test/tcn_M4_h6.0_test_predictions.csv"
        self.assertTrue(os.path.exists(csv_path))

        with open(pred_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        yt = [float(r["final_risk"]) for r in rows]
        yp = [float(r["predicted_risk"]) for r in rows]

        reg = compute_regression_metrics(yt, yp)

        with open(csv_path, "r", encoding="utf-8") as f:
            metrics = list(csv.DictReader(f))[0]

        self.assertAlmostEqual(float(metrics["test_mae"]), reg["mae"], places=5)
        self.assertAlmostEqual(float(metrics["test_rmse"]), reg["rmse"], places=5)
        self.assertAlmostEqual(float(metrics["test_r2"]), reg["r2"], places=5)
        self.assertAlmostEqual(float(metrics["test_spearman"]), reg["spearman_correlation"], places=5)


if __name__ == "__main__":
    unittest.main()
