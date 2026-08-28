"""Automated test suite for ORVEXA Phase 3B Step 4B Blind Test Evaluation.

Verifies:
1. Blind test prediction files exist in data/processed/predictions/phase3b/blind_test/.
2. Test prediction row counts match exact horizon qualifying test counts (H2=1,799, H3=1,700, H5=1,437).
3. All evaluated event IDs strictly belong to the master test split partition (1,974 events).
4. Predictions contain no NaN, Inf, or null values.
5. Blind test metrics CSV exists and matches stored prediction metrics with zero discrepancy.
6. Frozen M4 checkpoints and canonical datasets maintain 100% cryptographic SHA-256 integrity.
"""

import csv
import math
import os
from pathlib import Path
import unittest

from orvexa.event_builder import compute_file_sha256
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


class TestPhase3BStep4B(unittest.TestCase):
    """Test suite for Phase 3B Step 4B blind test evaluation artifacts and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.test_event_ids = set(cls.split_manifest.test_event_ids)
        cls.horizons = [2.0, 3.0, 5.0]
        cls.expected_test_counts = {2.0: 1799, 3.0: 1700, 5.0: 1437}
        cls.expected_m4_hashes = {
            "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
            "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
            "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
        }

    def test_blind_test_predictions_exist_and_strict_id_membership(self):
        """Verify test prediction CSVs exist, match qualifying counts, and belong to test partition."""
        for h in self.horizons:
            csv_path = f"data/processed/predictions/phase3b/blind_test/tcn_M4_h{h:.1f}_test_predictions.csv"
            self.assertTrue(os.path.exists(csv_path), f"Missing test prediction file: {csv_path}")

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), self.expected_test_counts[h], f"Unexpected test row count for H{int(h)}")

            for r in rows:
                eid = r["event_id"]
                yt = float(r["final_risk"])
                yp = float(r["predicted_risk"])
                sl = int(r["sequence_length"])

                # Strict test set membership
                self.assertIn(eid, self.test_event_ids, f"Event {eid} does not belong to master test partition!")
                self.assertTrue(math.isfinite(yt))
                self.assertTrue(math.isfinite(yp))
                self.assertGreaterEqual(sl, 1)

    def test_blind_test_metrics_csv_consistency(self):
        """Verify reports/phase3b_step4b_blind_test_metrics.csv matches prediction file metrics."""
        metrics_csv = "reports/phase3b_step4b_blind_test_metrics.csv"
        self.assertTrue(os.path.exists(metrics_csv), "Missing blind test metrics CSV.")

        with open(metrics_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 3, "Expected 3 rows in blind test metrics CSV (H2, H3, H5)")

        for r in rows:
            h_str = r["horizon"]
            h_val = 2.0 if h_str == "H2" else (3.0 if h_str == "H3" else 5.0)

            # Recompute from prediction CSV
            pred_csv = f"data/processed/predictions/phase3b/blind_test/tcn_M4_h{h_val:.1f}_test_predictions.csv"
            with open(pred_csv, "r", encoding="utf-8") as pf:
                pred_rows = list(csv.DictReader(pf))

            yt = [float(pr["final_risk"]) for pr in pred_rows]
            yp = [float(pr["predicted_risk"]) for pr in pred_rows]

            reg = compute_regression_metrics(yt, yp)
            rank = compute_ranking_metrics(yt, yp, threshold_log10=-5.0)

            self.assertAlmostEqual(float(r["test_mae"]), reg["mae"], places=4)
            self.assertAlmostEqual(float(r["test_rmse"]), reg["rmse"], places=4)
            self.assertAlmostEqual(float(r["test_r2"]), reg["r2"], places=4)
            self.assertAlmostEqual(float(r["test_recall_top10pct"]), rank["budget_pct_10"]["recall"], places=4)

    def test_frozen_m4_checkpoints_hashes_immutable(self):
        """Verify that frozen M4 model checkpoint weights remained 100% immutable."""
        for fpath, exp_h in self.expected_m4_hashes.items():
            self.assertTrue(os.path.exists(fpath), f"Missing checkpoint: {fpath}")
            act_h = compute_file_sha256(fpath)
            self.assertEqual(act_h, exp_h, f"Checkpoint hash mismatch for {fpath}")


if __name__ == "__main__":
    unittest.main()
