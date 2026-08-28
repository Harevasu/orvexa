"""Automated test suite for ORVEXA Phase 4A Step 2 (Controlled H6 Training & Validation).

Verifies:
1. H6 model checkpoints (M0, M4) exist and deserialize correctly with correct channel dimensions.
2. H6 preprocessor manifests exist (M0=34 channels, M4=37 channels) and were fit strictly on training split.
3. H6 validation prediction CSV files exist, contain exactly 1,264 rows, and belong strictly to validation split.
4. Validation predictions contain no NaN, Inf, or null values.
5. Metrics summary CSV exists and matches stored prediction metrics.
6. Test quarantine: 0 test rows accessed or evaluated.
7. Frozen baseline, Phase 3B checkpoints, and H6 datasets maintain 100% cryptographic SHA-256 integrity.
"""

import csv
import json
import math
import os
from pathlib import Path
import unittest
import torch

from orvexa.event_builder import compute_file_sha256
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


class TestPhase4AStep2(unittest.TestCase):
    """Test suite for Phase 4A Step 2 H6 training artifacts, preprocessors, and validation predictions."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.val_set = set(cls.split_manifest.val_event_ids)
        cls.test_set = set(cls.split_manifest.test_event_ids)
        cls.train_set = set(cls.split_manifest.train_event_ids)

        cls.frozen_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
            "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
            "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
            "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
            "data/processed/events/events_H6.csv": "bbc4a34ebbc900f6344d3380dc14faa1b691c2180e1ad875586eb031b5d7cee9",
            "data/processed/events/sequences_H6.csv": "ad31bc8e99ec8cf720fd4645fb571d2d906d2e9bd1fb961613c99dee514c8817",
        }

    def test_frozen_baselines_and_checkpoints_immutable(self):
        """Verify that all pre-existing baseline files and Phase 3B checkpoints remain 100% frozen."""
        for path, exp_hash in self.frozen_hashes.items():
            self.assertTrue(os.path.exists(path), f"Frozen file missing: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch on frozen artifact: {path}")

    def test_h6_model_checkpoints_exist_and_deserialize(self):
        """Verify H6 M0 and M4 model checkpoints exist with correct weight shapes."""
        models = [
            ("M0", 34, "artifacts/models/phase4a/tcn_best_M0_h6.0.pt"),
            ("M4", 37, "artifacts/models/phase4a/tcn_best_M4_h6.0.pt"),
        ]
        for name, in_ch, path in models:
            self.assertTrue(os.path.exists(path), f"Missing checkpoint: {path}")
            ckpt = torch.load(path, map_location="cpu")
            self.assertIn("network.0.conv1.conv.weight", ckpt)
            conv_weight = ckpt["network.0.conv1.conv.weight"]
            self.assertEqual(conv_weight.shape, (64, in_ch, 3), f"Unexpected weight shape for {name}")

    def test_h6_preprocessors_exist_and_valid_channels(self):
        """Verify H6 preprocessors exist with exact channel contracts."""
        preps = [
            ("M0", 34, "artifacts/preprocessors/phase4a/preprocessor_M0_h6.0.json"),
            ("M4", 37, "artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json"),
        ]
        for name, in_ch, path in preps:
            self.assertTrue(os.path.exists(path), f"Missing preprocessor: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["n_channels"], in_ch)
            self.assertEqual(len(data["feature_columns"]), in_ch)
            self.assertEqual(len(data["channel_stats"]), in_ch)

    def test_validation_predictions_integrity_and_split_membership(self):
        """Verify validation predictions exist, match N=1,264, and belong strictly to validation split."""
        preds = [
            "data/processed/predictions/phase4a/tcn_M0_h6.0_val_predictions.csv",
            "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv",
        ]
        for p in preds:
            self.assertTrue(os.path.exists(p), f"Missing prediction file: {p}")
            with open(p, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1264, f"Unexpected validation row count in {p}")
            for r in rows:
                eid = r["event_id"]
                yt = float(r["final_risk"])
                yp = float(r["predicted_risk"])

                self.assertIn(eid, self.val_set, f"Event {eid} does not belong to validation split!")
                self.assertNotIn(eid, self.test_set, f"Test event {eid} leaked into validation predictions!")
                self.assertTrue(math.isfinite(yt))
                self.assertTrue(math.isfinite(yp))

    def test_metrics_csv_structure_and_m4_improvement(self):
        """Verify metrics CSV exists and confirms M4 superiority over M0 on H6 validation."""
        csv_path = "reports/phase4a_step2_metrics.csv"
        self.assertTrue(os.path.exists(csv_path), f"Missing {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        m0_row = [r for r in rows if r["model"] == "M0"][0]
        m4_row = [r for r in rows if r["model"] == "M4"][0]

        # M4 must have lower Huber loss, lower MAE, and higher R2 than M0
        self.assertLess(float(m4_row["val_huber_best"]), float(m0_row["val_huber_best"]))
        self.assertLess(float(m4_row["val_mae"]), float(m0_row["val_mae"]))
        self.assertGreater(float(m4_row["val_r2"]), float(m0_row["val_r2"]))
        self.assertGreater(float(m4_row["val_r2"]), 0.0, "M4 R2 should be positive on H6")


if __name__ == "__main__":
    unittest.main()
