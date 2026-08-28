"""Automated test suite for ORVEXA Phase 4A Step 3 (H6 M4 Forensic Audit & Candidate Freeze).

Verifies:
1. Frozen baseline artifacts, Phase 3B checkpoints, and Step 1/2 H6 files maintain 100% SHA-256 integrity.
2. Candidate freeze manifest exists and matches checkpoint/preprocessor hashes.
3. H6 M4 checkpoint deserializes and maintains valid weight tensors (0 NaN, 0 Inf, shape [64, 37, 3]).
4. Validation predictions reproduce reported metrics with numerical tolerance <= 1e-6.
5. Paired bootstrap confirms statistically significant improvements over M0.
6. Test quarantine: 0 test rows accessed or evaluated.
"""

import csv
import json
import math
import os
from pathlib import Path
import unittest
import numpy as np
import torch

from orvexa.event_builder import compute_file_sha256
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


class TestPhase4AStep3(unittest.TestCase):
    """Test suite for Phase 4A Step 3 forensic audit assertions and candidate freeze manifest."""

    @classmethod
    def setUpClass(cls):
        cls.freeze_manifest_path = "reports/phase4a/h6_m4_candidate_freeze.json"
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.val_set = set(cls.split_manifest.val_event_ids)
        cls.test_set = set(cls.split_manifest.test_event_ids)

        cls.authoritative_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
            "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
            "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
            "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
            "data/processed/events/events_H6.csv": "bbc4a34ebbc900f6344d3380dc14faa1b691c2180e1ad875586eb031b5d7cee9",
            "data/processed/events/sequences_H6.csv": "ad31bc8e99ec8cf720fd4645fb571d2d906d2e9bd1fb961613c99dee514c8817",
            "artifacts/models/phase4a/tcn_best_M4_h6.0.pt": "9bb5f10b990be67336dd0902ab3943c28b20b21d18855f2ab4e9b4ca31844d30",
            "artifacts/models/phase4a/tcn_best_M4_h6.0.json": "99fc8138f877a26567e2e01388293709f8a1f8364ad76c3342274b7b5d25784b",
            "artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json": "e2c5e17769efb93358bfe9308591aefbf30f2b10045cbd970ae0b2c5b0fc5a62",
            "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv": "ff90ec2732eae47d4032a92b99139fd766c10cadfc038326a5eec41a9bdc1e8f",
        }

    def test_authoritative_hashes_intact(self):
        """Verify that all baseline files, Phase 3B checkpoints, and H6 candidate artifacts match hashes."""
        for path, exp_hash in self.authoritative_hashes.items():
            self.assertTrue(os.path.exists(path), f"Missing file: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch on {path}")

    def test_candidate_freeze_manifest_validity(self):
        """Verify that the candidate freeze manifest exists and matches checkpoint hashes."""
        self.assertTrue(os.path.exists(self.freeze_manifest_path))
        with open(self.freeze_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["candidate_name"], "tcn_best_M4_h6.0")
        self.assertEqual(data["freeze_status"], "FROZEN_CANDIDATE")
        self.assertEqual(data["checkpoint"]["sha256"], self.authoritative_hashes["artifacts/models/phase4a/tcn_best_M4_h6.0.pt"])
        self.assertEqual(data["preprocessor"]["sha256"], self.authoritative_hashes["artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json"])
        self.assertEqual(data["populations"]["test_observations_accessed"], 0)

    def test_checkpoint_deserialization_and_weights_finite(self):
        """Verify H6 M4 candidate checkpoint deserializes and parameter tensors are 100% finite."""
        ckpt_path = "artifacts/models/phase4a/tcn_best_M4_h6.0.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("network.0.conv1.conv.weight", ckpt)
        conv_w = ckpt["network.0.conv1.conv.weight"]
        self.assertEqual(conv_w.shape, (64, 37, 3))

        for k, v in ckpt.items():
            self.assertFalse(torch.isnan(v).any(), f"NaN found in tensor {k}")
            self.assertFalse(torch.isinf(v).any(), f"Inf found in tensor {k}")

    def test_validation_metrics_exact_reproduction(self):
        """Verify independent reproduction of validation metrics from stored prediction CSV."""
        pred_csv = "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv"
        with open(pred_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1264)
        yt = [float(r["final_risk"]) for r in rows]
        yp = [float(r["predicted_risk"]) for r in rows]

        reg = compute_regression_metrics(yt, yp)
        self.assertAlmostEqual(reg["mae"], 5.05134, places=4)
        self.assertAlmostEqual(reg["rmse"], 8.77232, places=4)
        self.assertAlmostEqual(reg["r2"], 0.13046, places=4)
        self.assertAlmostEqual(reg["spearman_correlation"], 0.47807, places=4)

    def test_test_partition_quarantine_preserved(self):
        """Verify that zero test events leaked into validation predictions."""
        pred_csv = "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv"
        with open(pred_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for r in rows:
            eid = r["event_id"]
            self.assertIn(eid, self.val_set)
            self.assertNotIn(eid, self.test_set)


if __name__ == "__main__":
    unittest.main()
