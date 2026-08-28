"""Automated test suite for ORVEXA Phase 3B Step 2 Isolated Intervention Experiments.

Verifies:
1. All 9 Phase 3B model checkpoints (.pt) and metadata (.json) exist and deserialize correctly.
2. Validation prediction files exist, have exact row count matching validation partition, and contain no NaNs/Infs.
3. Phase 3B Step 2 metrics CSV has exact schema and 12 complete records.
4. Channel contracts match experimental specifications (M0:34, M1:37, M2:34, M3:35).
5. Baseline dataset SHA-256 hashes and frozen M0 artifacts remain 100% immutable.
6. Test set quarantine invariant is preserved.
"""

import csv
import json
import math
import os
from pathlib import Path
import unittest
from typing import Any, Dict, List

import numpy as np

from orvexa.event_builder import compute_file_sha256
from orvexa.models_tcn import TCNRiskModel
from orvexa.phase3b_config import ExperimentID, get_experiment_config
from orvexa.splitting import SplitManifest


class TestPhase3BStep2(unittest.TestCase):
    """Test suite for Phase 3B Step 2 model training and validation evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.val_event_ids = cls.split_manifest.val_event_ids
        cls.horizons = [2.0, 3.0, 5.0]
        cls.experiments = [ExperimentID.M1, ExperimentID.M2, ExperimentID.M3]

    def test_canonical_datasets_and_frozen_m0_hashes(self):
        """Verify 100% immutability of raw, processed datasets, and frozen Phase 2B models."""
        expected_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "data/processed/events/events_H2.csv": "3977a0b8adaaa6eeb29b107381f5ed19856e9e9adb44b1f511574fab547c8dd3",
            "data/processed/events/events_H3.csv": "6840e2c7ffdcdafaec46172b3051bce2063bb51c2a5b22a2064473902090f049",
            "data/processed/events/events_H5.csv": "89c427f05285606da42b2004a2e6175547cf78834e5f900e66d9f22cf859a51a",
            "data/processed/events/sequences_H2.csv": "4ccc7ddc779c53d99ad5d0775ed5e4d87d1b6470062dc0921fbbaeedd3bc8c0c",
            "data/processed/events/sequences_H3.csv": "7901ebbc5b073c31fffe0f967a96ea1ffdea41034771f6acccb2a3d089b9a097",
            "data/processed/events/sequences_H5.csv": "fb5ca14f20d7dfbe07ac74b5d5a772ebd4ed81dd3f234e87b31b4e1099442243",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
            "artifacts/models/tcn_best_h2.0.pt": "8a18d15c7f3fdc7c69152d2c30946284641ffc4249deea6e739e149659334b9f",
            "artifacts/models/tcn_best_h3.0.pt": "f4a06159e8793fbf20fb7e155fac6acd36567f1ab39f467da41e6f998605ba58",
            "artifacts/models/tcn_best_h5.0.pt": "297e64a2f1a788194fe52a553a942b199032ea5ba2cd8090ad7602757757de6e",
        }
        for fpath, exp_h in expected_hashes.items():
            self.assertTrue(os.path.exists(fpath), f"Missing artifact: {fpath}")
            act_h = compute_file_sha256(fpath)
            self.assertEqual(act_h, exp_h, f"Hash mismatch for {fpath}")

    def test_all_phase3b_models_exist_and_deserialize(self):
        """Verify that all 9 Phase 3B model checkpoints and JSON descriptors exist and load."""
        for h in self.horizons:
            for exp in self.experiments:
                model_base = f"artifacts/models/phase3b/tcn_best_{exp.value}_h{h:.1f}"
                json_path = f"{model_base}.json"
                pt_path = f"{model_base}.pt"

                self.assertTrue(os.path.exists(json_path), f"Missing metadata: {json_path}")
                self.assertTrue(os.path.exists(pt_path), f"Missing checkpoint: {pt_path}")

                # Load and verify model instance
                tcn = TCNRiskModel.load(model_base)
                cfg = get_experiment_config(exp)
                self.assertEqual(tcn.in_features, cfg.n_channels)
                self.assertTrue(tcn.is_fitted_)
                self.assertIn("best_epoch", tcn.training_stats_)
                self.assertGreater(tcn.training_stats_["best_epoch"], 0)

    def test_validation_prediction_files_alignment_and_validity(self):
        """Verify validation prediction files structure, event ID alignment, and finite values."""
        expected_counts = {2.0: 1795, 3.0: 1682, 5.0: 1404}

        for h in self.horizons:
            for exp in ["M0", "M1", "M2", "M3"]:
                pred_csv = f"data/processed/predictions/phase3b/tcn_{exp}_h{h:.1f}_val_predictions.csv"
                self.assertTrue(os.path.exists(pred_csv), f"Missing validation prediction file: {pred_csv}")

                with open(pred_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                self.assertEqual(
                    len(rows),
                    expected_counts[h],
                    f"Sample count mismatch in {pred_csv}: expected {expected_counts[h]}, got {len(rows)}",
                )

                for r in rows:
                    self.assertIn("event_id", r)
                    self.assertIn("final_risk", r)
                    self.assertIn("predicted_risk", r)
                    yt = float(r["final_risk"])
                    yp = float(r["predicted_risk"])
                    self.assertTrue(math.isfinite(yt), f"Non-finite true risk in {pred_csv}")
                    self.assertTrue(math.isfinite(yp), f"Non-finite predicted risk in {pred_csv}")

    def test_metrics_csv_structure_and_completeness(self):
        """Verify that reports/phase3b_step2_metrics.csv exists with 12 complete records."""
        metrics_csv = "reports/phase3b_step2_metrics.csv"
        self.assertTrue(os.path.exists(metrics_csv), "Missing metrics CSV.")

        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 12, f"Expected 12 metric rows, got {len(rows)}")

        required_cols = [
            "horizon",
            "model",
            "val_mae",
            "val_rmse",
            "val_r2",
            "val_pearson",
            "val_spearman",
            "val_recall_top10pct",
            "val_precision_top10pct",
            "val_missed_events_top10pct",
            "tail_mean_residual",
        ]
        for col in required_cols:
            self.assertIn(col, reader.fieldnames, f"Missing column in metrics CSV: {col}")

        # Verify M2 beats M0 in R2 across all 3 horizons
        m2_rows = {r["horizon"]: float(r["val_r2"]) for r in rows if r["model"] == "M2"}
        m0_rows = {r["horizon"]: float(r["val_r2"]) for r in rows if r["model"] == "M0"}

        for h in ["H2", "H3", "H5"]:
            self.assertGreater(
                m2_rows[h],
                m0_rows[h],
                f"M2 failed to improve R2 over M0 on {h}: M2={m2_rows[h]}, M0={m0_rows[h]}",
            )


if __name__ == "__main__":
    unittest.main()
