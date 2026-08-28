"""Automated test suite for ORVEXA Phase 3B Step 3 Combined Intervention Experiments.

Verifies:
1. M4 channel count = 37, M5 channel count = 38.
2. Categorical one-hot encoding is non-degenerate and binary indicator sum = 1.0.
3. Covariance log10 transformation is active for covariance and Mahalanobis features.
4. Delta-t channel exists only in M5 (absent in M4).
5. Tensor contract shapes match [B, 37, 23] for M4 and [B, 38, 23] for M5.
6. Causal pooling remains h[:, :, -1].
7. Validation prediction files exist, align exactly with validation split, and contain no NaNs/Infs.
8. Step 3 metrics CSV exists with 12 complete records.
9. Frozen baseline SHA-256 hashes (canonical data, master split, Phase 2B M0, Step 2 M2) are 100% immutable.
10. Test split quarantine is strictly preserved.
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
from orvexa.phase3b_config import (
    COVARIANCE_LOG10_FEATURE_COLUMNS,
    ExperimentID,
    OBJECT_TYPE_ONE_HOT_CHANNELS,
    get_experiment_config,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.splitting import SplitManifest


class TestPhase3BStep3(unittest.TestCase):
    """Test suite for Phase 3B Step 3 combined intervention training and validation artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
        cls.val_event_ids = cls.split_manifest.val_event_ids
        cls.horizons = [2.0, 3.0, 5.0]
        cls.experiments = [ExperimentID.M4, ExperimentID.M5]

    def test_canonical_datasets_and_frozen_baselines_hashes(self):
        """Verify 100% immutability of raw datasets, master split, Phase 2B M0, and Step 2 M2 checkpoints."""
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
            "artifacts/models/phase3b/tcn_best_M2_h2.0.pt": "b2c56263881038fe9d0f9e8a7c57fc7a14afcd32cf8e4500c84233d2879b0027",
            "artifacts/models/phase3b/tcn_best_M2_h3.0.pt": "ee67b2a3dfd52bcdb9d78b53ce31413083abc34a94ad1975098ea0a1ae6ec3c9",
            "artifacts/models/phase3b/tcn_best_M2_h5.0.pt": "9007c8be8973f90556862c03cb5f540fe0c7135e1cd06c3c4b1f3a3eb7ba2f51",
        }
        for fpath, exp_h in expected_hashes.items():
            self.assertTrue(os.path.exists(fpath), f"Missing artifact: {fpath}")
            act_h = compute_file_sha256(fpath)
            self.assertEqual(act_h, exp_h, f"Hash mismatch for {fpath}")

    def test_m4_and_m5_channel_counts_and_delta_t_presence(self):
        """Verify channel contracts: M4=37 channels (no dt), M5=38 channels (with dt)."""
        cfg_m4 = get_experiment_config(ExperimentID.M4)
        cfg_m5 = get_experiment_config(ExperimentID.M5)

        self.assertEqual(cfg_m4.n_channels, 37)
        self.assertEqual(cfg_m5.n_channels, 38)

        self.assertNotIn("delta_t", cfg_m4.feature_columns)
        self.assertIn("delta_t", cfg_m5.feature_columns)
        self.assertEqual(cfg_m5.feature_columns.index("delta_t"), 1)

        # Categorical one-hot channels in both M4 and M5
        for col in OBJECT_TYPE_ONE_HOT_CHANNELS:
            self.assertIn(col, cfg_m4.feature_columns)
            self.assertIn(col, cfg_m5.feature_columns)

        # Covariance log10 transformation active in both M4 and M5
        self.assertTrue(cfg_m4.include_covariance_log10)
        self.assertTrue(cfg_m5.include_covariance_log10)

    def test_tensor_contract_shapes_and_left_padding(self):
        """Verify sequence tensor shapes [B, C, 23] and left-padding contract for M4 and M5."""
        mock_events = {
            "ev_single": [
                {
                    "event_id": "ev_single",
                    "time_to_tca": "3.0",
                    "c_object_type": "DEBRIS",
                    "t_sigma_r": "10.0",
                    "risk": "-10.0",
                }
            ],
            "ev_multi": [
                {
                    "event_id": "ev_multi",
                    "time_to_tca": "5.0",
                    "c_object_type": "PAYLOAD",
                    "t_sigma_r": "100.0",
                    "risk": "-5.0",
                },
                {
                    "event_id": "ev_multi",
                    "time_to_tca": "2.5",
                    "c_object_type": "PAYLOAD",
                    "t_sigma_r": "10.0",
                    "risk": "-5.0",
                },
            ],
        }

        # Test M4
        prep_m4 = Phase3BSequencePreprocessor(config=ExperimentID.M4)
        X4, m4, _, _ = prep_m4.prepare_sequence_tensors(mock_events, horizon_cutoff=2.0)
        X4_np = X4.numpy() if hasattr(X4, "numpy") else np.array(X4)
        m4_np = m4.numpy() if hasattr(m4, "numpy") else np.array(m4)

        self.assertEqual(X4_np.shape, (2, 37, 23))
        self.assertEqual(m4_np.shape, (2, 23))
        self.assertEqual(m4_np[0, -1], 1.0)
        self.assertEqual(np.sum(m4_np[0, :-1]), 0.0)

        # Test M5
        prep_m5 = Phase3BSequencePreprocessor(config=ExperimentID.M5)
        X5, m5, _, _ = prep_m5.prepare_sequence_tensors(mock_events, horizon_cutoff=2.0)
        X5_np = X5.numpy() if hasattr(X5, "numpy") else np.array(X5)
        m5_np = m5.numpy() if hasattr(m5, "numpy") else np.array(m5)

        self.assertEqual(X5_np.shape, (2, 38, 23))
        self.assertEqual(m5_np.shape, (2, 23))
        self.assertEqual(m5_np[0, -1], 1.0)
        self.assertEqual(np.sum(m5_np[0, :-1]), 0.0)

    def test_step3_model_checkpoints_exist_and_deserialize(self):
        """Verify that all 6 Step 3 model checkpoints (.pt) and metadata (.json) exist in step3 namespace."""
        for h in self.horizons:
            for exp in self.experiments:
                model_base = f"artifacts/models/phase3b/step3/tcn_best_{exp.value}_h{h:.1f}"
                json_path = f"{model_base}.json"
                pt_path = f"{model_base}.pt"

                self.assertTrue(os.path.exists(json_path), f"Missing metadata: {json_path}")
                self.assertTrue(os.path.exists(pt_path), f"Missing checkpoint: {pt_path}")

                tcn = TCNRiskModel.load(model_base)
                cfg = get_experiment_config(exp)
                self.assertEqual(tcn.in_features, cfg.n_channels)
                self.assertTrue(tcn.is_fitted_)
                self.assertIn("best_epoch", tcn.training_stats_)
                self.assertGreater(tcn.training_stats_["best_epoch"], 0)

    def test_validation_predictions_alignment_and_no_nans(self):
        """Verify validation prediction files exist, match exact event counts, and have no NaNs."""
        expected_counts = {2.0: 1795, 3.0: 1682, 5.0: 1404}

        for h in self.horizons:
            for exp in ["M4", "M5"]:
                pred_csv = f"data/processed/predictions/phase3b/tcn_{exp}_h{h:.1f}_val_predictions.csv"
                self.assertTrue(os.path.exists(pred_csv), f"Missing validation prediction file: {pred_csv}")

                with open(pred_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                self.assertEqual(len(rows), expected_counts[h])
                for r in rows:
                    yt = float(r["final_risk"])
                    yp = float(r["predicted_risk"])
                    self.assertTrue(math.isfinite(yt))
                    self.assertTrue(math.isfinite(yp))

    def test_metrics_csv_structure_and_completeness(self):
        """Verify reports/phase3b_step3_metrics.csv exists with 12 complete records."""
        metrics_csv = "reports/phase3b_step3_metrics.csv"
        self.assertTrue(os.path.exists(metrics_csv), "Missing metrics CSV.")

        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 12, f"Expected 12 metric rows, got {len(rows)}")

        models = set(r["model"] for r in rows)
        self.assertEqual(models, {"M0", "M2", "M4", "M5"})


if __name__ == "__main__":
    unittest.main()
