"""Unit and safety tests for ORVEXA Phase 3B experimental framework.

Verifies:
A. Category encoding deterministic mappings, binary indicators, and unseen handling.
B. Covariance log10 monotonicity, positive finite preservation, and raw data immutability.
C. Delta-t derivation strictly from time_to_tca, non-negativity, initial step handling, and train-only stats.
D. Strict split isolation and zero-leakage during tensor preparation and normalizer fitting.
E. Tensor contract shapes across M0, M1, M2, M3, M4, and M5.
F. Causal left-padding invariants, newest-step alignment at index -1, and validity masking.
G. Cryptographic immutability of frozen canonical baseline datasets and split manifest.
"""

import math
import os
from pathlib import Path
import unittest
from typing import Any, Dict, List

import numpy as np

from orvexa.event_builder import compute_file_sha256, DIRECT_FEATURE_COLUMNS
from orvexa.phase3b_config import (
    COVARIANCE_LOG10_FEATURE_COLUMNS,
    ExperimentID,
    OBJECT_TYPE_ONE_HOT_CHANNELS,
    Phase3BExperimentConfig,
    RAW_OBJECT_TYPE_CATEGORIES,
    get_experiment_config,
)
from orvexa.preprocessing_phase3b import (
    encode_c_object_type_one_hot,
    Phase3BSequencePreprocessor,
)
from orvexa.splitting import SplitManifest


class TestPhase3BFramework(unittest.TestCase):
    """Automated test suite for Phase 3B controlled experimental framework."""

    @classmethod
    def setUpClass(cls):
        cls.split_manifest_path = "artifacts/splits/master_split_manifest.json"
        cls.manifest = SplitManifest.load(cls.split_manifest_path)
        cls.raw_path = "data/raw/esa/train_data.csv"

    def test_canonical_datasets_and_split_immutability(self):
        """Verify that all canonical dataset files and master split manifest remain 100% untouched."""
        expected_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "data/processed/events/events_H2.csv": "3977a0b8adaaa6eeb29b107381f5ed19856e9e9adb44b1f511574fab547c8dd3",
            "data/processed/events/events_H3.csv": "6840e2c7ffdcdafaec46172b3051bce2063bb51c2a5b22a2064473902090f049",
            "data/processed/events/events_H5.csv": "89c427f05285606da42b2004a2e6175547cf78834e5f900e66d9f22cf859a51a",
            "data/processed/events/sequences_H2.csv": "4ccc7ddc779c53d99ad5d0775ed5e4d87d1b6470062dc0921fbbaeedd3bc8c0c",
            "data/processed/events/sequences_H3.csv": "7901ebbc5b073c31fffe0f967a96ea1ffdea41034771f6acccb2a3d089b9a097",
            "data/processed/events/sequences_H5.csv": "fb5ca14f20d7dfbe07ac74b5d5a772ebd4ed81dd3f234e87b31b4e1099442243",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
        }
        for fpath, expected_h in expected_hashes.items():
            self.assertTrue(os.path.exists(fpath), f"Missing file: {fpath}")
            actual_h = compute_file_sha256(fpath)
            self.assertEqual(actual_h, expected_h, f"Hash mismatch for {fpath}")

    def test_category_encoding_deterministic_and_binary(self):
        """Test A: Verify c_object_type encoding mappings, binary outputs, and unknown handling."""
        # Test exact known categories
        debris = encode_c_object_type_one_hot("DEBRIS")
        self.assertEqual(debris, [1.0, 0.0, 0.0, 0.0])

        payload = encode_c_object_type_one_hot("PAYLOAD")
        self.assertEqual(payload, [0.0, 1.0, 0.0, 0.0])

        rocket_body = encode_c_object_type_one_hot("ROCKET BODY")
        self.assertEqual(rocket_body, [0.0, 0.0, 1.0, 0.0])

        rocket_body_alt = encode_c_object_type_one_hot("ROCKET_BODY")
        self.assertEqual(rocket_body_alt, [0.0, 0.0, 1.0, 0.0])

        # Test unknown / fallback categories
        unknown = encode_c_object_type_one_hot("UNKNOWN")
        self.assertEqual(unknown, [0.0, 0.0, 0.0, 1.0])

        tba = encode_c_object_type_one_hot("TBA")
        self.assertEqual(tba, [0.0, 0.0, 0.0, 1.0])

        none_val = encode_c_object_type_one_hot(None)
        self.assertEqual(none_val, [0.0, 0.0, 0.0, 1.0])

        empty_val = encode_c_object_type_one_hot("")
        self.assertEqual(empty_val, [0.0, 0.0, 0.0, 1.0])

        unseen_val = encode_c_object_type_one_hot("MYSTERY_OBJECT_999")
        self.assertEqual(unseen_val, [0.0, 0.0, 0.0, 1.0])

        # Every output must sum to exactly 1.0 (mutually exclusive)
        for cat in ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN", "TBA", "", None, "X"]:
            encoded = encode_c_object_type_one_hot(cat)
            self.assertEqual(sum(encoded), 1.0)
            self.assertTrue(all(v in (0.0, 1.0) for v in encoded))

    def test_covariance_log10_monotonicity_and_finite_preservation(self):
        """Test B: Verify log10 monotonic properties on covariance sigma values."""
        exp_config_m2 = get_experiment_config(ExperimentID.M2)
        prep = Phase3BSequencePreprocessor(config=exp_config_m2)

        # Mock sequence with positive sigma values
        test_cdms = [
            {
                "event_id": "test_ev",
                "time_to_tca": "5.0",
                "risk": "-10.0",
                "t_sigma_r": "1.0",
                "t_sigma_t": "100.0",
                "c_sigma_r": "10.0",
                "c_sigma_t": "63781000.0",  # Sentinel
                "mahalanobis_distance": "0.01",
                "miss_distance": "1000.0",
            },
            {
                "event_id": "test_ev",
                "time_to_tca": "2.5",
                "risk": "-10.0",
                "t_sigma_r": "10.0",
                "t_sigma_t": "1000.0",
                "c_sigma_r": "100.0",
                "c_sigma_t": "63781000.0",
                "mahalanobis_distance": "100.0",
                "miss_distance": "500.0",
            }
        ]

        X_raw, mask, y, ids = prep.prepare_sequence_tensors({"test_ev": test_cdms}, horizon_cutoff=2.0)
        X_np = X_raw.numpy() if hasattr(X_raw, "numpy") else np.array(X_raw)

        # In M2, t_sigma_r is log10-transformed
        col_idx_t_sigma_r = prep.feature_columns.index("t_sigma_r")
        val_step0 = X_np[0, col_idx_t_sigma_r, -2]  # First valid step
        val_step1 = X_np[0, col_idx_t_sigma_r, -1]  # Second valid step

        # log10(1.0) == 0.0, log10(10.0) == 1.0
        self.assertAlmostEqual(val_step0, 0.0, places=5)
        self.assertAlmostEqual(val_step1, 1.0, places=5)
        self.assertGreater(val_step1, val_step0)  # Strict monotonicity preserved

        # Sentinel log10(63781000.0) ~ 7.80469
        col_idx_c_sigma_t = prep.feature_columns.index("c_sigma_t")
        sentinel_log = X_np[0, col_idx_c_sigma_t, -1]
        self.assertAlmostEqual(sentinel_log, math.log10(63781000.0), places=4)
        self.assertTrue(math.isfinite(sentinel_log))

    def test_delta_t_derivation_non_negativity_and_first_step(self):
        """Test C: Verify Delta-t calculation from time_to_tca, initial step zero, and non-negativity."""
        exp_config_m3 = get_experiment_config(ExperimentID.M3)
        prep = Phase3BSequencePreprocessor(config=exp_config_m3)

        # Mock sequence ordered oldest to newest: time_to_tca = [6.0, 5.2, 4.0, 2.5]
        test_cdms = [
            {"event_id": "ev1", "time_to_tca": "6.0", "risk": "-8.0"},
            {"event_id": "ev1", "time_to_tca": "5.2", "risk": "-8.0"},
            {"event_id": "ev1", "time_to_tca": "4.0", "risk": "-8.0"},
            {"event_id": "ev1", "time_to_tca": "2.5", "risk": "-8.0"},
        ]

        X_raw, mask, y, ids = prep.prepare_sequence_tensors({"ev1": test_cdms}, horizon_cutoff=2.0)
        X_np = X_raw.numpy() if hasattr(X_raw, "numpy") else np.array(X_raw)

        dt_col_idx = prep.feature_columns.index("delta_t")
        self.assertEqual(dt_col_idx, 1)  # Immediately follows time_to_tca

        # Valid timesteps are the last 4 positions (-4, -3, -2, -1)
        dt_0 = X_np[0, dt_col_idx, -4]  # Initial step
        dt_1 = X_np[0, dt_col_idx, -3]  # 6.0 - 5.2 = 0.8
        dt_2 = X_np[0, dt_col_idx, -2]  # 5.2 - 4.0 = 1.2
        dt_3 = X_np[0, dt_col_idx, -1]  # 4.0 - 2.5 = 1.5

        self.assertEqual(dt_0, 0.0)
        self.assertAlmostEqual(dt_1, 0.8, places=5)
        self.assertAlmostEqual(dt_2, 1.2, places=5)
        self.assertAlmostEqual(dt_3, 1.5, places=5)

        # All dt values must be >= 0
        self.assertTrue(all(v >= 0.0 for v in [dt_0, dt_1, dt_2, dt_3]))

    def test_tensor_contract_channel_counts_and_left_padding(self):
        """Test E & F: Verify tensor shapes, channel counts, and left padding across M0..M5."""
        configs = {
            ExperimentID.M0: 34,
            ExperimentID.M1: 37,
            ExperimentID.M2: 34,
            ExperimentID.M3: 35,
            ExperimentID.M4: 37,
            ExperimentID.M5: 38,
        }

        mock_event = {
            "ev_single": [
                {"event_id": "ev_single", "time_to_tca": "3.5", "c_object_type": "DEBRIS", "risk": "-12.0"}
            ],
            "ev_multi": [
                {"event_id": "ev_multi", "time_to_tca": "5.0", "c_object_type": "PAYLOAD", "risk": "-5.0"},
                {"event_id": "ev_multi", "time_to_tca": "3.0", "c_object_type": "PAYLOAD", "risk": "-5.0"},
                {"event_id": "ev_multi", "time_to_tca": "2.1", "c_object_type": "PAYLOAD", "risk": "-5.0"},
            ]
        }

        for exp_id, expected_ch in configs.items():
            cfg = get_experiment_config(exp_id)
            self.assertEqual(cfg.n_channels, expected_ch, f"Config channel count mismatch for {exp_id.value}")

            prep = Phase3BSequencePreprocessor(config=cfg)
            X_raw, mask, y, ids = prep.prepare_sequence_tensors(mock_event, horizon_cutoff=2.0)
            X_np = X_raw.numpy() if hasattr(X_raw, "numpy") else np.array(X_raw)
            mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)

            # Shape contract: [Batch=2, Channels=expected_ch, Time=23]
            self.assertEqual(X_np.shape, (2, expected_ch, 23))
            self.assertEqual(mask_np.shape, (2, 23))

            # Left padding invariant: ev_single has 1 valid step (at index -1)
            self.assertEqual(mask_np[0, -1], 1.0)
            self.assertEqual(np.sum(mask_np[0, :-1]), 0.0)
            self.assertEqual(np.sum(X_np[0, :, :-1]), 0.0)  # Padded positions are exactly zero

            # ev_multi has 3 valid steps (indices -3, -2, -1)
            self.assertEqual(mask_np[1, -1], 1.0)
            self.assertEqual(mask_np[1, -2], 1.0)
            self.assertEqual(mask_np[1, -3], 1.0)
            self.assertEqual(np.sum(mask_np[1, :-3]), 0.0)
            self.assertEqual(np.sum(X_np[1, :, :-3]), 0.0)

    def test_split_isolation_and_training_only_normalizer_fitting(self):
        """Test D: Verify that normalizer is fitted strictly on training data and test data is untouched."""
        exp_config = get_experiment_config(ExperimentID.M1)
        prep = Phase3BSequencePreprocessor(config=exp_config)

        # Mock disjoint train and validation sets
        train_events = {
            "tr_1": [{"event_id": "tr_1", "time_to_tca": "4.0", "miss_distance": "100.0", "risk": "-10.0"}],
            "tr_2": [{"event_id": "tr_2", "time_to_tca": "3.0", "miss_distance": "200.0", "risk": "-10.0"}],
        }
        val_events = {
            "val_1": [{"event_id": "val_1", "time_to_tca": "2.5", "miss_distance": "5000.0", "risk": "-10.0"}],
        }

        X_tr, mask_tr, _, _ = prep.prepare_sequence_tensors(train_events, horizon_cutoff=2.0)
        X_val, mask_val, _, _ = prep.prepare_sequence_tensors(val_events, horizon_cutoff=2.0)

        # Fit strictly on train
        prep.fit(X_tr, mask_tr)
        self.assertTrue(prep.is_fitted_)

        miss_dist_idx = prep.feature_columns.index("miss_distance")
        fitted_mean = prep.channel_stats_["miss_distance"]["mean"]

        # Mean of train values (100.0 and 200.0) is 150.0
        self.assertAlmostEqual(fitted_mean, 150.0, places=4)

        # Transform validation: must use train mean (150.0), NOT validation value (5000.0)
        X_val_norm = prep.transform(X_val, mask_val)
        X_val_norm_np = X_val_norm.numpy() if hasattr(X_val_norm, "numpy") else np.array(X_val_norm)

        fitted_std = prep.channel_stats_["miss_distance"]["std"]
        expected_val_norm = (5000.0 - 150.0) / fitted_std
        actual_val_norm = X_val_norm_np[0, miss_dist_idx, -1]
        self.assertAlmostEqual(actual_val_norm, expected_val_norm, places=4)

    def test_preprocessor_manifest_serialization_and_reload(self):
        """Test preprocessor manifest JSON serialization, reloading, and fidelity."""
        for exp_id in [ExperimentID.M0, ExperimentID.M1, ExperimentID.M2, ExperimentID.M3]:
            manifest_file = f"artifacts/preprocessors/phase3b/preprocessor_{exp_id.value}_h2.0.json"
            self.assertTrue(os.path.exists(manifest_file), f"Missing manifest: {manifest_file}")

            prep_reloaded = Phase3BSequencePreprocessor.load(manifest_file)
            self.assertTrue(prep_reloaded.is_fitted_)
            self.assertEqual(prep_reloaded.config.experiment_id, exp_id)
            self.assertGreater(len(prep_reloaded.channel_stats_), 30)


if __name__ == "__main__":
    unittest.main()
