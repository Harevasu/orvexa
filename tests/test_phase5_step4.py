"""Unit tests for Phase 5 Step 4 Candidate Comparison, Calibration Audit & Freeze."""

import hashlib
import json
import math
import os
from pathlib import Path
import unittest

import numpy as np

from orvexa.conformal import SplitConformalCalibrator
from orvexa.models_probabilistic import QuantileTCNRiskModel
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.splitting import Phase5SplitManifest


def compute_file_sha256(file_path: str) -> str:
    """Deterministically compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase5Step4CandidateFreeze(unittest.TestCase):
    """Test suite for Phase 5 candidate comparison, calibration integrity, and freeze manifests."""

    def setUp(self):
        self.workspace = Path(__file__).resolve().parent.parent
        self.split_manifest_path = self.workspace / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"
        self.freeze_manifest_path = self.workspace / "artifacts" / "models" / "phase5" / "candidate_freeze_manifest.json"
        self.freeze_report_path = self.workspace / "reports" / "phase5" / "step4_candidate_freeze_manifest.json"

    def test_candidate_freeze_manifest_exists_and_matches_report(self):
        """Verify candidate freeze manifest exists in both artifacts and reports with identical SHA-256."""
        self.assertTrue(self.freeze_manifest_path.exists())
        self.assertTrue(self.freeze_report_path.exists())

        hash_art = compute_file_sha256(str(self.freeze_manifest_path))
        hash_rep = compute_file_sha256(str(self.freeze_report_path))
        self.assertEqual(hash_art, hash_rep)

        with open(self.freeze_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["manifest_type"], "ORVEXA_PHASE5_CANDIDATE_FREEZE_MANIFEST")
        self.assertEqual(data["selected_candidate"]["candidate_id"], "Candidate_C_QuantileM4_CQR")
        self.assertTrue(data["governance_status"]["candidate_frozen"])

    def test_frozen_artifacts_exist_and_hashes_match_manifest(self):
        """Verify all frozen candidate weights, configurations, calibrators, and preprocessors exist and match hashes."""
        with open(self.freeze_manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frozen = data["frozen_artifacts"]
        for h_key in ["H2", "H3", "H5", "H6"]:
            self.assertIn(h_key, frozen)
            h_data = frozen[h_key]

            for key_prefix in ["model_config", "model_weights", "cqr_calibrator", "preprocessor"]:
                rel_path = h_data[key_prefix]
                exp_hash = h_data[f"{key_prefix}_sha256"]

                full_path = self.workspace / rel_path
                self.assertTrue(full_path.exists(), f"Missing frozen artifact: {rel_path}")

                act_hash = compute_file_sha256(str(full_path))
                self.assertEqual(
                    act_hash,
                    exp_hash,
                    f"Hash mismatch on {rel_path}! Expected {exp_hash}, got {act_hash}",
                )

    def test_calibration_artifacts_reproducibility(self):
        """Verify calibration scores and quantile recalculation from saved calibration artifacts."""
        for h in [2.0, 3.0, 5.0, 6.0]:
            cqr_path = self.workspace / f"artifacts/models/phase5/cqr_calibrator_h{h}.json"
            self.assertTrue(cqr_path.exists())

            calibrator = SplitConformalCalibrator.load(str(cqr_path))
            self.assertTrue(calibrator.is_calibrated_)
            self.assertGreater(calibrator.n_calibration_samples_, 600)

            # Check finite-sample quantile calculation matches
            stored_summary_p90 = calibrator.to_dict()["calibration_score_summary"]["p90"]
            computed_p90 = calibrator.get_conformal_quantile(alpha=0.10)
            self.assertAlmostEqual(stored_summary_p90, computed_p90, places=5)

    def test_quantile_models_load_and_predict_strictly_monotonic(self):
        """Ensure all frozen quantile models load and generate strictly monotonic predictions."""
        for h in [2.0, 3.0, 5.0, 6.0]:
            h_str = int(h) if h.is_integer() else h
            model_path = self.workspace / f"artifacts/models/phase5/tcn_quantile_M4_h{h_str}"
            model = QuantileTCNRiskModel.load(str(model_path))

            # Dummy causal batch [batch=4, in_features=37, time=15]
            x = np.random.randn(4, 37, 15).astype(np.float32)
            preds = model.predict_quantiles(x)

            self.assertEqual(preds.shape, (4, 7))
            for b in range(4):
                for k in range(6):
                    self.assertGreaterEqual(
                        preds[b, k + 1],
                        preds[b, k],
                        f"Quantile crossing in frozen model h={h}, sample={b}, k={k}",
                    )

    def test_split_manifest_immutability(self):
        """Verify the Phase 5 Split Manifest hash remains identical to Step 2 specification."""
        self.assertTrue(self.split_manifest_path.exists())
        act_hash = compute_file_sha256(str(self.split_manifest_path))
        expected_hash = "c304cb88c48b1f2c066c65a69aa947fb5902f8892633b0244f6746d1c37b15d2"
        self.assertEqual(act_hash, expected_hash)

    def test_quarantine_preservation(self):
        """Verify that validation predictions contain zero events from historical or internal test partitions."""
        manifest = Phase5SplitManifest.load(str(self.split_manifest_path))
        quarantined = set(manifest.quarantined_test_event_ids)
        internal_test = set(manifest.test_event_ids)

        for h in [2.0, 3.0, 5.0, 6.0]:
            h_str = f"h{h}"
            pred_csv = self.workspace / f"data/processed/predictions/phase5/tcn_quantile_M4_{h_str}_val_predictions.csv"
            self.assertTrue(pred_csv.exists())

            import pandas as pd
            df = pd.read_csv(pred_csv)
            val_event_ids = set(df["event_id"].astype(str))

            # Check zero intersection with historical locked test
            self.assertTrue(val_event_ids.isdisjoint(quarantined))
            # Check zero intersection with internal test
            self.assertTrue(val_event_ids.isdisjoint(internal_test))


if __name__ == "__main__":
    unittest.main()
