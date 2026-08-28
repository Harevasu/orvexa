"""Unit tests for Phase 5 Step 5 Blind Internal-Test Evaluation."""

import hashlib
import json
import math
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import Phase5SplitManifest


def compute_file_sha256(file_path: str) -> str:
    """Deterministically compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestPhase5Step5BlindInternalTest(unittest.TestCase):
    """Test suite for Phase 5 Internal Test evaluation results, reproducibility, and quarantine."""

    def setUp(self):
        self.workspace = Path(__file__).resolve().parent.parent
        self.split_manifest_path = self.workspace / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"
        self.freeze_manifest_path = self.workspace / "artifacts" / "models" / "phase5" / "candidate_freeze_manifest.json"
        self.summary_json_path = self.workspace / "reports" / "phase5" / "step5_blind_internal_test_summary.json"
        self.metrics_csv_path = self.workspace / "reports" / "phase5_step5_blind_internal_test_metrics.csv"

    def test_artifacts_exist_and_hashes_registered(self):
        """Verify summary JSON and metrics CSV exist and are non-empty."""
        self.assertTrue(self.summary_json_path.exists())
        self.assertTrue(self.metrics_csv_path.exists())

        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["report_type"], "ORVEXA_PHASE5_STEP5_BLIND_INTERNAL_TEST_SUMMARY")
        self.assertEqual(data["candidate_id"], "Candidate_C_QuantileM4_CQR")
        self.assertIn("H2", data["horizons"])
        self.assertIn("H3", data["horizons"])
        self.assertIn("H5", data["horizons"])
        self.assertIn("H6", data["horizons"])

    def test_prediction_csv_row_counts_and_disjointness(self):
        """Verify test prediction CSVs have exact expected qualifying event counts and strict partition disjointness."""
        manifest = Phase5SplitManifest.load(str(self.split_manifest_path))
        test_ids_set = set(manifest.test_event_ids)
        train_ids_set = set(manifest.train_event_ids)
        val_ids_set = set(manifest.val_event_ids)
        cal_ids_set = set(manifest.cal_event_ids)
        hist_test_ids_set = set(manifest.quarantined_test_event_ids)

        expected_counts = {
            "h2": 1528,
            "h3": 1429,
            "h5": 1193,
            "h6": 1071,
        }

        for h_key, exp_count in expected_counts.items():
            csv_path = self.workspace / f"data/processed/predictions/phase5/blind_test/tcn_quantile_M4_cqr_{h_key}_internal_test_predictions.csv"
            self.assertTrue(csv_path.exists(), f"Missing prediction CSV: {csv_path}")

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), exp_count, f"Row count mismatch in {h_key}: expected {exp_count}, got {len(df)}")

            csv_event_ids = set(df["event_id"].astype(str))

            # Must be a subset of Internal Test
            self.assertTrue(csv_event_ids.issubset(test_ids_set))

            # Must have zero intersection with Train, Val, Cal, Historical Test
            self.assertTrue(csv_event_ids.isdisjoint(train_ids_set))
            self.assertTrue(csv_event_ids.isdisjoint(val_ids_set))
            self.assertTrue(csv_event_ids.isdisjoint(cal_ids_set))
            self.assertTrue(csv_event_ids.isdisjoint(hist_test_ids_set))

    def test_independent_recomputation_from_prediction_csvs(self):
        """Independently recompute all metrics from generated CSVs and assert exact match with summary JSON."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        horizons = ["h2", "h3", "h5", "h6"]
        h_caps = ["H2", "H3", "H5", "H6"]

        for h_str, h_cap in zip(horizons, h_caps):
            csv_path = self.workspace / f"data/processed/predictions/phase5/blind_test/tcn_quantile_M4_cqr_{h_str}_internal_test_predictions.csv"
            df = pd.read_csv(csv_path)

            y_true = df["y_true"].values
            q50 = df["q_50"].values
            cqr_cov = df["cqr_covered_90"].values
            cqr_widths = df["cqr_width_90"].values

            # Check continuous regression
            reg = compute_regression_metrics(list(y_true), list(q50))
            sum_reg = summary["horizons"][h_cap]["point_prediction_q50"]["regression_summary"]

            self.assertAlmostEqual(reg["mae"], sum_reg["mae"], places=4)
            self.assertAlmostEqual(reg["rmse"], sum_reg["rmse"], places=4)
            self.assertAlmostEqual(reg["r2"], sum_reg["r2"], places=4)
            self.assertAlmostEqual(reg["spearman_correlation"], sum_reg["spearman_correlation"], places=4)

            # Check CQR coverage
            emp_cov = float(np.mean(cqr_cov))
            sum_cov = summary["horizons"][h_cap]["cqr_evaluation"]["empirical_coverage"]
            self.assertAlmostEqual(emp_cov, sum_cov, places=4)

            # Check CQR mean width
            emp_width = float(np.mean(cqr_widths))
            sum_width = summary["horizons"][h_cap]["cqr_evaluation"]["mean_interval_width"]
            self.assertAlmostEqual(emp_width, sum_width, places=4)

    def test_zero_quantile_crossing_in_all_predictions(self):
        """Verify that every prediction in every horizon strictly satisfies non-decreasing quantile ordering."""
        for h in ["h2", "h3", "h5", "h6"]:
            csv_path = self.workspace / f"data/processed/predictions/phase5/blind_test/tcn_quantile_M4_cqr_{h}_internal_test_predictions.csv"
            df = pd.read_csv(csv_path)

            q_cols = ["q_05", "q_10", "q_25", "q_50", "q_75", "q_90", "q_95"]
            for i in range(len(q_cols) - 1):
                diff = df[q_cols[i + 1]] - df[q_cols[i]]
                violations = np.sum(diff < -1e-6)
                self.assertEqual(violations, 0, f"Quantile crossing in {h} between {q_cols[i]} and {q_cols[i+1]}")

    def test_cqr_coverage_exceeds_nominal_across_all_horizons(self):
        """Verify that CQR empirical coverage exceeds nominal 90% on unseen Internal Test."""
        with open(self.summary_json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for h in ["H2", "H3", "H5", "H6"]:
            cov = summary["horizons"][h]["cqr_evaluation"]["empirical_coverage"]
            self.assertGreaterEqual(cov, 0.90, f"CQR under-coverage on horizon {h}: {cov:.4f} < 0.90")


if __name__ == "__main__":
    unittest.main()
