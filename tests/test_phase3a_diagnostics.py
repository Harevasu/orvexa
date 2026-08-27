"""Unit tests for Phase 3A Temporal Model Diagnostic Analysis."""

import csv
import hashlib
import os
import sys
import unittest
from pathlib import Path

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.event_builder import compute_file_sha256
from orvexa.splitting import SplitManifest


class TestPhase3ADiagnostics(unittest.TestCase):
    """Test suite validating Phase 3A diagnostic outputs, figures, and artifact immutability."""

    @classmethod
    def setUpClass(cls):
        cls.horizons = ["H2", "H3", "H5"]
        cls.expected_figures = [
            "prediction_vs_target_scatter.png",
            "residual_distribution.png",
            "mae_vs_sequence_length.png",
            "performance_vs_horizon.png",
            "distribution_comparison.png",
            "ranking_alert_curves.png",
            "tcn_loss_trajectories.png",
        ]
        cls.figure_dir = "reports/figures/phase3a"
        cls.diagnostics_csv = "reports/phase3a_diagnostics.csv"
        cls.report_md = "reports/PHASE_3A_TEMPORAL_DIAGNOSTIC_REPORT.md"

    def test_diagnostics_csv_structure_and_completeness(self):
        """Verify reports/phase3a_diagnostics.csv exists, has correct schema and all sections."""
        self.assertTrue(os.path.exists(self.diagnostics_csv), f"Missing: {self.diagnostics_csv}")

        expected_headers = [
            "Section",
            "Horizon",
            "Model_or_Comparison",
            "Subgroup",
            "N",
            "Metric_1_Name",
            "Metric_1_Value",
            "Metric_2_Name",
            "Metric_2_Value",
            "Metric_3_Name",
            "Metric_3_Value",
            "Metric_4_Name",
            "Metric_4_Value",
            "Metric_5_Name",
            "Metric_5_Value",
            "Notes",
        ]

        with open(self.diagnostics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, expected_headers)
            rows = list(reader)

        self.assertGreater(len(rows), 100, f"Expected >100 diagnostic rows, found {len(rows)}")

        sections_found = {r["Section"] for r in rows}
        expected_sections = {
            "Prediction_Distribution",
            "Residuals_Overall",
            "Residuals_By_Risk_Bin",
            "Performance_By_Sequence_Length",
            "Performance_By_Risk_Regime",
            "High_Risk_Ranking",
            "Bootstrap_Significance",
        }
        self.assertTrue(expected_sections.issubset(sections_found))

    def test_diagnostic_report_structure_and_scientific_sections(self):
        """Verify diagnostic report exists and contains required FACT, INFERENCE, UNKNOWN sections."""
        self.assertTrue(os.path.exists(self.report_md), f"Missing: {self.report_md}")

        with open(self.report_md, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("## 1. Executive Summary & Primary Scientific Inquiry", content)
        self.assertIn("## 2. Prediction Distribution Analysis & Variance Compression", content)
        self.assertIn("## 3. Residual Error Analysis & Risk Bin Breakdown", content)
        self.assertIn("## 4. Performance Stratification by Sequence Length", content)
        self.assertIn("## 5. Performance Stratification by Risk Regime", content)
        self.assertIn("## 6. High-Risk Operational Alert Ranking Diagnostics", content)
        self.assertIn("## 7. Warning Horizon Trajectory Analysis ($H2 \\to H3 \\to H5$)", content)
        self.assertIn("## 8. TCN Training Trajectory & Optimization Diagnostics", content)
        self.assertIn("## 9. Input & Normalization Pipeline Audit", content)
        self.assertIn("## 10. Temporal XGBoost vs. Static XGBoost Head-to-Head", content)
        self.assertIn("## 11. Statistical Significance & Paired Bootstrap Analysis", content)
        self.assertIn("## 12. Visualizations Overview", content)
        self.assertIn("## 13. Rigorous Scientific Conclusions (FACT vs INFERENCE vs UNKNOWN)", content)
        self.assertIn("## 14. Recommended Next Experimental Phase (Phase 3B)", content)

        # Confirm explicit epistemic boundaries
        self.assertIn("### FACTS", content)
        self.assertIn("### INFERENCES", content)
        self.assertIn("### UNKNOWNS", content)

    def test_all_publication_figures_exist_and_non_empty(self):
        """Verify all 7 diagnostic figures were generated and are non-empty PNG files."""
        for fig_name in self.expected_figures:
            fig_path = os.path.join(self.figure_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Figure missing: {fig_path}")
            size_kb = os.path.getsize(fig_path) / 1024.0
            self.assertGreater(size_kb, 50.0, f"Figure {fig_name} is too small ({size_kb:.1f} KB)")

    def test_baseline_dataset_and_split_immutability(self):
        """Verify dataset files and master split manifest remain 100% unmodified."""
        expected_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "data/processed/events/events_H2.csv": "3977a0b8adaaa6eeb29b107381f5ed19856e9e9adb44b1f511574fab547c8dd3",
            "data/processed/events/events_H3.csv": "6840e2c7ffdcdafaec46172b3051bce2063bb51c2a5b22a2064473902090f049",
            "data/processed/events/events_H5.csv": "89c427f05285606da42b2004a2e6175547cf78834e5f900e66d9f22cf859a51a",
        }

        for path, exp_hash in expected_hashes.items():
            actual_hash = compute_file_sha256(path)
            self.assertEqual(actual_hash, exp_hash, f"Dataset file {path} was modified!")

        # Check master split manifest
        split_manifest_path = "artifacts/splits/master_split_manifest.json"
        self.assertTrue(os.path.exists(split_manifest_path))
        manifest = SplitManifest.load(split_manifest_path)
        self.assertEqual(len(manifest.train_event_ids), 9207)
        self.assertEqual(len(manifest.val_event_ids), 1973)
        self.assertEqual(len(manifest.test_event_ids), 1974)
        self.assertEqual(len(manifest.train_event_ids) + len(manifest.val_event_ids) + len(manifest.test_event_ids), 13154)


if __name__ == "__main__":
    unittest.main()
