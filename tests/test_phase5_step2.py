"""Unit tests for Phase 5 Step 2 Split Design and Governance."""

import json
import os
import unittest
from pathlib import Path

from orvexa.splitting import (
    Phase5SplitManifest,
    SplitManifest,
    make_phase5_splits,
)


class TestPhase5Step2(unittest.TestCase):
    """Test suite ensuring strict zero-leakage, quarantine, and integrity for Phase 5 splits."""

    def setUp(self):
        self.workspace = Path(__file__).resolve().parent.parent
        self.master_manifest_path = self.workspace / "artifacts" / "splits" / "master_split_manifest.json"
        self.phase5_artifact_path = self.workspace / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"
        self.phase5_report_path = self.workspace / "reports" / "phase5" / "step2_split_manifest.json"

    def test_phase5_manifest_files_exist_and_match(self):
        """Verify both artifact and report manifest files exist and have identical contents."""
        self.assertTrue(self.phase5_artifact_path.exists(), f"Missing {self.phase5_artifact_path}")
        self.assertTrue(self.phase5_report_path.exists(), f"Missing {self.phase5_report_path}")

        with open(self.phase5_artifact_path, "r", encoding="utf-8") as f:
            art_data = json.load(f)
        with open(self.phase5_report_path, "r", encoding="utf-8") as f:
            rep_data = json.load(f)

        self.assertEqual(art_data, rep_data, "Artifact and report manifests must be bit-for-bit identical.")

    def test_historical_test_quarantine_enforced(self):
        """Verify zero event overlap between any Phase 5 partition and the locked historical test partition."""
        with open(self.master_manifest_path, "r", encoding="utf-8") as f:
            master = json.load(f)
        hist_test_ids = set(master["test_event_ids"])
        self.assertEqual(len(hist_test_ids), 1974, "Historical master test partition must contain exactly 1,974 events.")

        with open(self.phase5_artifact_path, "r", encoding="utf-8") as f:
            p5 = json.load(f)

        tr_set = set(p5["train_event_ids"])
        va_set = set(p5["val_event_ids"])
        ca_set = set(p5["cal_event_ids"])
        te_set = set(p5["test_event_ids"])

        # Quarantine verification
        self.assertTrue(tr_set.isdisjoint(hist_test_ids), "Phase 5 Train overlaps with locked historical test set!")
        self.assertTrue(va_set.isdisjoint(hist_test_ids), "Phase 5 Validation overlaps with locked historical test set!")
        self.assertTrue(ca_set.isdisjoint(hist_test_ids), "Phase 5 Calibration overlaps with locked historical test set!")
        self.assertTrue(te_set.isdisjoint(hist_test_ids), "Phase 5 Internal Test overlaps with locked historical test set!")

    def test_phase5_partitions_pairwise_disjoint(self):
        """Verify all four Phase 5 partitions are strictly pairwise disjoint."""
        with open(self.phase5_artifact_path, "r", encoding="utf-8") as f:
            p5 = json.load(f)

        tr = set(p5["train_event_ids"])
        va = set(p5["val_event_ids"])
        ca = set(p5["cal_event_ids"])
        te = set(p5["test_event_ids"])

        self.assertEqual(len(tr), 6708, "Train count must be 6,708 (60%).")
        self.assertEqual(len(va), 1677, "Validation count must be 1,677 (15%).")
        self.assertEqual(len(ca), 1118, "Calibration count must be 1,118 (10%).")
        self.assertEqual(len(te), 1677, "Internal Test count must be 1,677 (15%).")

        total_unique = len(tr | va | ca | te)
        self.assertEqual(total_unique, 11180, "Total unique eligible events must be 11,180.")

        self.assertTrue(tr.isdisjoint(va), "Train and Validation overlap!")
        self.assertTrue(tr.isdisjoint(ca), "Train and Calibration overlap!")
        self.assertTrue(tr.isdisjoint(te), "Train and Internal Test overlap!")
        self.assertTrue(va.isdisjoint(ca), "Validation and Calibration overlap!")
        self.assertTrue(va.isdisjoint(te), "Validation and Internal Test overlap!")
        self.assertTrue(ca.isdisjoint(te), "Calibration and Internal Test overlap!")

    def test_chronological_integrity_and_contiguous_blocks(self):
        """Verify that event assignments follow strictly contiguous, monotonically increasing chronological blocks."""
        with open(self.phase5_artifact_path, "r", encoding="utf-8") as f:
            p5 = json.load(f)

        tr = p5["train_event_ids"]
        va = p5["val_event_ids"]
        ca = p5["cal_event_ids"]
        te = p5["test_event_ids"]

        combined = tr + va + ca + te
        self.assertEqual(len(combined), 11180)

        # Convert to integer index to verify monotonic continuity
        int_ids = [int(x) for x in combined]
        self.assertEqual(int_ids, list(range(11180)), "Events must be strictly sequential 0..11179.")

        # Verify exact partition boundaries
        self.assertEqual(int(tr[0]), 0)
        self.assertEqual(int(tr[-1]), 6707)
        self.assertEqual(int(va[0]), 6708)
        self.assertEqual(int(va[-1]), 8384)
        self.assertEqual(int(ca[0]), 8385)
        self.assertEqual(int(ca[-1]), 9502)
        self.assertEqual(int(te[0]), 9503)
        self.assertEqual(int(te[-1]), 11179)

    def test_horizon_coverage_completeness(self):
        """Verify horizon coverage dictionary is present for H2, H3, H5, H6 with valid counts."""
        with open(self.phase5_artifact_path, "r", encoding="utf-8") as f:
            p5 = json.load(f)

        coverage = p5["horizon_coverage"]
        for h in ["H2", "H3", "H5", "H6"]:
            self.assertIn(h, coverage, f"Missing horizon {h} in coverage report.")
            h_data = coverage[h]
            self.assertGreater(h_data["total_eligible_events"], 0)
            self.assertGreater(h_data["total_eligible_observations"], 0)

            for part in ["train", "validation", "calibration", "internal_test"]:
                p_stats = h_data["partitions"][part]
                self.assertGreater(p_stats["events_count"], 0)
                self.assertGreater(p_stats["observations_count"], 0)
                self.assertGreaterEqual(p_stats["critical_events_count"], 0)
                self.assertGreater(p_stats["mean_sequence_length"], 0.0)

    def test_phase5_split_manifest_dataclass_validation(self):
        """Test Phase5SplitManifest rejection on partition overlap and quarantine violation."""
        # Overlap within partitions
        with self.assertRaises(ValueError):
            Phase5SplitManifest(
                train_event_ids=["1", "2"],
                val_event_ids=["2", "3"],  # overlap "2"
                cal_event_ids=["4"],
                test_event_ids=["5"],
            )

        # Overlap with quarantined test partition
        with self.assertRaises(ValueError):
            Phase5SplitManifest(
                train_event_ids=["1", "2"],
                val_event_ids=["3"],
                cal_event_ids=["4"],
                test_event_ids=["5"],
                quarantined_test_event_ids=["2", "99"],  # overlap "2" with train
            )


if __name__ == "__main__":
    unittest.main()
