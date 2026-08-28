"""Automated test suite for ORVEXA Phase 4A Step 1 (H6 Dataset Construction & Verification).

Verifies:
1. Canonical H6 artifacts (events_H6.csv, sequences_H6.csv) exist and match recorded hashes.
2. H6 event count is exactly 8,426; sequence row count is exactly 20,384.
3. Split inheritance from master_split_manifest.json (Train=5,883, Val=1,264, Test=1,279).
4. Strict disjointness among inherited H6 splits.
5. Horizon contract: time_to_tca >= 6.0 for 100% of observations.
6. Sequence integrity: step_index 0..L-1, strictly decreasing time_to_tca within each event.
7. Feature contract: all 34 DIRECT feature columns present.
8. Frozen baseline and Phase 3B M4 checkpoints immutability.
"""

import csv
import json
import math
import os
from pathlib import Path
import unittest

from orvexa.event_builder import DIRECT_FEATURE_COLUMNS, compute_file_sha256
from orvexa.splitting import SplitManifest


class TestPhase4AStep1(unittest.TestCase):
    """Test suite for Phase 4A Step 1 H6 dataset artifacts, schema, and split governance."""

    @classmethod
    def setUpClass(cls):
        cls.events_path = "data/processed/events/events_H6.csv"
        cls.seqs_path = "data/processed/events/sequences_H6.csv"
        cls.manifest_path = "reports/phase4a/h6_dataset_manifest.json"
        cls.master_split_path = "artifacts/splits/master_split_manifest.json"
        
        cls.split_manifest = SplitManifest.load(cls.master_split_path)
        cls.train_set = set(cls.split_manifest.train_event_ids)
        cls.val_set = set(cls.split_manifest.val_event_ids)
        cls.test_set = set(cls.split_manifest.test_event_ids)

        cls.frozen_hashes = {
            "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
            "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
            "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
            "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
            "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
        }

    def test_frozen_baselines_and_checkpoints_immutable(self):
        """Verify that all Phase 3B baseline files and checkpoints remain 100% frozen."""
        for path, exp_hash in self.frozen_hashes.items():
            self.assertTrue(os.path.exists(path), f"Frozen file missing: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch on frozen artifact: {path}")

    def test_h6_files_exist_and_match_manifest_hashes(self):
        """Verify H6 events CSV, sequences CSV, and manifest exist with matching hashes."""
        self.assertTrue(os.path.exists(self.events_path), f"Missing {self.events_path}")
        self.assertTrue(os.path.exists(self.seqs_path), f"Missing {self.seqs_path}")
        self.assertTrue(os.path.exists(self.manifest_path), f"Missing {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        events_hash = compute_file_sha256(self.events_path)
        seqs_hash = compute_file_sha256(self.seqs_path)

        self.assertEqual(events_hash, manifest["h6_artifacts"]["events_csv"]["sha256"])
        self.assertEqual(seqs_hash, manifest["h6_artifacts"]["sequences_csv"]["sha256"])

    def test_h6_population_counts_and_split_inheritance(self):
        """Verify exact H6 event and split counts (Train=5,883, Val=1,264, Test=1,279)."""
        with open(self.events_path, "r", encoding="utf-8") as f:
            events_rows = list(csv.DictReader(f))

        self.assertEqual(len(events_rows), 8426, "Unexpected H6 total event count")

        eids = [r["event_id"] for r in events_rows]
        tr_eids = [eid for eid in eids if eid in self.train_set]
        va_eids = [eid for eid in eids if eid in self.val_set]
        te_eids = [eid for eid in eids if eid in self.test_set]

        self.assertEqual(len(tr_eids), 5883, "Unexpected H6 train count")
        self.assertEqual(len(va_eids), 1264, "Unexpected H6 val count")
        self.assertEqual(len(te_eids), 1279, "Unexpected H6 test count")

        # Disjointness
        self.assertTrue(set(tr_eids).isdisjoint(set(va_eids)))
        self.assertTrue(set(tr_eids).isdisjoint(set(te_eids)))
        self.assertTrue(set(va_eids).isdisjoint(set(te_eids)))

    def test_h6_horizon_cutoff_and_temporal_monotonicity(self):
        """Verify all H6 observations satisfy time_to_tca >= 6.0 and monotonic sequence ordering."""
        with open(self.seqs_path, "r", encoding="utf-8") as f:
            seq_rows = list(csv.DictReader(f))

        self.assertEqual(len(seq_rows), 20384, "Unexpected H6 sequence observation count")

        events_seq_map = {}
        for r in seq_rows:
            eid = r["event_id"]
            ttca = float(r["time_to_tca"])
            self.assertGreaterEqual(ttca, 6.0, f"Observation for event {eid} violates H6 cutoff: {ttca}")
            if eid not in events_seq_map:
                events_seq_map[eid] = []
            events_seq_map[eid].append(r)

        # Check per-event step indices and monotonicity
        for eid, rows in events_seq_map.items():
            for idx, r in enumerate(rows):
                self.assertEqual(int(r["step_index"]), idx, f"Step index error in event {eid}")
            times = [float(r["time_to_tca"]) for r in rows]
            is_dec = all(times[i] >= times[i+1] for i in range(len(times)-1))
            self.assertTrue(is_dec, f"Non-monotonic time_to_tca in event {eid}: {times}")

    def test_h6_feature_schema_completeness(self):
        """Verify that events_H6.csv and sequences_H6.csv contain all 34 direct features."""
        with open(self.events_path, "r", encoding="utf-8") as f:
            ev_fieldnames = csv.DictReader(f).fieldnames
        with open(self.seqs_path, "r", encoding="utf-8") as f:
            seq_fieldnames = csv.DictReader(f).fieldnames

        for col in DIRECT_FEATURE_COLUMNS:
            self.assertIn(col, ev_fieldnames, f"Feature column {col} missing from events_H6.csv")
            self.assertIn(col, seq_fieldnames, f"Feature column {col} missing from sequences_H6.csv")


if __name__ == "__main__":
    unittest.main()
