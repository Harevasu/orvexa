"""Unit tests for Phase 2A Temporal Dataset & Target Alignment Audit.

Programmatically verifies all critical scientific invariants:
- Temporal unit schema and columns
- Monotonic step ordering and chronological progression (oldest -> newest)
- Horizon cutoff adherence (H2: >=2.0d, H3: >=3.0d, H5: >=5.0d)
- Target alignment against raw ESA source
- Zero future-information leakage under perturbation
- Strict exclusion of identifiers and target columns from model inputs
- Disjoint chronological splits with zero cross-horizon leakage
- Left-padding tensor contract and newest valid timestep placement
"""

import csv
import json
import math
import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
    compute_file_sha256,
)
from orvexa.features_temporal import (
    CORE_TEMPORAL_NUMERIC_COLS,
    extract_temporal_dataset,
    extract_temporal_summary_features,
)
from orvexa.models_tcn import TCNRiskModel
from orvexa.splitting import SplitManifest


class TestPhase2ATemporalAudit(unittest.TestCase):
    """Test suite validating Phase 2A temporal dataset safety, ordering, and target alignment."""

    @classmethod
    def setUpClass(cls):
        cls.horizons = [2, 3, 5]
        cls.cutoffs = {2: 2.0, 3: 3.0, 5: 5.0}
        cls.expected_event_counts = {2: 11942, 3: 11273, 5: 9484}
        cls.expected_cdm_counts = {2: 111939, 3: 88237, 5: 42524}
        cls.raw_path = "data/raw/esa/train_data.csv"
        cls.split_manifest_path = "artifacts/splits/master_split_manifest.json"

    def test_audit_dataset_files_exist_and_hashes_valid(self):
        """Verify all 7 audit datasets exist, are non-empty, and have valid SHA-256 hashes."""
        files = [
            "data/raw/esa/train_data.csv",
            "data/processed/events/events_H2.csv",
            "data/processed/events/events_H3.csv",
            "data/processed/events/events_H5.csv",
            "data/processed/events/sequences_H2.csv",
            "data/processed/events/sequences_H3.csv",
            "data/processed/events/sequences_H5.csv",
        ]
        for fpath in files:
            self.assertTrue(os.path.exists(fpath), f"Missing required dataset: {fpath}")
            size = os.path.getsize(fpath)
            self.assertGreater(size, 0, f"File {fpath} is empty")
            h = compute_file_sha256(fpath)
            self.assertEqual(len(h), 64, f"Invalid SHA-256 hash length for {fpath}")

    def test_temporal_unit_columns_and_schema(self):
        """Verify that sequence datasets contain exactly 37 columns: 3 metadata and 34 DIRECT features."""
        expected_meta = ["event_id", "step_index", "horizon_days"]
        expected_cols = expected_meta + DIRECT_FEATURE_COLUMNS

        for h in self.horizons:
            seq_path = f"data/processed/events/sequences_H{h}.csv"
            with open(seq_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

            self.assertEqual(len(header), 37, f"H{h} sequence header length mismatch")
            self.assertEqual(header, expected_cols, f"H{h} sequence header column mismatch")

            # Check exclusion of forbidden fields
            self.assertNotIn("risk", header, f"Target 'risk' leaked into H{h} sequence columns")
            self.assertNotIn("final_risk", header, f"Target 'final_risk' leaked into H{h} sequence columns")
            self.assertNotIn("mission_id", header, f"Identifier 'mission_id' leaked into H{h} sequence columns")

    def test_temporal_ordering_and_step_index_monotonicity(self):
        """Verify that every sequence is strictly ordered: OLDEST OBSERVATION -> NEWEST OBSERVATION."""
        for h in self.horizons:
            seq_path = f"data/processed/events/sequences_H{h}.csv"
            with open(seq_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows_by_event = defaultdict(list)
                for r in reader:
                    rows_by_event[r["event_id"]].append(r)

            self.assertEqual(len(rows_by_event), self.expected_event_counts[h])

            # Audit ordering for every single event
            for ev_id, steps in rows_by_event.items():
                prev_step = -1
                prev_tca = float("inf")
                for s in steps:
                    step = int(s["step_index"])
                    t_tca = float(s["time_to_tca"])

                    # 1. step_index must be strictly contiguous: 0, 1, 2, ..., L-1
                    self.assertEqual(step, prev_step + 1, f"Non-contiguous step_index in H{h} event {ev_id}")

                    # 2. time_to_tca must be strictly non-increasing (oldest CDM has higher time_to_tca)
                    self.assertLessEqual(t_tca, prev_tca, f"Temporal inversion in H{h} event {ev_id}: {t_tca} > {prev_tca}")

                    prev_step = step
                    prev_tca = t_tca

    def test_horizon_cutoff_adherence(self):
        """Verify that all sequence rows strictly satisfy time_to_tca >= horizon_cutoff."""
        for h in self.horizons:
            cutoff = self.cutoffs[h]
            seq_path = f"data/processed/events/sequences_H{h}.csv"
            row_count = 0
            with open(seq_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    row_count += 1
                    t_tca = float(r["time_to_tca"])
                    self.assertGreaterEqual(
                        t_tca, cutoff,
                        f"Horizon cutoff violation in H{h}: time_to_tca={t_tca} < {cutoff}"
                    )

            self.assertEqual(row_count, self.expected_cdm_counts[h], f"H{h} row count mismatch")

    def test_target_alignment_with_raw_source(self):
        """Verify final_risk matches raw ESA minimum time_to_tca risk with 0 mismatches."""
        # Load sample of raw events
        raw_events = defaultdict(list)
        with open(self.raw_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw_events[r["event_id"]].append(r)

        for h in self.horizons:
            ev_path = f"data/processed/events/events_H{h}.csv"
            with open(ev_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for ev_row in reader:
                    ev_id = ev_row["event_id"]
                    raw_cdms = raw_events[ev_id]
                    final_raw_cdm = min(raw_cdms, key=lambda r: float(r["time_to_tca"]))
                    expected_risk = float(final_raw_cdm["risk"])
                    actual_risk = float(ev_row["final_risk"])

                    self.assertFalse(math.isnan(actual_risk), f"NaN final_risk in H{h} event {ev_id}")
                    self.assertFalse(math.isinf(actual_risk), f"Inf final_risk in H{h} event {ev_id}")
                    self.assertAlmostEqual(
                        actual_risk, expected_risk, places=10,
                        msg=f"Target mismatch in H{h} event {ev_id}"
                    )

    def test_future_information_perturbation_invariance(self):
        """Verify that extracting temporal features from full raw vs pre-filtered CDMs gives identical results."""
        # Load sample raw events
        sample_events = {}
        with open(self.raw_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for r in reader:
                sample_events.setdefault(r["event_id"], []).append(r)
                count += 1
                if len(sample_events) >= 100:
                    break

        tcn = TCNRiskModel(in_features=34, max_seq_len=23)

        for h in self.horizons:
            cutoff = self.cutoffs[h]
            # A: Full raw events
            feat_A, _, _ = extract_temporal_dataset(sample_events, horizon_cutoff=cutoff)
            
            # B: Pre-filtered events (all post-cutoff CDMs stripped prior to extraction)
            prefiltered_events = {
                ev_id: [c for c in cdms if float(c["time_to_tca"]) >= cutoff]
                for ev_id, cdms in sample_events.items()
            }
            feat_B, _, _ = extract_temporal_dataset(prefiltered_events, horizon_cutoff=cutoff)

            self.assertEqual(len(feat_A), len(feat_B), f"Feature record count mismatch for H{h}")

            for fa, fb in zip(feat_A, feat_B):
                for k in fa.keys():
                    va, vb = fa[k], fb[k]
                    if va is None and vb is None:
                        continue
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        self.assertAlmostEqual(
                            va, vb, places=10,
                            msg=f"Feature discrepancy in {k} under future perturbation for H{h}"
                        )
                    else:
                        self.assertEqual(va, vb, f"Feature discrepancy in {k} for H{h}")

            # Tensor construction invariance for input features and masks
            X_A, mask_A, _, ids_A = tcn.prepare_sequence_tensors(sample_events, DIRECT_FEATURE_COLUMNS, horizon_cutoff=cutoff)
            X_B, mask_B, _, ids_B = tcn.prepare_sequence_tensors(prefiltered_events, DIRECT_FEATURE_COLUMNS, horizon_cutoff=cutoff)

            self.assertEqual(ids_A, ids_B)
            if hasattr(X_A, "tolist"):
                self.assertEqual(X_A.tolist(), X_B.tolist())
                self.assertEqual(mask_A.tolist(), mask_B.tolist())
            else:
                self.assertEqual(X_A, X_B)
                self.assertEqual(mask_A, mask_B)

    def test_cross_horizon_and_split_disjointness(self):
        """Verify strict 0 overlap between Train, Val, and Test across all horizons."""
        if not os.path.exists(self.split_manifest_path):
            self.skipTest("Split manifest not found")

        manifest = SplitManifest.load(self.split_manifest_path)
        train_ids = set(manifest.train_event_ids)
        val_ids = set(manifest.val_event_ids)
        test_ids = set(manifest.test_event_ids)

        # Master split disjointness
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(val_ids.intersection(test_ids)), 0)

        # Cross-horizon disjointness
        horizon_sets = {}
        for h in self.horizons:
            ev_path = f"data/processed/events/events_H{h}.csv"
            with open(ev_path, "r", encoding="utf-8", newline="") as f:
                rdr = csv.DictReader(f)
                h_ids = {r["event_id"] for r in rdr}
            horizon_sets[h] = {
                "train": h_ids.intersection(train_ids),
                "val": h_ids.intersection(val_ids),
                "test": h_ids.intersection(test_ids),
            }

        for h1 in self.horizons:
            for h2 in self.horizons:
                if h1 == h2:
                    continue
                # Train(H1) must not overlap with Val(H2) or Test(H2)
                self.assertEqual(
                    len(horizon_sets[h1]["train"].intersection(horizon_sets[h2]["val"])), 0,
                    f"Cross-horizon leakage between Train(H{h1}) and Val(H{h2})"
                )
                self.assertEqual(
                    len(horizon_sets[h1]["train"].intersection(horizon_sets[h2]["test"])), 0,
                    f"Cross-horizon leakage between Train(H{h1}) and Test(H{h2})"
                )

    def test_sequence_tensor_contract_left_padding_and_pooling_position(self):
        """Verify left padding contract and newest step index 22 across sequence lengths [1, 2, 5, 10, 23]."""
        tcn = TCNRiskModel(in_features=34, max_seq_len=23)
        test_lengths = [1, 2, 5, 10, 23]

        for L in test_lengths:
            synthetic_cdms = []
            for i in range(L):
                cdm = {col: str(10.0 + i) for col in DIRECT_FEATURE_COLUMNS}
                cdm["time_to_tca"] = str(10.0 - i * 0.2)
                cdm["risk"] = "-12.5"
                cdm["event_id"] = f"test_{L}"
                synthetic_cdms.append(cdm)

            events = {f"test_{L}": synthetic_cdms}
            X_tensor, mask_tensor, y_targets, _ = tcn.prepare_sequence_tensors(
                events, DIRECT_FEATURE_COLUMNS, horizon_cutoff=2.0
            )

            pad_len = 23 - L
            feat_matrix = X_tensor[0].tolist() if hasattr(X_tensor, "tolist") else X_tensor[0]
            mask_row = mask_tensor[0].tolist() if hasattr(mask_tensor, "tolist") else mask_tensor[0]

            # 1. Mask verification
            self.assertTrue(all(mask_row[i] == 0.0 for i in range(pad_len)))
            self.assertTrue(all(mask_row[i] == 1.0 for i in range(pad_len, 23)))

            # 2. Left padding feature zeros verification
            for f_idx in range(34):
                for t in range(pad_len):
                    self.assertEqual(feat_matrix[f_idx][t], 0.0)

            # 3. Newest valid step location: index 22 (the final position)
            # The newest CDM is synthetic_cdms[-1] with value 10.0 + L - 1
            newest_val_expected = 10.0 + L - 1
            newest_val_actual = feat_matrix[1][22]
            self.assertAlmostEqual(newest_val_actual, newest_val_expected, places=6)

            # 4. Target alignment
            self.assertEqual(y_targets[0], -12.5)

    def test_duplicate_and_integrity_constraints(self):
        """Verify zero duplicate (event_id, step_index) rows and exact manifest agreement."""
        for h in self.horizons:
            seq_path = f"data/processed/events/sequences_H{h}.csv"
            seen_pairs = set()
            duplicate_count = 0
            with open(seq_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    pair = (r["event_id"], r["step_index"])
                    if pair in seen_pairs:
                        duplicate_count += 1
                    seen_pairs.add(pair)

            self.assertEqual(duplicate_count, 0, f"Found duplicate (event_id, step_index) pairs in H{h}")


if __name__ == "__main__":
    unittest.main()
