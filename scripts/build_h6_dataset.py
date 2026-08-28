"""ORVEXA Phase 4A Step 1: Canonical H6 Dataset Construction and Verification Script.

Strictly reads data/raw/esa/train_data.csv using only DIRECT features at H = 6.0 days.
Inherits split assignments from artifacts/splits/master_split_manifest.json.

Outputs:
- data/processed/events/events_H6.csv
- data/processed/events/sequences_H6.csv
- reports/phase4a/h6_dataset_manifest.json
"""

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    build_event_prefixes_from_raw,
    compute_file_sha256,
)
from orvexa.splitting import SplitManifest

# Authoritative Frozen Hashes
FROZEN_HASHES = {
    "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
    "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
    "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
    "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
    "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
}


def verify_frozen_state() -> bool:
    print("=== 1. VERIFYING FROZEN BASELINE & CHECKPOINT STATE ===")
    all_ok = True
    for path, expected in FROZEN_HASHES.items():
        if not os.path.exists(path):
            print(f"  [FAIL - MISSING] {path}")
            all_ok = False
        else:
            actual = compute_file_sha256(path)
            if actual == expected:
                print(f"  [PASS] {path} -> {actual[:16]}...")
            else:
                print(f"  [FAIL - MISMATCH] {path}\n    Expected: {expected}\n    Actual:   {actual}")
                all_ok = False
    return all_ok


def compute_percentiles(values: List[float], percentiles=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)) -> Dict[str, float]:
    """Compute exact percentiles."""
    if not values:
        return {f"p{int(p*100)}": None for p in percentiles}
    sorted_v = sorted(values)
    n = len(sorted_v)
    res = {}
    for p in percentiles:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = sorted_v[int(k)]
        else:
            d0 = sorted_v[int(f)] * (c - k)
            d1 = sorted_v[int(c)] * (k - f)
            val = d0 + d1
        p_name = f"p{int(p*100)}" if p * 100 == int(p * 100) else f"p{p*100:.1f}"
        res[p_name] = float(val)
    return res


def main():
    if not verify_frozen_state():
        print("ERROR: Frozen baseline integrity check failed! Halting.")
        sys.exit(1)

    raw_path = "data/raw/esa/train_data.csv"
    manifest_path = "artifacts/splits/master_split_manifest.json"
    
    print("\n=== 2. LOADING RAW DATA AND BUILDING H6 PREFIXES ===")
    raw_rows_by_event: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    raw_row_count = 0

    with open(raw_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_row_count += 1
            ev_id = str(row["event_id"])
            raw_rows_by_event[ev_id].append(row)

    total_unique_events = len(raw_rows_by_event)
    print(f"Loaded {raw_row_count:,} raw rows across {total_unique_events:,} unique events.")

    # Horizon H = 6.0 days
    horizon_days = 6.0
    print(f"Executing build_event_prefixes_from_raw for H = {horizon_days} days...")
    records, stats = build_event_prefixes_from_raw(
        raw_rows_by_event=raw_rows_by_event,
        horizon_days=horizon_days,
        feature_columns=DIRECT_FEATURE_COLUMNS,
    )
    n_events = len(records)
    print(f"Generated {n_events:,} eligible H6 events.")

    # 3. Write H6 events and sequences
    os.makedirs("data/processed/events", exist_ok=True)
    os.makedirs("reports/phase4a", exist_ok=True)

    events_csv_path = "data/processed/events/events_H6.csv"
    seq_csv_path = "data/processed/events/sequences_H6.csv"

    event_fieldnames = [
        "event_id",
        "horizon_days",
        "sequence_length",
        "earliest_time_to_tca",
        "anchor_time_to_tca",
        "final_risk",
    ] + DIRECT_FEATURE_COLUMNS

    seq_fieldnames = [
        "event_id",
        "step_index",
        "horizon_days",
    ] + DIRECT_FEATURE_COLUMNS

    seq_lens = []
    final_risks = []
    feature_missing_counts = defaultdict(int)
    total_horizon_cdms = 0

    with open(events_csv_path, "w", encoding="utf-8", newline="") as f_ev, \
         open(seq_csv_path, "w", encoding="utf-8", newline="") as f_seq:

        ev_writer = csv.DictWriter(f_ev, fieldnames=event_fieldnames)
        seq_writer = csv.DictWriter(f_seq, fieldnames=seq_fieldnames)

        ev_writer.writeheader()
        seq_writer.writeheader()

        for rec in records:
            seq_lens.append(rec.sequence_length)
            final_risks.append(rec.final_risk)
            total_horizon_cdms += rec.sequence_length

            # Anchor CDM is the last CDM in the prefix
            anchor_cdm = rec.cdms[-1]

            ev_row = {
                "event_id": rec.event_id,
                "horizon_days": rec.horizon_days,
                "sequence_length": rec.sequence_length,
                "earliest_time_to_tca": rec.earliest_time_to_tca,
                "anchor_time_to_tca": rec.anchor_time_to_tca,
                "final_risk": rec.final_risk,
            }
            for col in DIRECT_FEATURE_COLUMNS:
                ev_row[col] = anchor_cdm[col]
            ev_writer.writerow(ev_row)

            # Sequence rows
            for step_idx, cdm in enumerate(rec.cdms):
                seq_row = {
                    "event_id": rec.event_id,
                    "step_index": step_idx,
                    "horizon_days": rec.horizon_days,
                }
                for col in DIRECT_FEATURE_COLUMNS:
                    val = cdm[col]
                    seq_row[col] = val
                    if val is None or str(val).strip().lower() in ("nan", "null", "none", ""):
                        feature_missing_counts[col] += 1
                seq_writer.writerow(seq_row)

    print(f"  Wrote {events_csv_path} ({n_events:,} events)")
    print(f"  Wrote {seq_csv_path} ({total_horizon_cdms:,} sequence CDMs)")

    # 4. Verify Split Inheritance
    print("\n=== 3. VERIFYING SPLIT INHERITANCE ===")
    manifest = SplitManifest.load(manifest_path)
    tr_set = set(manifest.train_event_ids)
    va_set = set(manifest.val_event_ids)
    te_set = set(manifest.test_event_ids)

    h6_eids = [r.event_id for r in records]
    h6_eids_set = set(h6_eids)

    h6_tr = [eid for eid in h6_eids if eid in tr_set]
    h6_va = [eid for eid in h6_eids if eid in va_set]
    h6_te = [eid for eid in h6_eids if eid in te_set]

    # Check unassigned
    unassigned = [eid for eid in h6_eids if eid not in tr_set and eid not in va_set and eid not in te_set]

    print(f"H6 Event Split Counts:")
    print(f"  Train:      {len(h6_tr):,}")
    print(f"  Validation: {len(h6_va):,}")
    print(f"  Test:       {len(h6_te):,}")
    print(f"  Total:      {len(h6_eids):,}")
    print(f"  Unassigned: {len(unassigned)}")

    # Check disjointness
    tr_va_overlap = len(set(h6_tr).intersection(set(h6_va)))
    tr_te_overlap = len(set(h6_tr).intersection(set(h6_te)))
    va_te_overlap = len(set(h6_va).intersection(set(h6_te)))
    print(f"Disjointness Overlaps: Tr-Val={tr_va_overlap}, Tr-Test={tr_te_overlap}, Val-Test={va_te_overlap}")

    assert len(h6_tr) == 5883, f"Expected 5,883 Train, got {len(h6_tr)}"
    assert len(h6_va) == 1264, f"Expected 1,264 Val, got {len(h6_va)}"
    assert len(h6_te) == 1279, f"Expected 1,279 Test, got {len(h6_te)}"
    assert len(unassigned) == 0, f"Expected 0 unassigned, got {len(unassigned)}"
    assert tr_va_overlap == 0 and tr_te_overlap == 0 and va_te_overlap == 0, "Split overlap detected!"
    print("Split inheritance verified: 100% PASS.")

    # 5. Verify Horizon Contract
    print("\n=== 4. VERIFYING HORIZON CONTRACT (time_to_tca >= 6.0) ===")
    with open(seq_csv_path, "r", encoding="utf-8") as f:
        seq_rows = list(csv.DictReader(f))

    all_ttcas = [float(r["time_to_tca"]) for r in seq_rows]
    min_ttca = min(all_ttcas)
    max_ttca = max(all_ttcas)
    below_6_count = sum(1 for t in all_ttcas if t < 6.0)
    missing_ttca_count = sum(1 for r in seq_rows if r.get("time_to_tca") is None or r["time_to_tca"] == "")

    print(f"Sequence Row Count: {len(seq_rows):,}")
    print(f"Minimum time_to_tca: {min_ttca:.6f} days")
    print(f"Maximum time_to_tca: {max_ttca:.6f} days")
    print(f"Observations < 6.0 days: {below_6_count}")
    print(f"Missing time_to_tca:     {missing_ttca_count}")

    assert below_6_count == 0, f"Found {below_6_count} observations below 6.0 days!"
    assert missing_ttca_count == 0, f"Found {missing_ttca_count} missing time_to_tca values!"
    print("Horizon contract verified: 100% PASS.")

    # 6. Verify Event and Sequence Structure
    print("\n=== 5. VERIFYING EVENT AND SEQUENCE STRUCTURE ===")
    single_obs_count = sum(1 for l in seq_lens if l == 1)
    multi_obs_count = sum(1 for l in seq_lens if l > 1)
    avg_len = sum(seq_lens) / len(seq_lens)
    med_len = float(np.median(seq_lens))

    print(f"Total Qualifying Events: {n_events:,}")
    print(f"Total Sequence Observations: {total_horizon_cdms:,}")
    print(f"Sequence Lengths: Min={min(seq_lens)}, Max={max(seq_lens)}, Mean={avg_len:.4f}, Median={med_len:.1f}")
    print(f"Single-Observation Events (L=1): {single_obs_count:,} ({single_obs_count/n_events*100:.2f}%)")
    print(f"Multi-Observation Events (L>1):  {multi_obs_count:,} ({multi_obs_count/n_events*100:.2f}%)")

    # Verify per-event monotonicity and step indices
    events_in_seq = defaultdict(list)
    for r in seq_rows:
        events_in_seq[r["event_id"]].append(r)

    non_monotonic_seqs = 0
    step_idx_errors = 0
    for eid, s_rows in events_in_seq.items():
        # verify step indices 0 .. len-1
        for i, row in enumerate(s_rows):
            if int(row["step_index"]) != i:
                step_idx_errors += 1
        # verify decreasing time_to_tca
        ttca_seq = [float(r["time_to_tca"]) for r in s_rows]
        is_dec = all(ttca_seq[i] >= ttca_seq[i+1] for i in range(len(ttca_seq)-1))
        if not is_dec:
            non_monotonic_seqs += 1

    print(f"Step index discrepancies: {step_idx_errors}")
    print(f"Non-monotonic event sequences: {non_monotonic_seqs}")
    assert step_idx_errors == 0, "Step index errors detected!"
    assert non_monotonic_seqs == 0, "Non-monotonic sequences detected!"
    print("Sequence structure verified: 100% PASS.")

    # 7. Verify Feature Schema
    print("\n=== 6. VERIFYING FEATURE SCHEMA ===")
    events_header = list(csv.DictReader(open(events_csv_path, "r", encoding="utf-8")).fieldnames)
    seqs_header = list(csv.DictReader(open(seq_csv_path, "r", encoding="utf-8")).fieldnames)
    
    print(f"Events CSV Columns ({len(events_header)}): {events_header[:6]} + {len(events_header)-6} features")
    print(f"Sequences CSV Columns ({len(seqs_header)}): {seqs_header[:3]} + {len(seqs_header)-3} features")

    for col in DIRECT_FEATURE_COLUMNS:
        assert col in events_header, f"Missing {col} in events_H6.csv"
        assert col in seqs_header, f"Missing {col} in sequences_H6.csv"
    print("Feature schema verified: 100% PASS.")

    # 8. Compute Checksums & Write Manifest
    print("\n=== 7. COMPUTING SHA-256 HASHES & CREATING MANIFEST ===")
    events_sha256 = compute_file_sha256(events_csv_path)
    seqs_sha256 = compute_file_sha256(seq_csv_path)
    raw_sha256 = compute_file_sha256(raw_path)
    manifest_sha256 = compute_file_sha256(manifest_path)

    events_size = os.path.getsize(events_csv_path)
    seqs_size = os.path.getsize(seq_csv_path)

    print(f"  {events_csv_path}: {events_size:,} bytes | SHA-256: {events_sha256}")
    print(f"  {seq_csv_path}: {seqs_size:,} bytes | SHA-256: {seqs_sha256}")

    h6_manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 4A Step 1",
        "horizon_days": 6.0,
        "source_dataset": {
            "path": raw_path,
            "sha256": raw_sha256,
            "total_raw_rows": raw_row_count,
            "total_unique_events": total_unique_events,
        },
        "split_manifest": {
            "path": manifest_path,
            "sha256": manifest_sha256,
        },
        "h6_artifacts": {
            "events_csv": {
                "path": events_csv_path,
                "size_bytes": events_size,
                "sha256": events_sha256,
                "row_count": n_events,
                "feature_columns_count": len(DIRECT_FEATURE_COLUMNS),
            },
            "sequences_csv": {
                "path": seq_csv_path,
                "size_bytes": seqs_size,
                "sha256": seqs_sha256,
                "row_count": total_horizon_cdms,
                "feature_columns_count": len(DIRECT_FEATURE_COLUMNS),
            },
        },
        "population_statistics": {
            "total_qualifying_events": n_events,
            "total_observations": total_horizon_cdms,
            "split_counts": {
                "train": len(h6_tr),
                "validation": len(h6_va),
                "test": len(h6_te),
            },
            "sequence_lengths": {
                "min": min(seq_lens),
                "max": max(seq_lens),
                "mean": round(avg_len, 4),
                "median": med_len,
                "single_obs_count": single_obs_count,
                "multi_obs_count": multi_obs_count,
                "single_obs_pct": round(single_obs_count / n_events * 100, 2),
                "multi_obs_pct": round(multi_obs_count / n_events * 100, 2),
                "percentiles": compute_percentiles(seq_lens),
            },
            "lead_times": {
                "min_time_to_tca": min_ttca,
                "max_time_to_tca": max_ttca,
                "below_cutoff_count": below_6_count,
                "missing_count": missing_ttca_count,
            },
            "target_final_risk": {
                "min": min(final_risks),
                "max": max(final_risks),
                "mean": round(sum(final_risks) / len(final_risks), 4),
                "median": float(np.median(final_risks)),
                "percentiles": compute_percentiles(final_risks),
            },
            "feature_missing_counts": {
                col: {
                    "missing_count": feature_missing_counts[col],
                    "missing_pct": round(feature_missing_counts[col] / total_horizon_cdms * 100, 4),
                }
                for col in DIRECT_FEATURE_COLUMNS
            },
        },
    }

    manifest_out_path = "reports/phase4a/h6_dataset_manifest.json"
    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(h6_manifest, f, indent=2)
    print(f"Manifest written to {manifest_out_path}")

    print("\n==================================================")
    print("PHASE 4A STEP 1 H6 DATASET GENERATION & VERIFICATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
