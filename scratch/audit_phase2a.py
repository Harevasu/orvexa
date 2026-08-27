"""ORVEXA Phase 2A Temporal Dataset & Target Alignment Audit Script.

Comprehensive verification of:
1. Checksums (SHA-256)
2. Temporal unit representation
3. Temporal ordering (Oldest -> Newest)
4. Sequence length distribution & percentiles
5. Horizon cutoff enforcement (H2: >=2.0, H3: >=3.0, H5: >=5.0)
6. Target alignment against raw ESA train_data.csv
7. Future information leakage analysis
8. Temporal feature perturbation test
9. Input feature whitelist vs manifest/registry
10. Target leakage check
11. Cross-horizon & split event leakage
12. Sequence tensor contract (left padding, pooling at -1)
13. Variable-length sequence handling (1, 2, 5, 10, 23)
14. Missingness statistics
15. Duplicate & sequence integrity checks
16. Raw -> Sequence traceability (10+ events per horizon)
"""

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
    build_event_prefixes_from_raw,
    compute_file_sha256,
)
from orvexa.features_temporal import (
    CORE_TEMPORAL_NUMERIC_COLS,
    extract_temporal_dataset,
    extract_temporal_summary_features,
)
from orvexa.models_tcn import TCNRiskModel, causal_conv1d_forward_numpy
from orvexa.splitting import SplitManifest


def compute_percentiles(values: List[float], percentiles=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)) -> Dict[str, float]:
    if not values:
        return {f"p{int(p*100)}": 0.0 for p in percentiles}
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


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    else:
        return float(sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def run_full_audit() -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING ORVEXA PHASE 2A COMPREHENSIVE TEMPORAL AUDIT")
    print("=" * 70)

    audit_results: Dict[str, Any] = {}

    # 1. FILE CHECKSUMS BEFORE
    files_to_audit = {
        "raw_train": "data/raw/esa/train_data.csv",
        "events_H2": "data/processed/events/events_H2.csv",
        "events_H3": "data/processed/events/events_H3.csv",
        "events_H5": "data/processed/events/events_H5.csv",
        "sequences_H2": "data/processed/events/sequences_H2.csv",
        "sequences_H3": "data/processed/events/sequences_H3.csv",
        "sequences_H5": "data/processed/events/sequences_H5.csv",
    }

    checksums_initial = {}
    for key, fpath in files_to_audit.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required audit file missing: {fpath}")
        h = compute_file_sha256(fpath)
        size = os.path.getsize(fpath)
        checksums_initial[key] = {"path": fpath, "sha256": h, "size_bytes": size}
        print(f"File {fpath} ({size:,} bytes) SHA-256: {h}")

    audit_results["checksums_initial"] = checksums_initial

    # Load raw ESA data
    print("\nLoading raw ESA dataset...")
    raw_rows_by_event: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    raw_total_rows = 0
    with open(files_to_audit["raw_train"], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_fieldnames = reader.fieldnames
        for r in reader:
            raw_total_rows += 1
            raw_rows_by_event[r["event_id"]].append(r)

    print(f"Raw data: {raw_total_rows:,} rows, {len(raw_rows_by_event):,} unique events.")
    audit_results["raw_stats"] = {
        "total_rows": raw_total_rows,
        "unique_events": len(raw_rows_by_event),
        "raw_columns": raw_fieldnames,
    }

    # Load sequence files & events files
    seq_data: Dict[str, List[Dict[str, str]]] = {}
    seq_by_event: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    events_data: Dict[str, List[Dict[str, str]]] = {}
    events_by_id: Dict[str, Dict[str, Dict[str, str]]] = {}

    for h_key in ["H2", "H3", "H5"]:
        seq_path = files_to_audit[f"sequences_{h_key}"]
        ev_path = files_to_audit[f"events_{h_key}"]

        with open(seq_path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            s_rows = list(rdr)
            seq_data[h_key] = s_rows
            s_by_ev = defaultdict(list)
            for r in s_rows:
                s_by_ev[r["event_id"]].append(r)
            seq_by_event[h_key] = s_by_ev

        with open(ev_path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            e_rows = list(rdr)
            events_data[h_key] = e_rows
            events_by_id[h_key] = {r["event_id"]: r for r in e_rows}

        print(f"{h_key}: {len(events_data[h_key]):,} events, {len(seq_data[h_key]):,} sequence rows.")

    # 2. AUDIT THE TEMPORAL UNIT
    print("\n--- 2. AUDITING TEMPORAL UNIT ---")
    temporal_unit_audit = {}
    for h_key in ["H2", "H3", "H5"]:
        rows = seq_data[h_key]
        first_row = rows[0]
        cols = list(first_row.keys())
        feature_cols = [c for c in cols if c not in ("event_id", "step_index", "horizon_days")]
        temporal_unit_audit[h_key] = {
            "total_columns": len(cols),
            "columns": cols,
            "metadata_cols": ["event_id", "step_index", "horizon_days"],
            "feature_cols_count": len(feature_cols),
            "feature_cols": feature_cols,
        }
    audit_results["temporal_unit"] = temporal_unit_audit

    # 3. TEMPORAL ORDERING AUDIT
    print("\n--- 3. AUDITING TEMPORAL ORDERING ---")
    ordering_results = {}
    for h_key, cutoff in [("H2", 2.0), ("H3", 3.0), ("H5", 5.0)]:
        events_dict = seq_by_event[h_key]
        n_events_checked = len(events_dict)
        monotonic_step_index_violations = 0
        time_to_tca_non_decreasing_violations = 0
        total_steps_checked = 0

        for ev_id, steps in events_dict.items():
            prev_step = -1
            prev_tca = float("inf")
            for r in steps:
                total_steps_checked += 1
                step = int(r["step_index"])
                t_tca = float(r["time_to_tca"])

                # Verify step_index is strictly monotonic: 0, 1, 2, ...
                if step != prev_step + 1:
                    monotonic_step_index_violations += 1

                # Verify time_to_tca is strictly non-increasing (oldest observation has highest time_to_tca, newest has lowest)
                if t_tca > prev_tca:
                    time_to_tca_non_decreasing_violations += 1

                prev_step = step
                prev_tca = t_tca

        ordering_results[h_key] = {
            "events_checked": n_events_checked,
            "total_steps_checked": total_steps_checked,
            "step_index_violations": monotonic_step_index_violations,
            "time_to_tca_ordering_violations": time_to_tca_non_decreasing_violations,
            "is_monotonic_step_index": monotonic_step_index_violations == 0,
            "is_correct_temporal_direction": time_to_tca_non_decreasing_violations == 0,
        }
        print(f"{h_key} Temporal Ordering: {n_events_checked:,} events, {total_steps_checked:,} steps. "
              f"Step index violations: {monotonic_step_index_violations}, Time ordering violations: {time_to_tca_non_decreasing_violations}")

    audit_results["temporal_ordering"] = ordering_results

    # 4. SEQUENCE LENGTHS AUDIT
    print("\n--- 4. AUDITING SEQUENCE LENGTHS ---")
    seq_length_results = {}
    for h_key in ["H2", "H3", "H5"]:
        events_dict = seq_by_event[h_key]
        lengths = [len(steps) for steps in events_dict.values()]
        length_counts = Counter(lengths)
        pcts = compute_percentiles(lengths)
        
        # Verify sequence lengths agree with events CSV
        mismatch_with_events_csv = 0
        for ev_id, steps in events_dict.items():
            ev_rec = events_by_id[h_key].get(ev_id)
            if not ev_rec or int(ev_rec["sequence_length"]) != len(steps):
                mismatch_with_events_csv += 1

        # Verify sequence lengths agree with raw CDMs satisfying time_to_tca >= H
        cutoff = float(h_key[1:])
        mismatch_with_raw_cdms = 0
        for ev_id, steps in events_dict.items():
            raw_cdms = raw_rows_by_event[ev_id]
            expected_count = sum(1 for c in raw_cdms if float(c["time_to_tca"]) >= cutoff)
            if len(steps) != expected_count:
                mismatch_with_raw_cdms += 1

        seq_length_results[h_key] = {
            "unique_events": len(events_dict),
            "total_cdms": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": round(sum(lengths) / len(lengths), 4),
            "median_length": median(lengths),
            "percentiles": pcts,
            "distribution": dict(sorted(length_counts.items())),
            "mismatch_with_events_csv": mismatch_with_events_csv,
            "mismatch_with_raw_cdms": mismatch_with_raw_cdms,
        }
        print(f"{h_key} Lengths: Events={len(events_dict):,}, Min={min(lengths)}, Max={max(lengths)}, "
              f"Mean={seq_length_results[h_key]['mean_length']}, Median={seq_length_results[h_key]['median_length']}, "
              f"p90={pcts['p90']}, p95={pcts['p95']}, p99={pcts['p99']}")
        print(f"  Mismatches (Events CSV: {mismatch_with_events_csv}, Raw CDMs: {mismatch_with_raw_cdms})")

    audit_results["sequence_lengths"] = seq_length_results

    # 5. HORIZON CUTOFF VERIFICATION
    print("\n--- 5. AUDITING HORIZON CUTOFFS ---")
    cutoff_results = {}
    for h_key, cutoff in [("H2", 2.0), ("H3", 3.0), ("H5", 5.0)]:
        rows = seq_data[h_key]
        violations = 0
        min_tca = float("inf")
        max_tca = float("-inf")
        for r in rows:
            t_tca = float(r["time_to_tca"])
            if t_tca < cutoff:
                violations += 1
            if t_tca < min_tca:
                min_tca = t_tca
            if t_tca > max_tca:
                max_tca = t_tca

        cutoff_results[h_key] = {
            "cutoff_days": cutoff,
            "total_rows": len(rows),
            "unique_events": len(seq_by_event[h_key]),
            "min_time_to_tca": min_tca,
            "max_time_to_tca": max_tca,
            "violations_count": violations,
            "is_valid": violations == 0 and min_tca >= cutoff,
        }
        print(f"{h_key} (Cutoff >= {cutoff}d): Checked {len(rows):,} rows across {len(seq_by_event[h_key]):,} events. "
              f"Min TCA: {min_tca:.6f}d, Max TCA: {max_tca:.6f}d, Violations: {violations}")

    audit_results["horizon_cutoffs"] = cutoff_results

    # 6. TARGET ALIGNMENT AUDIT
    print("\n--- 6. AUDITING TARGET ALIGNMENT AGAINST RAW DATA ---")
    target_results = {}
    for h_key, cutoff in [("H2", 2.0), ("H3", 3.0), ("H5", 5.0)]:
        events_dict = seq_by_event[h_key]
        mismatches = 0
        missing_targets = 0
        nan_inf_targets = 0
        checked_events = 0

        # Also test model pipeline target alignment
        # Convert raw_rows_by_event for eligible events
        eligible_raw_events = {ev_id: raw_rows_by_event[ev_id] for ev_id in events_dict.keys()}
        
        # Test extract_temporal_dataset
        feat_recs, temp_targets, temp_ids = extract_temporal_dataset(
            eligible_raw_events, horizon_cutoff=cutoff
        )
        temp_target_map = {ev_id: t for ev_id, t in zip(temp_ids, temp_targets)}

        for ev_id in events_dict.keys():
            checked_events += 1
            raw_cdms = raw_rows_by_event[ev_id]
            final_raw_cdm = min(raw_cdms, key=lambda r: float(r["time_to_tca"]))
            raw_final_risk = float(final_raw_cdm["risk"])

            if math.isnan(raw_final_risk) or math.isinf(raw_final_risk):
                nan_inf_targets += 1

            ev_row = events_by_id[h_key].get(ev_id)
            if not ev_row or "final_risk" not in ev_row:
                missing_targets += 1
                continue

            ev_final_risk = float(ev_row["final_risk"])
            if abs(raw_final_risk - ev_final_risk) > 1e-12:
                mismatches += 1

            # Check temporal pipeline target
            pipe_target = temp_target_map.get(ev_id)
            if pipe_target is None or abs(raw_final_risk - pipe_target) > 1e-12:
                mismatches += 1

        target_results[h_key] = {
            "events_checked": checked_events,
            "mismatches": mismatches,
            "missing_targets": missing_targets,
            "nan_inf_targets": nan_inf_targets,
            "is_aligned": mismatches == 0 and missing_targets == 0 and nan_inf_targets == 0,
        }
        print(f"{h_key} Target Alignment: Checked {checked_events:,} events. Mismatches: {mismatches}, "
              f"Missing: {missing_targets}, NaN/Inf: {nan_inf_targets}")

    audit_results["target_alignment"] = target_results

    # 7 & 8. FUTURE INFORMATION & TEMPORAL FEATURE LEAKAGE PERTURBATION TEST
    print("\n--- 7 & 8. CRITICAL FUTURE-INFORMATION AUDIT & PERTURBATION TEST ---")
    perturbation_results = {}
    for h_key, cutoff in [("H2", 2.0), ("H3", 3.0), ("H5", 5.0)]:
        events_dict = seq_by_event[h_key]
        sample_ev_ids = list(events_dict.keys())  # Test all eligible events!
        n_tested = len(sample_ev_ids)
        discrepancies = 0
        max_num_diff = 0.0

        tcn_model = TCNRiskModel(in_features=34, max_seq_len=23)

        # Batch test
        test_events_full = {ev_id: raw_rows_by_event[ev_id] for ev_id in sample_ev_ids}
        test_events_prefiltered = {
            ev_id: [c for c in raw_rows_by_event[ev_id] if float(c["time_to_tca"]) >= cutoff]
            for ev_id in sample_ev_ids
        }

        # A & B: Feature extraction comparison
        for ev_id in sample_ev_ids:
            cdms_full = test_events_full[ev_id]
            cdms_pre = test_events_prefiltered[ev_id]

            # Method A: pass full CDM sequence with horizon_cutoff=cutoff
            feat_A = extract_temporal_summary_features(cdms_full, horizon_cutoff=cutoff)
            # Method B: pass prefiltered CDMs with horizon_cutoff=cutoff
            feat_B = extract_temporal_summary_features(cdms_pre, horizon_cutoff=cutoff)

            if feat_A is None or feat_B is None:
                discrepancies += 1
                continue

            for k in feat_A.keys():
                val_A = feat_A[k]
                val_B = feat_B[k]
                if val_A is None and val_B is None:
                    continue
                if isinstance(val_A, (int, float)) and isinstance(val_B, (int, float)):
                    diff = abs(val_A - val_B)
                    if diff > max_num_diff:
                        max_num_diff = diff
                    if diff > 1e-12:
                        discrepancies += 1
                elif val_A != val_B:
                    discrepancies += 1

        # C: Tensor construction comparison
        X_A, mask_A, y_A, ids_A = tcn_model.prepare_sequence_tensors(
            test_events_full, feature_cols=DIRECT_FEATURE_COLUMNS, horizon_cutoff=cutoff
        )
        X_B, mask_B, y_B, ids_B = tcn_model.prepare_sequence_tensors(
            test_events_prefiltered, feature_cols=DIRECT_FEATURE_COLUMNS, horizon_cutoff=cutoff
        )

        tensor_discrepancies = 0
        if isinstance(X_A, list):
            # Pure python lists
            for i in range(len(X_A)):
                for c in range(len(X_A[0])):
                    for t in range(len(X_A[0][0])):
                        diff = abs(X_A[i][c][t] - X_B[i][c][t])
                        if diff > max_num_diff:
                            max_num_diff = diff
                        if diff > 1e-12:
                            tensor_discrepancies += 1
        else:
            # PyTorch tensors
            diff = (X_A - X_B).abs().max().item()
            if diff > max_num_diff:
                max_num_diff = diff
            if diff > 1e-12:
                tensor_discrepancies += 1

        perturbation_results[h_key] = {
            "events_tested": n_tested,
            "discrepancies": discrepancies + tensor_discrepancies,
            "max_numerical_difference": max_num_diff,
            "is_leakage_free": (discrepancies + tensor_discrepancies) == 0,
        }
        print(f"{h_key} Perturbation Test: Tested {n_tested:,} events. Discrepancies: {discrepancies + tensor_discrepancies}, "
              f"Max diff: {max_num_diff:.2e}")

    audit_results["perturbation_test"] = perturbation_results

    # 9 & 10. INPUT FEATURE AUDIT & TARGET LEAKAGE
    print("\n--- 9 & 10. INPUT FEATURE AUDIT & TARGET LEAKAGE CHECK ---")
    manifest_path = "data/manifests/event_dataset_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_direct = manifest["direct_features_used"]
    manifest_needs_review = manifest["needs_human_review_features_excluded"]
    manifest_not_avail = manifest["not_available_features_excluded"]
    manifest_id_excluded = manifest["identifiers_excluded"]
    manifest_target_excluded = manifest["target_excluded_from_inputs"]

    feature_audit = {
        "direct_features_count": len(DIRECT_FEATURE_COLUMNS),
        "manifest_direct_count": len(manifest_direct),
        "direct_matches_manifest": DIRECT_FEATURE_COLUMNS == manifest_direct,
        "needs_review_count": len(NEEDS_HUMAN_REVIEW_FEATURES),
        "not_available_count": len(NOT_AVAILABLE_FEATURES),
        "target_in_direct_features": "risk" in DIRECT_FEATURE_COLUMNS or "final_risk" in DIRECT_FEATURE_COLUMNS,
        "event_id_in_direct_features": "event_id" in DIRECT_FEATURE_COLUMNS,
        "mission_id_in_direct_features": "mission_id" in DIRECT_FEATURE_COLUMNS,
    }

    # Verify columns in sequences_H*.csv
    for h_key in ["H2", "H3", "H5"]:
        seq_cols = list(seq_data[h_key][0].keys())
        feature_cols_in_seq = [c for c in seq_cols if c not in ("event_id", "step_index", "horizon_days")]
        feature_audit[f"{h_key}_features_equal_direct"] = feature_cols_in_seq == DIRECT_FEATURE_COLUMNS
        feature_audit[f"{h_key}_target_in_seq_cols"] = "risk" in seq_cols or "final_risk" in seq_cols
        feature_audit[f"{h_key}_mission_id_in_seq_cols"] = "mission_id" in seq_cols

    audit_results["feature_audit"] = feature_audit
    print(f"Feature Audit: Direct count = {len(DIRECT_FEATURE_COLUMNS)}, Manifest direct count = {len(manifest_direct)}")
    print(f"Target in direct features: {feature_audit['target_in_direct_features']}")
    print(f"Sequences contain risk/final_risk: {any(feature_audit[f'{h}_target_in_seq_cols'] for h in ['H2', 'H3', 'H5'])}")

    # 11. CROSS-HORIZON & SPLIT EVENT LEAKAGE
    print("\n--- 11. CROSS-HORIZON & SPLIT LEAKAGE AUDIT ---")
    split_manifest_path = "artifacts/splits/master_split_manifest.json"
    split_manifest = SplitManifest.load(split_manifest_path)
    
    train_ids = set(split_manifest.train_event_ids)
    val_ids = set(split_manifest.val_event_ids)
    test_ids = set(split_manifest.test_event_ids)

    split_leakage_results = {
        "master_train_count": len(train_ids),
        "master_val_count": len(val_ids),
        "master_test_count": len(test_ids),
        "master_total_events": len(train_ids) + len(val_ids) + len(test_ids),
        "master_train_val_overlap": len(train_ids.intersection(val_ids)),
        "master_train_test_overlap": len(train_ids.intersection(test_ids)),
        "master_val_test_overlap": len(val_ids.intersection(test_ids)),
    }

    horizon_splits = {}
    for h_key in ["H2", "H3", "H5"]:
        ev_ids = set(seq_by_event[h_key].keys())
        h_train = ev_ids.intersection(train_ids)
        h_val = ev_ids.intersection(val_ids)
        h_test = ev_ids.intersection(test_ids)
        horizon_splits[h_key] = {
            "total_eligible": len(ev_ids),
            "train": len(h_train),
            "val": len(h_val),
            "test": len(h_test),
            "train_val_overlap": len(h_train.intersection(h_val)),
            "train_test_overlap": len(h_train.intersection(h_test)),
            "val_test_overlap": len(h_val.intersection(h_test)),
            "sum_equals_total": (len(h_train) + len(h_val) + len(h_test)) == len(ev_ids),
        }

    # Cross-horizon check: Train(H2) overlap with Val(H3), Test(H3), etc.
    cross_horizon_leakage = {}
    for h1 in ["H2", "H3", "H5"]:
        for h2 in ["H2", "H3", "H5"]:
            if h1 == h2:
                continue
            evs_1 = set(seq_by_event[h1].keys())
            evs_2 = set(seq_by_event[h2].keys())
            t1 = evs_1.intersection(train_ids)
            v2 = evs_2.intersection(val_ids)
            test2 = evs_2.intersection(test_ids)
            
            leak_t1_v2 = len(t1.intersection(v2))
            leak_t1_test2 = len(t1.intersection(test2))
            cross_horizon_leakage[f"Train({h1})_Val({h2})"] = leak_t1_v2
            cross_horizon_leakage[f"Train({h1})_Test({h2})"] = leak_t1_test2

    split_leakage_results["horizon_splits"] = horizon_splits
    split_leakage_results["cross_horizon_leakage"] = cross_horizon_leakage
    audit_results["split_leakage"] = split_leakage_results
    print(f"Master Split Overlaps: Train/Val={split_leakage_results['master_train_val_overlap']}, "
          f"Train/Test={split_leakage_results['master_train_test_overlap']}, Val/Test={split_leakage_results['master_val_test_overlap']}")
    print(f"Cross-Horizon Leakage items checked: {len(cross_horizon_leakage)}, All Zero: {all(v == 0 for v in cross_horizon_leakage.values())}")

    # 12 & 13. SEQUENCE TENSOR CONTRACT & VARIABLE-LENGTH AUDIT
    print("\n--- 12 & 13. SEQUENCE TENSOR CONTRACT & VARIABLE LENGTH AUDIT ---")
    tensor_contract_results = {}
    test_lengths = [1, 2, 5, 10, 23]
    tcn = TCNRiskModel(in_features=34, max_seq_len=23)

    for L in test_lengths:
        # Create synthetic event with L steps
        synthetic_cdms = []
        for i in range(L):
            cdm = {col: str(10.0 + i) for col in DIRECT_FEATURE_COLUMNS}
            # Oldest step has highest time_to_tca
            cdm["time_to_tca"] = str(10.0 - i * 0.2)
            cdm["risk"] = "-15.0"
            cdm["event_id"] = f"syn_{L}"
            synthetic_cdms.append(cdm)

        synthetic_events = {f"syn_{L}": synthetic_cdms}
        X_out, mask_out, y_out, ids_out = tcn.prepare_sequence_tensors(
            synthetic_events, feature_cols=DIRECT_FEATURE_COLUMNS, horizon_cutoff=2.0
        )

        pad_len = 23 - L
        if isinstance(X_out, list):
            feat_matrix = X_out[0]  # [34, 23]
            mask_row = mask_out[0]  # [23]
        else:
            feat_matrix = X_out[0].tolist()
            mask_row = mask_out[0].tolist()

        # Check mask: 0..pad_len-1 is 0.0; pad_len..22 is 1.0
        pad_mask_correct = all(mask_row[i] == 0.0 for i in range(pad_len))
        valid_mask_correct = all(mask_row[i] == 1.0 for i in range(pad_len, 23))

        # Check feature matrix padding: indices 0..pad_len-1 must be 0.0
        pad_feats_zero = all(
            feat_matrix[f_idx][t] == 0.0
            for f_idx in range(34)
            for t in range(pad_len)
        )

        # Check newest observation location: index 22 must correspond to synthetic_cdms[-1] (value 10.0 + L - 1)
        newest_val_expected = 10.0 + L - 1
        newest_val_actual = feat_matrix[1][22]  # c_object_type or miss_distance etc.
        # Check time_to_tca at index 22: synthetic_cdms[-1]["time_to_tca"]
        newest_tca_expected = float(synthetic_cdms[-1]["time_to_tca"])
        newest_tca_actual = feat_matrix[0][22]

        newest_matches = (
            abs(newest_val_actual - newest_val_expected) < 1e-6 and
            abs(newest_tca_actual - newest_tca_expected) < 1e-6
        )

        tensor_contract_results[f"len_{L}"] = {
            "seq_len": L,
            "pad_len": pad_len,
            "pad_mask_correct": pad_mask_correct,
            "valid_mask_correct": valid_mask_correct,
            "pad_features_strictly_zero": pad_feats_zero,
            "newest_step_at_index_22": newest_matches,
            "target_aligned": y_out[0] == -15.0,
        }

    audit_results["tensor_contract"] = tensor_contract_results
    print(f"Tensor Contract for L in [1, 2, 5, 10, 23]: "
          f"All correct: {all(v['pad_mask_correct'] and v['valid_mask_correct'] and v['pad_features_strictly_zero'] and v['newest_step_at_index_22'] for v in tensor_contract_results.values())}")

    # 14. MISSING-VALUE AUDIT
    print("\n--- 14. AUDITING MISSING VALUES ACROSS HORIZONS ---")
    missingness_results = {}
    for h_key in ["H2", "H3", "H5"]:
        rows = seq_data[h_key]
        total_rows = len(rows)
        feature_missing = defaultdict(int)
        for r in rows:
            for col in DIRECT_FEATURE_COLUMNS:
                val = r.get(col, "")
                if not val or val.lower() in ("nan", "null", "none", ""):
                    feature_missing[col] += 1

        missing_summary = [
            {
                "feature": col,
                "missing_count": feature_missing[col],
                "missing_pct": round(feature_missing[col] / total_rows * 100, 4),
            }
            for col in DIRECT_FEATURE_COLUMNS
        ]
        missing_summary.sort(key=lambda x: x["missing_count"], reverse=True)
        missingness_results[h_key] = {
            "total_cdms": total_rows,
            "missing_by_feature": missing_summary,
            "top_3_missing": missing_summary[:3],
        }
        print(f"{h_key} Top Missing: {missing_summary[0]['feature']} ({missing_summary[0]['missing_pct']}%), "
              f"{missing_summary[1]['feature']} ({missing_summary[1]['missing_pct']}%), "
              f"{missing_summary[2]['feature']} ({missing_summary[2]['missing_pct']}%)")

    audit_results["missingness"] = missingness_results

    # 15. DUPLICATE & INTEGRITY AUDIT
    print("\n--- 15. DUPLICATE & INTEGRITY AUDIT ---")
    integrity_results = {}
    for h_key in ["H2", "H3", "H5"]:
        rows = seq_data[h_key]
        events_dict = seq_by_event[h_key]
        
        seen_pairs = set()
        duplicate_pairs = 0
        non_contiguous_indices = 0
        missing_step_indices = 0

        for r in rows:
            pair = (r["event_id"], r["step_index"])
            if pair in seen_pairs:
                duplicate_pairs += 1
            seen_pairs.add(pair)

        for ev_id, steps in events_dict.items():
            expected_indices = list(range(len(steps)))
            actual_indices = [int(s["step_index"]) for s in steps]
            if actual_indices != expected_indices:
                non_contiguous_indices += 1

        # Check events missing from expected
        manifest_expected = manifest["horizon_summary"][h_key]["eligible_events"]
        actual_events_count = len(events_dict)

        integrity_results[h_key] = {
            "total_rows": len(rows),
            "unique_events": actual_events_count,
            "expected_events": manifest_expected,
            "duplicate_event_step_pairs": duplicate_pairs,
            "non_contiguous_indices_events": non_contiguous_indices,
            "is_integral": duplicate_pairs == 0 and non_contiguous_indices == 0 and actual_events_count == manifest_expected,
        }
        print(f"{h_key} Integrity: Duplicate pairs = {duplicate_pairs}, Non-contiguous = {non_contiguous_indices}, "
              f"Events count match = {actual_events_count == manifest_expected}")

    audit_results["integrity"] = integrity_results

    # 16. RAW -> SEQUENCE TRACEABILITY AUDIT (10+ events per horizon)
    print("\n--- 16. RAW -> SEQUENCE TRACEABILITY AUDIT ---")
    traceability_results = {}
    for h_key, cutoff in [("H2", 2.0), ("H3", 3.0), ("H5", 5.0)]:
        events_dict = seq_by_event[h_key]
        
        # Pick 10 diverse representative events: short, medium, long, high risk, low risk
        sorted_by_len = sorted(events_dict.items(), key=lambda item: len(item[1]))
        sample_ids = [
            sorted_by_len[0][0],                    # min len
            sorted_by_len[1][0],                    # short
            sorted_by_len[len(sorted_by_len)//4][0], # p25 len
            sorted_by_len[len(sorted_by_len)//2][0], # median len
            sorted_by_len[3*len(sorted_by_len)//4][0], # p75 len
            sorted_by_len[-1][0],                   # max len
            sorted_by_len[-2][0],                   # second max
            "0",                                    # first event
            "100",                                  # arbitrary id
            "1000",                                 # arbitrary id
            "5000",                                 # arbitrary id
            "10000",                                # arbitrary id
        ]
        # Filter to those actually present in this horizon
        sample_ids = [eid for eid in sample_ids if eid in events_dict][:12]

        event_traces = []
        all_traces_exact = True

        for ev_id in sample_ids:
            raw_cdms = raw_rows_by_event[ev_id]
            seq_cdms = events_dict[ev_id]
            ev_row = events_by_id[h_key][ev_id]

            # Filter raw CDMs by cutoff
            qualifying_raw = [c for c in raw_cdms if float(c["time_to_tca"]) >= cutoff]
            final_raw = min(raw_cdms, key=lambda r: float(r["time_to_tca"]))

            # Verify length
            len_match = len(seq_cdms) == len(qualifying_raw)
            # Verify target
            target_match = abs(float(ev_row["final_risk"]) - float(final_raw["risk"])) < 1e-12

            # Verify feature-by-feature across every step
            feat_match = True
            for step_i, (s_row, r_row) in enumerate(zip(seq_cdms, qualifying_raw)):
                for col in DIRECT_FEATURE_COLUMNS:
                    s_val = s_row[col]
                    r_val = r_row[col]
                    if s_val != r_val:
                        feat_match = False
                        break

            is_exact = len_match and target_match and feat_match
            if not is_exact:
                all_traces_exact = False

            event_traces.append({
                "event_id": ev_id,
                "raw_total_cdms": len(raw_cdms),
                "qualifying_cdms": len(qualifying_raw),
                "sequence_rows": len(seq_cdms),
                "raw_final_risk": float(final_raw["risk"]),
                "dataset_final_risk": float(ev_row["final_risk"]),
                "earliest_raw_tca": float(qualifying_raw[0]["time_to_tca"]),
                "earliest_seq_tca": float(seq_cdms[0]["time_to_tca"]),
                "anchor_raw_tca": float(qualifying_raw[-1]["time_to_tca"]),
                "anchor_seq_tca": float(seq_cdms[-1]["time_to_tca"]),
                "is_exact_match": is_exact,
            })

        traceability_results[h_key] = {
            "events_traced_count": len(event_traces),
            "all_traces_exact": all_traces_exact,
            "traces": event_traces,
        }
        print(f"{h_key} Traceability: Traced {len(event_traces)} events. All exact: {all_traces_exact}")

    audit_results["traceability"] = traceability_results

    # 17. VERIFY SHA-256 AFTER AUDIT
    print("\n--- VERIFYING SHA-256 CHECKSUMS AFTER AUDIT ---")
    checksums_final = {}
    checksums_unchanged = True
    for key, fpath in files_to_audit.items():
        h = compute_file_sha256(fpath)
        size = os.path.getsize(fpath)
        checksums_final[key] = {"path": fpath, "sha256": h, "size_bytes": size}
        if h != checksums_initial[key]["sha256"]:
            checksums_unchanged = False
            print(f"CRITICAL ERROR: Checksum changed for {fpath}!")

    audit_results["checksums_final"] = checksums_final
    audit_results["checksums_unchanged"] = checksums_unchanged
    print(f"All 7 files checksums verified unchanged: {checksums_unchanged}")

    return audit_results


if __name__ == "__main__":
    results = run_full_audit()
    with open("scratch/audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nFull audit completed and saved to scratch/audit_results.json")
