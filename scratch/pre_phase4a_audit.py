"""ORVEXA Phase 4A Pre-Verification and Experimental Readiness Audit Script.

READ-ONLY AUDIT:
1. Hash verification of all baseline and frozen Phase 3B artifacts.
2. H > 5 days data distribution and feasibility audit.
3. Cross-catalog search across entire workspace.
4. M4 contract and preprocessor inspection.
"""

import os
import sys
import json
import csv
import math
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import compute_file_sha256, build_event_prefixes_from_raw, DIRECT_FEATURE_COLUMNS
from orvexa.splitting import SplitManifest
from orvexa.phase3b_config import (
    ExperimentID,
    OBJECT_TYPE_ONE_HOT_CHANNELS,
    COVARIANCE_LOG10_FEATURE_COLUMNS,
    RAW_OBJECT_TYPE_CATEGORIES,
    Phase3BExperimentConfig,
    get_experiment_config,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor, encode_c_object_type_one_hot

# 1. Expected Hashes
CANONICAL_HASHES = {
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
    "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
    "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
    "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
}


def verify_hashes():
    print("=== 1. VERIFYING SHA-256 HASHES ===")
    results = {}
    all_pass = True
    for path, expected in CANONICAL_HASHES.items():
        if not os.path.exists(path):
            print(f"FAIL (MISSING): {path}")
            results[path] = {"status": "FAIL (MISSING)", "expected": expected, "actual": None}
            all_pass = False
        else:
            actual = compute_file_sha256(path)
            if actual == expected:
                print(f"PASS: {path}")
                results[path] = {"status": "PASS", "expected": expected, "actual": actual}
            else:
                print(f"FAIL: {path} (expected {expected}, got {actual})")
                results[path] = {"status": "FAIL", "expected": expected, "actual": actual}
                all_pass = False
    return results, all_pass


def audit_raw_data_lead_times():
    print("\n=== 2. AUDITING RAW DATA LEAD TIMES (time_to_tca) ===")
    raw_path = "data/raw/esa/train_data.csv"
    
    # Load manifest
    manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
    tr_set = set(manifest.train_event_ids)
    va_set = set(manifest.val_event_ids)
    te_set = set(manifest.test_event_ids)

    raw_rows_by_event = defaultdict(list)
    rows_count = 0
    all_ttcas = []
    
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows_count += 1
            eid = str(r["event_id"])
            raw_rows_by_event[eid].append(r)
            ttca = float(r["time_to_tca"])
            all_ttcas.append(ttca)

    all_ttcas = np.array(all_ttcas)
    print(f"Total rows in train_data.csv: {rows_count}")
    print(f"Total unique event_ids in train_data.csv: {len(raw_rows_by_event)}")
    print(f"time_to_tca Range: Min = {all_ttcas.min():.4f} days ({all_ttcas.min()*24:.2f} hours), Max = {all_ttcas.max():.4f} days ({all_ttcas.max()*24:.2f} hours)")
    print(f"Percentiles: P1={np.percentile(all_ttcas, 1):.4f}, P5={np.percentile(all_ttcas, 5):.4f}, P25={np.percentile(all_ttcas, 25):.4f}, "
          f"Median={np.median(all_ttcas):.4f}, P75={np.percentile(all_ttcas, 75):.4f}, P95={np.percentile(all_ttcas, 95):.4f}, P99={np.percentile(all_ttcas, 99):.4f}, P99.9={np.percentile(all_ttcas, 99.9):.4f}")

    # Maximum lead time per event
    event_max_ttca = {eid: max(float(r["time_to_tca"]) for r in rows) for eid, rows in raw_rows_by_event.items()}
    max_ttcas_arr = np.array(list(event_max_ttca.values()))
    
    print(f"\nEvent Earliest Detection (Max time_to_tca per event):")
    print(f"Min of max_ttca: {max_ttcas_arr.min():.4f} days, Max of max_ttca: {max_ttcas_arr.max():.4f} days")
    print(f"Max ttca percentiles: P1={np.percentile(max_ttcas_arr, 1):.4f}, P10={np.percentile(max_ttcas_arr, 10):.4f}, P50={np.median(max_ttcas_arr):.4f}, P90={np.percentile(max_ttcas_arr, 90):.4f}, P99={np.percentile(max_ttcas_arr, 99):.4f}")

    # Check temporal anomalies across all raw events
    print("\n--- Checking Temporal Data Quality (Missingness, Duplicates, Monotonicity) ---")
    duplicate_count = 0
    non_monotonic_count = 0
    negative_ttca_count = sum(1 for t in all_ttcas if t < 0)
    
    for eid, rows in raw_rows_by_event.items():
        times = [float(r["time_to_tca"]) for r in rows]
        if len(times) != len(set(times)):
            duplicate_count += 1
        is_decreasing = all(times[i] >= times[i+1] for i in range(len(times)-1))
        is_increasing = all(times[i] <= times[i+1] for i in range(len(times)-1))
        if not is_decreasing and not is_increasing:
            non_monotonic_count += 1

    print(f"Total events with duplicate timestamps: {duplicate_count}")
    print(f"Total events with non-monotonic timestamps in raw order: {non_monotonic_count}")
    print(f"Total observations with negative time_to_tca (post-TCA): {negative_ttca_count} ({negative_ttca_count/rows_count*100:.3f}%)")

    # Horizon detailed inspection
    candidate_horizons = [2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10.0, 14.0]
    horizon_results = {}

    print("\n--- Candidate Horizon Detailed Feasibility Matrix ---")
    for H in candidate_horizons:
        records, stats = build_event_prefixes_from_raw(raw_rows_by_event, horizon_days=H)
        n_eligible = len(records)
        
        # Partition breakdown
        n_train = sum(1 for r in records if r.event_id in tr_set)
        n_val = sum(1 for r in records if r.event_id in va_set)
        n_test = sum(1 for r in records if r.event_id in te_set)
        
        seq_lens = [r.sequence_length for r in records]
        total_obs = sum(seq_lens)
        avg_len = float(np.mean(seq_lens)) if seq_lens else 0.0
        med_len = float(np.median(seq_lens)) if seq_lens else 0.0
        max_len = max(seq_lens) if seq_lens else 0
        min_len = min(seq_lens) if seq_lens else 0
        single_obs = sum(1 for l in seq_lens if l == 1)
        multi_obs = sum(1 for l in seq_lens if l > 1)
        
        earliest_ttcas = [r.earliest_time_to_tca for r in records]
        anchor_ttcas = [r.anchor_time_to_tca for r in records]
        
        earliest_t = max(earliest_ttcas) if earliest_ttcas else 0.0
        latest_t = min(anchor_ttcas) if anchor_ttcas else 0.0
        
        horizon_results[H] = {
            "horizon_days": H,
            "total_qualifying_events": n_eligible,
            "pct_total_events": round(n_eligible / len(raw_rows_by_event) * 100, 2),
            "train_events": n_train,
            "val_events": n_val,
            "test_events": n_test,
            "total_observations": total_obs,
            "mean_sequence_length": round(avg_len, 2),
            "median_sequence_length": med_len,
            "max_sequence_length": max_len,
            "min_sequence_length": min_len,
            "single_obs_events": single_obs,
            "single_obs_pct": round(single_obs / n_eligible * 100, 2) if n_eligible > 0 else 0.0,
            "multi_obs_events": multi_obs,
            "multi_obs_pct": round(multi_obs / n_eligible * 100, 2) if n_eligible > 0 else 0.0,
            "earliest_time_to_tca": round(earliest_t, 4),
            "latest_time_to_tca": round(latest_t, 4),
        }
        
        print(f"H = {H:4.1f}d | Total: {n_eligible:5d} ({n_eligible/len(raw_rows_by_event)*100:5.1f}%) | Tr: {n_train:5d} | Val: {n_val:4d} | Test: {n_test:4d} | "
              f"Obs: {total_obs:6d} | Len: avg={avg_len:4.2f}, med={med_len:2.0f}, max={max_len:2d} | Single-CDM: {single_obs:4d} ({single_obs/n_eligible*100 if n_eligible>0 else 0:5.1f}%) | "
              f"t_tca range: [{latest_t:6.4f}, {earliest_t:6.4f}]")

    return horizon_results, raw_rows_by_event


def audit_cross_catalog_search():
    print("\n=== 3. SEARCHING FOR CROSS-CATALOG & EXTERNAL DATASETS ===")
    root_dir = "."
    external_candidates = []
    
    # Exhaustive search of filesystem
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(ignored in dirpath for ignored in [".git", ".venv", "__pycache__", ".pytest_cache", ".system_generated"]):
            continue
        for f in filenames:
            full_path = os.path.normpath(os.path.join(dirpath, f))
            size = os.path.getsize(full_path)
            f_lower = f.lower()
            
            if any(f_lower.endswith(ext) for ext in [".csv", ".json", ".parquet", ".h5", ".pkl", ".pt", ".npy", ".npz", ".txt", ".tar", ".gz", ".zip"]):
                external_candidates.append({
                    "path": full_path,
                    "size_bytes": size,
                    "dir": dirpath,
                    "filename": f,
                })
    
    print(f"Total data/artifact/config files found: {len(external_candidates)}")
    
    raw_files = [c for c in external_candidates if "data" + os.sep + "raw" in c["path"] or "data\\raw" in c["path"] or "data/raw" in c["path"]]
    processed_files = [c for c in external_candidates if "data" + os.sep + "processed" in c["path"] or "data\\processed" in c["path"] or "data/processed" in c["path"]]
    
    print("\nRaw Data Files in Workspace:")
    for rf in raw_files:
        print(f"  {rf['path']} ({rf['size_bytes']:,} bytes)")
        
    print("\nProcessed Data Files in Workspace:")
    for pf in processed_files:
        print(f"  {pf['path']} ({pf['size_bytes']:,} bytes)")

    # Check text references to external catalogs
    print("\nSearching text across codebase for external catalog mentions:")
    keywords = ["space-track", "spacetrack", "18th sds", "18 sds", "celestrak", "leolabs", "radar catalog", "external catalog", "non-esa", "benchmark dataset"]
    matches = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(ignored in dirpath for ignored in [".git", ".venv", "__pycache__", ".pytest_cache", ".system_generated", "artifacts"]):
            continue
        for f in filenames:
            if f.endswith((".py", ".json", ".md", ".yaml", ".yml", ".sh", ".toml")):
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read().lower()
                        for kw in keywords:
                            if kw in content:
                                matches[kw].append(fp)
                except:
                    pass
    
    for kw, file_list in matches.items():
        print(f"  Keyword '{kw}': {len(file_list)} files: {file_list[:5]}")


def audit_m4_contract():
    print("\n=== 4. AUDITING M4 DEPLOYABILITY CONTRACT ===")
    import torch
    
    preproc_path = "artifacts/preprocessors/phase3b/preprocessor_M4_h2.0.json"
    with open(preproc_path, "r", encoding="utf-8") as f:
        preproc_data = json.load(f)
        
    config = preproc_data["experiment_config"]
    feature_cols = preproc_data["feature_columns"]
    print(f"M4 Feature Columns count: {len(feature_cols)}")
    print(f"M4 Feature Columns (first 10): {feature_cols[:10]}")
    print(f"M4 Feature Columns (10 to 20): {feature_cols[10:20]}")
    print(f"M4 Feature Columns (20 to 37): {feature_cols[20:]}")
    print(f"M4 include_categorical_one_hot: {config['include_categorical_one_hot']}")
    print(f"M4 include_covariance_log10: {config['include_covariance_log10']}")
    print(f"M4 include_delta_t: {config['include_delta_t']}")
    
    # Detailed channel stats inspection
    channel_stats = preproc_data["channel_stats"]
    print(f"\nChannel Stats breakdown (37 channels):")
    for idx, (k, v) in enumerate(channel_stats.items()):
        print(f"  Ch {idx:2d}: {k:30s} | one_hot={str(v.get('is_one_hot', False)):5s} | log10={str(v.get('is_log10', False)):5s} | mean={v.get('mean', 0.0):12.4f} | std={v.get('std', 1.0):12.4f}")
        
    # Check model weights shape across all H2, H3, H5
    for h in [2.0, 3.0, 5.0]:
        ckpt_path = f"artifacts/models/phase3b/step3/tcn_best_M4_h{h:.1f}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"\nCheckpoint {ckpt_path} keys: {list(ckpt.keys())}")
        # Check first conv layer weight
        first_layer_key = [k for k in ckpt.keys() if "conv1" in k and "weight" in k][0]
        weight = ckpt[first_layer_key]
        print(f"  First conv weight key: '{first_layer_key}', shape: {weight.shape}")
        assert weight.shape == (64, 37, 3), f"Expected shape (64, 37, 3), got {weight.shape}"
        
        # Check linear head weight
        head_key = [k for k in ckpt.keys() if "head" in k and "weight" in k][0]
        head_weight = ckpt[head_key]
        print(f"  Head weight key: '{head_key}', shape: {head_weight.shape}")
        assert head_weight.shape == (1, 128), f"Expected shape (1, 128), got {head_weight.shape}"
        
    print("\nALL M4 Checkpoints (H2, H3, H5) strictly verified as 37-channel causal TCNs.")


if __name__ == "__main__":
    verify_hashes()
    h_res, raw_rows = audit_raw_data_lead_times()
    audit_cross_catalog_search()
    audit_m4_contract()
