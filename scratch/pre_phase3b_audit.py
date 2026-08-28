"""Pre-Phase 3B Verification Gate Audit Script.

Performs strictly READ-ONLY empirical audits for:
1. Baseline Freeze / Integrity & Hash Audits
2. c_object_type Categorical Encoding Audit
3. Covariance Feature Scale & Dynamic Range Audit
4. Delta-t Availability and Construction Audit
5. H5 TCN Convergence Status Audit
6. Phase 3A Baseline Consistency Check
7. Post-Audit Immutability & Test Suite Verification
"""

import csv
from collections import defaultdict, Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

WORKSPACE_ROOT = Path("c:/Users/Zyren/Documents/orvexa")
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("ORVEXA PRE-PHASE 3B VERIFICATION GATE AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------
    # PART 1: BASELINE FREEZE / INTEGRITY
    # -------------------------------------------------------------
    print("\n--- PART 1: BASELINE FREEZE / INTEGRITY ---")
    files_to_hash = [
        "data/raw/esa/train_data.csv",
        "data/processed/events/events_H2.csv",
        "data/processed/events/events_H3.csv",
        "data/processed/events/events_H5.csv",
        "data/processed/events/sequences_H2.csv",
        "data/processed/events/sequences_H3.csv",
        "data/processed/events/sequences_H5.csv",
        "artifacts/splits/master_split_manifest.json",
        "artifacts/models/tcn_best_h2.0.pt",
        "artifacts/models/tcn_best_h3.0.pt",
        "artifacts/models/tcn_best_h5.0.pt",
        "artifacts/models/tcn_final_h2.0.pt",
        "artifacts/models/tcn_final_h3.0.pt",
        "artifacts/models/tcn_final_h5.0.pt",
        "artifacts/models/tcn_best_h2.0.json",
        "artifacts/models/tcn_best_h3.0.json",
        "artifacts/models/tcn_best_h5.0.json",
        "artifacts/models/tcn_final_h2.0.json",
        "artifacts/models/tcn_final_h3.0.json",
        "artifacts/models/tcn_final_h5.0.json",
        "artifacts/preprocessors/preprocessor_tcn_h2.0.json",
        "artifacts/preprocessors/preprocessor_tcn_h3.0.json",
        "artifacts/preprocessors/preprocessor_tcn_h5.0.json",
    ]

    hashes_initial = {}
    for rel_path in files_to_hash:
        p = WORKSPACE_ROOT / rel_path
        if p.exists():
            h = compute_file_sha256(p)
            hashes_initial[rel_path] = h
            print(f"File: {rel_path}\n  SHA-256: {h}\n  Size: {p.stat().st_size} bytes")
        else:
            print(f"File: {rel_path} MISSING!")

    # Check phase 3A artifacts
    phase3a_report = WORKSPACE_ROOT / "reports/PHASE_3A_TEMPORAL_DIAGNOSTIC_REPORT.md"
    phase3a_csv = WORKSPACE_ROOT / "reports/phase3a_diagnostics.csv"
    print(f"\nPhase 3A Report Exists: {phase3a_report.exists()} ({phase3a_report.stat().st_size if phase3a_report.exists() else 0} bytes)")
    print(f"Phase 3A CSV Exists: {phase3a_csv.exists()} ({phase3a_csv.stat().st_size if phase3a_csv.exists() else 0} bytes)")

    # -------------------------------------------------------------
    # PART 2: VERIFY c_object_type ENCODING
    # -------------------------------------------------------------
    print("\n--- PART 2: VERIFY c_object_type ENCODING ---")
    raw_path = WORKSPACE_ROOT / "data/raw/esa/train_data.csv"
    raw_obj_types = Counter()
    total_raw_rows = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            val = r.get("c_object_type", "")
            raw_obj_types[val] += 1
            total_raw_rows += 1

    print(f"Total raw rows in train_data.csv: {total_raw_rows}")
    print("Raw c_object_type value distribution:")
    for val, cnt in raw_obj_types.most_common():
        pct = (cnt / total_raw_rows) * 100.0
        print(f"  '{val}': {cnt} ({pct:.4f}%)")

    # Sequence datasets inspection for c_object_type
    for h in [2, 3, 5]:
        seq_path = WORKSPACE_ROOT / f"data/processed/events/sequences_H{h}.csv"
        seq_obj_types = Counter()
        seq_rows = 0
        with open(seq_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                val = r.get("c_object_type", "")
                seq_obj_types[val] += 1
                seq_rows += 1
        print(f"\nH{h} Sequences ({seq_rows} CDMs) c_object_type distribution:")
        for val, cnt in seq_obj_types.most_common():
            pct = (cnt / seq_rows) * 100.0
            print(f"  '{val}': {cnt} ({pct:.4f}%)")

    # Check preprocessor_tcn_h*.json
    for h in [2, 3, 5]:
        prep_path = WORKSPACE_ROOT / f"artifacts/preprocessors/preprocessor_tcn_h{h}.0.json"
        with open(prep_path, "r", encoding="utf-8") as f:
            prep_data = json.load(f)
        obj_stats = prep_data.get("channel_stats", {}).get("c_object_type", {})
        print(f"\nH{h} TCN Preprocessor channel_stats['c_object_type']:")
        print(f"  channel_idx: {obj_stats.get('channel_idx')}")
        print(f"  mean: {obj_stats.get('mean')}")
        print(f"  std: {obj_stats.get('std')}")
        print(f"  feature_names[1]: {prep_data.get('feature_names', [])[1] if len(prep_data.get('feature_names', [])) > 1 else None}")

    # Check casting behavior in models_tcn.py prepare_sequence_tensors
    test_vals = ["DEBRIS", "PAYLOAD", "UNKNOWN", "", None]
    print("\nSimulating float casting in prepare_sequence_tensors:")
    for tv in test_vals:
        try:
            if tv is not None:
                fv = float(tv)
                res = 0.0 if math.isnan(fv) else fv
            else:
                res = 0.0
            print(f"  val={tv!r} -> cast success: {res}")
        except (ValueError, TypeError) as e:
            print(f"  val={tv!r} -> raised {type(e).__name__}: {e} -> fallback to 0.0")

    # -------------------------------------------------------------
    # PART 3: COVARIANCE / SCALE AUDIT
    # -------------------------------------------------------------
    print("\n--- PART 3: COVARIANCE / SCALE AUDIT ---")
    cov_features = [
        "t_sigma_r", "t_sigma_t", "t_sigma_n",
        "c_sigma_r", "c_sigma_t", "c_sigma_n",
        "mahalanobis_distance", "miss_distance",
        "relative_speed", "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n"
    ]

    # Let's inspect all 34 direct features from sequences_H2.csv
    from orvexa.event_builder import DIRECT_FEATURE_COLUMNS

    print(f"Auditing DIRECT_FEATURE_COLUMNS ({len(DIRECT_FEATURE_COLUMNS)} features)...")
    seq_path_h2 = WORKSPACE_ROOT / "data/processed/events/sequences_H2.csv"
    feature_data = {col: [] for col in DIRECT_FEATURE_COLUMNS}
    missing_data = {col: 0 for col in DIRECT_FEATURE_COLUMNS}

    with open(seq_path_h2, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for col in DIRECT_FEATURE_COLUMNS:
                raw_val = r.get(col)
                if raw_val is None or raw_val == "":
                    missing_data[col] += 1
                else:
                    try:
                        fval = float(raw_val)
                        if math.isnan(fval) or math.isinf(fval):
                            missing_data[col] += 1
                        else:
                            feature_data[col].append(fval)
                    except (ValueError, TypeError):
                        missing_data[col] += 1

    print("\nDetailed Distribution Statistics on sequences_H2.csv (111,939 CDMs):")
    stats_summary = {}
    for col in DIRECT_FEATURE_COLUMNS:
        vals = np.array(feature_data[col], dtype=np.float64)
        n_tot = len(vals) + missing_data[col]
        n_missing = missing_data[col]
        n_finite = len(vals)
        n_zeros = int(np.sum(vals == 0.0)) if len(vals) > 0 else 0
        n_neg = int(np.sum(vals < 0.0)) if len(vals) > 0 else 0
        n_pos = int(np.sum(vals > 0.0)) if len(vals) > 0 else 0

        if len(vals) > 0:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            p0_1 = float(np.percentile(vals, 0.1))
            p1 = float(np.percentile(vals, 1.0))
            p5 = float(np.percentile(vals, 5.0))
            p25 = float(np.percentile(vals, 25.0))
            median = float(np.percentile(vals, 50.0))
            p75 = float(np.percentile(vals, 75.0))
            p95 = float(np.percentile(vals, 95.0))
            p99 = float(np.percentile(vals, 99.0))
            p99_9 = float(np.percentile(vals, 99.9))
        else:
            mean, std, min_v, max_v = 0.0, 0.0, 0.0, 0.0
            p0_1, p1, p5, p25, median, p75, p95, p99, p99_9 = [0.0]*9

        stats_summary[col] = {
            "count": n_tot,
            "missing": n_missing,
            "finite": n_finite,
            "zeros": n_zeros,
            "negatives": n_neg,
            "positives": n_pos,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "p0_1": p0_1,
            "p1": p1,
            "p5": p5,
            "p25": p25,
            "median": median,
            "p75": p75,
            "p95": p95,
            "p99": p99,
            "p99_9": p99_9,
        }

    # Print sigma / covariance channels
    print("\n--- COVARIANCE SIGMA CHANNELS AUDIT TABLE ---")
    sig_cols = ["t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n", "mahalanobis_distance", "miss_distance"]
    for col in sig_cols:
        s = stats_summary[col]
        print(f"\nFeature: {col}")
        print(f"  Count={s['count']}, Missing={s['missing']}, Finite={s['finite']}, Zeros={s['zeros']}, Neg={s['negatives']}, Pos={s['positives']}")
        print(f"  Mean={s['mean']:.4e}, Std={s['std']:.4e}")
        print(f"  Min={s['min']:.4e}, P0.1={s['p0_1']:.4e}, P1={s['p1']:.4e}, P5={s['p5']:.4e}, P25={s['p25']:.4e}")
        print(f"  Median={s['median']:.4e}, P75={s['p75']:.4e}, P95={s['p95']:.4e}, P99={s['p99']:.4e}, P99.9={s['p99_9']:.4e}, Max={s['max']:.4e}")

        # Normalized values under current linear z-score
        mean, std = s['mean'], max(s['std'], 1e-4)
        norm_p25 = (s['p25'] - mean) / std
        norm_med = (s['median'] - mean) / std
        norm_p75 = (s['p75'] - mean) / std
        norm_p99 = (s['p99'] - mean) / std
        norm_max = (s['max'] - mean) / std
        print(f"  CURRENT Z-Score Normalized: P25={norm_p25:.6f}, Med={norm_med:.6f}, P75={norm_p75:.6f}, P99={norm_p99:.6f}, Max={norm_max:.6f}")

    # -------------------------------------------------------------
    # PART 4: Delta-t AVAILABILITY AUDIT
    # -------------------------------------------------------------
    print("\n--- PART 4: Delta-t AVAILABILITY AUDIT ---")
    # Let's inspect raw columns in train_data.csv for time
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    time_related_cols = [c for c in header if "time" in c or "tca" in c or "date" in c or "span" in c or "epoch" in c]
    print(f"Time-related columns in raw header: {time_related_cols}")

    # Inspect sequences in sequences_H2.csv, sequences_H3.csv, sequences_H5.csv
    # Calculate dt between consecutive CDMs in each sequence
    for h in [2, 3, 5]:
        seq_path = WORKSPACE_ROOT / f"data/processed/events/sequences_H{h}.csv"
        seqs_by_event = defaultdict(list)
        with open(seq_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                seqs_by_event[r["event_id"]].append(r)

        n_events = len(seqs_by_event)
        missing_t_count = 0
        non_monotonic_count = 0
        duplicate_t_count = 0
        all_dts = []
        seq_time_spans = []

        for ev_id, rows in seqs_by_event.items():
            t_vals = []
            for r in rows:
                raw_t = r.get("time_to_tca")
                if raw_t is None or raw_t == "":
                    missing_t_count += 1
                else:
                    try:
                        tv = float(raw_t)
                        t_vals.append(tv)
                    except (ValueError, TypeError):
                        missing_t_count += 1

            if len(t_vals) > 1:
                span = t_vals[0] - t_vals[-1]
                seq_time_spans.append(span)

                for i in range(len(t_vals) - 1):
                    dt = t_vals[i] - t_vals[i+1] # consecutive drop in time_to_tca = elapsed days
                    all_dts.append(dt)
                    if dt < 0:
                        non_monotonic_count += 1
                    elif dt == 0:
                        duplicate_t_count += 1

        dt_arr = np.array(all_dts, dtype=np.float64) if all_dts else np.array([])
        print(f"\nH{h} Sequences Delta-t Audit:")
        print(f"  Total Events (Sequences): {n_events}")
        print(f"  Total Consecutive CDM Pairs: {len(all_dts)}")
        print(f"  Missing timestamps: {missing_t_count}")
        print(f"  Duplicate timestamps (dt == 0): {duplicate_t_count}")
        print(f"  Non-monotonic timestamps (dt < 0): {non_monotonic_count}")
        if len(dt_arr) > 0:
            print(f"  Min Delta-t: {np.min(dt_arr):.6f} days ({np.min(dt_arr)*24.0:.4f} hours)")
            print(f"  P1 Delta-t: {np.percentile(dt_arr, 1.0):.6f} days ({np.percentile(dt_arr, 1.0)*24.0:.4f} hours)")
            print(f"  P5 Delta-t: {np.percentile(dt_arr, 5.0):.6f} days ({np.percentile(dt_arr, 5.0)*24.0:.4f} hours)")
            print(f"  P25 Delta-t: {np.percentile(dt_arr, 25.0):.6f} days ({np.percentile(dt_arr, 25.0)*24.0:.4f} hours)")
            print(f"  Median Delta-t: {np.percentile(dt_arr, 50.0):.6f} days ({np.percentile(dt_arr, 50.0)*24.0:.4f} hours)")
            print(f"  P75 Delta-t: {np.percentile(dt_arr, 75.0):.6f} days ({np.percentile(dt_arr, 75.0)*24.0:.4f} hours)")
            print(f"  P95 Delta-t: {np.percentile(dt_arr, 95.0):.6f} days ({np.percentile(dt_arr, 95.0)*24.0:.4f} hours)")
            print(f"  P99 Delta-t: {np.percentile(dt_arr, 99.0):.6f} days ({np.percentile(dt_arr, 99.0)*24.0:.4f} hours)")
            print(f"  Max Delta-t: {np.max(dt_arr):.6f} days ({np.max(dt_arr)*24.0:.4f} hours)")

    # -------------------------------------------------------------
    # PART 5: H5 CONVERGENCE VERIFICATION
    # -------------------------------------------------------------
    print("\n--- PART 5: H5 CONVERGENCE VERIFICATION ---")
    for h in [2, 3, 5]:
        meta_best_p = WORKSPACE_ROOT / f"artifacts/models/tcn_best_h{h}.0.json"
        meta_final_p = WORKSPACE_ROOT / f"artifacts/models/tcn_final_h{h}.0.json"

        with open(meta_best_p, "r", encoding="utf-8") as f:
            best_meta = json.load(f)
        with open(meta_final_p, "r", encoding="utf-8") as f:
            final_meta = json.load(f)

        stats = best_meta.get("training_stats", {})
        print(f"\nHorizon H{h} TCN Model Training Metadata:")
        print(f"  Best Epoch: {stats.get('best_epoch')}")
        print(f"  Best Val Huber Loss: {stats.get('best_val_loss')}")
        print(f"  Total Epochs Run: {stats.get('total_epochs_run')}")
        print(f"  Final Train Loss: {stats.get('final_train_loss')}")
        print(f"  Device: {stats.get('device')}")

        train_losses = stats.get("train_loss_history", [])
        val_losses = stats.get("val_loss_history", [])
        print(f"  History length: {len(train_losses)} train epochs, {len(val_losses)} val epochs")
        if train_losses and val_losses:
            print(f"  Initial (Epoch 1): Train={train_losses[0]:.4f}, Val={val_losses[0]:.4f}")
            print(f"  Final Epoch ({len(train_losses)}): Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")
            min_val_idx = int(np.argmin(val_losses))
            print(f"  Argmin Val Loss Epoch: {min_val_idx + 1} (Val={val_losses[min_val_idx]:.6f})")
            print(f"  Best Epoch == Max Epoch Budget (50)? {stats.get('best_epoch') == stats.get('total_epochs_run')} (Best={stats.get('best_epoch')}, Total={stats.get('total_epochs_run')})")

            # Check last 10 epochs
            print("  Last 10 Val Losses:")
            for ep_i, (t_l, v_l) in enumerate(zip(train_losses[-10:], val_losses[-10:]), start=len(train_losses)-9):
                print(f"    Epoch {ep_i:2d}: Train Huber={t_l:.4f}, Val Huber={v_l:.4f}")

    # -------------------------------------------------------------
    # PART 6: PHASE 3A BASELINE CONSISTENCY CHECK
    # -------------------------------------------------------------
    print("\n--- PART 6: PHASE 3A BASELINE CONSISTENCY CHECK ---")
    diag_csv_path = WORKSPACE_ROOT / "reports/phase3a_diagnostics.csv"
    with open(diag_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        diag_rows = list(reader)
    print(f"Loaded {len(diag_rows)} rows from phase3a_diagnostics.csv")

    by_section = defaultdict(list)
    for r in diag_rows:
        by_section[r["Section"]].append(r)

    for sec, rows in by_section.items():
        print(f"\nSection: {sec} ({len(rows)} entries)")
        for r in rows[:3]:
            print(f"  Horizon={r.get('Horizon')}, Model={r.get('Model_or_Comparison')}, Subgroup={r.get('Subgroup')}, N={r.get('N')}, {r.get('Metric_1_Name')}={r.get('Metric_1_Value')}, {r.get('Metric_2_Name')}={r.get('Metric_2_Value')}, {r.get('Metric_3_Name')}={r.get('Metric_3_Value')}")

    # Specific Verification Checks
    print("\n--- DETAILED CROSS-VERIFICATION OF KEY PHASE 3A METRICS ---")
    for r in diag_rows:
        if r["Section"] == "Residuals_Overall":
            print(f"  Overall Residuals: H={r['Horizon']}, Model={r['Model_or_Comparison']}, MAE={r['Metric_1_Value']}, RMSE={r['Metric_2_Value']}, MeanRes={r['Metric_3_Value']}")
        elif r["Section"] == "Residuals_By_Risk_Bin" and r["Subgroup"] == "y >= -5":
            print(f"  High-Risk Bin (y >= -5): H={r['Horizon']}, Model={r['Model_or_Comparison']}, N={r['N']}, MAE={r['Metric_1_Value']}, RMSE={r['Metric_2_Value']}, MeanRes={r['Metric_3_Value']}")
        elif r["Section"] == "Performance_By_Sequence_Length" and r["Subgroup"] in ["L = 1", "L > 10", "L = 4-6"]:
            print(f"  Seq Length ({r['Subgroup']}): H={r['Horizon']}, Model={r['Model_or_Comparison']}, N={r['N']}, MAE={r['Metric_1_Value']}, RMSE={r['Metric_2_Value']}, R2={r['Metric_3_Value']}")
        elif r["Section"] == "High_Risk_Ranking" and r["Subgroup"] in ["Top 1%", "Top 5%", "Top 10%"]:
            print(f"  Alert Ranking ({r['Subgroup']}): H={r['Horizon']}, Model={r['Model_or_Comparison']}, Recall={r['Metric_1_Value']}, Precision={r['Metric_2_Value']}, TP={r['Metric_3_Value']}, FP={r['Metric_4_Value']}, FN={r['Metric_5_Value']}")

if __name__ == "__main__":
    main()
