"""
Phase 5 Step 2: Build Phase 5 Split Manifest and Verify All Invariants.
"""
import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()

def build_phase5_split():
    master_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(master_path, "r") as f:
        master_manifest = json.load(f)

    hist_train_ids = master_manifest["train_event_ids"]
    hist_val_ids = master_manifest["val_event_ids"]
    hist_test_ids = master_manifest["test_event_ids"]

    eligible_ordered = hist_train_ids + hist_val_ids
    N = len(eligible_ordered)
    assert N == 11180

    r_tr, r_va, r_ca, r_te = 0.60, 0.15, 0.10, 0.15
    n_tr = int(np.floor(N * r_tr))  # 6708
    n_va = int(np.floor(N * r_va))  # 1677
    n_ca = int(np.floor(N * r_ca))  # 1118
    n_te = N - n_tr - n_va - n_ca   # 1677

    p_tr = eligible_ordered[:n_tr]
    p_va = eligible_ordered[n_tr : n_tr + n_va]
    p_ca = eligible_ordered[n_tr + n_va : n_tr + n_va + n_ca]
    p_te = eligible_ordered[n_tr + n_va + n_ca :]

    # Disjointness checks
    s_tr, s_va, s_ca, s_te = set(p_tr), set(p_va), set(p_ca), set(p_te)
    s_hist_test = set(hist_test_ids)

    assert len(p_tr) == 6708
    assert len(p_va) == 1677
    assert len(p_ca) == 1118
    assert len(p_te) == 1677

    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_ca)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_ca)
    assert s_va.isdisjoint(s_te)
    assert s_ca.isdisjoint(s_te)

    # Quarantine checks
    assert s_tr.isdisjoint(s_hist_test)
    assert s_va.isdisjoint(s_hist_test)
    assert s_ca.isdisjoint(s_hist_test)
    assert s_te.isdisjoint(s_hist_test)

    # Load horizon event and sequence data for horizon-specific coverage
    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    horizon_coverage = {}

    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        sfile = os.path.join(events_dir, f"sequences_{h}.csv")
        edf = pd.read_csv(efile)
        sdf = pd.read_csv(sfile)
        edf['event_id_str'] = edf['event_id'].astype(str)
        sdf['event_id_str'] = sdf['event_id'].astype(str)

        risk_col = 'final_risk' if 'final_risk' in edf.columns else ('risk' if 'risk' in edf.columns else 'target_risk')
        events_set = set(edf['event_id_str'])
        seq_lens = sdf.groupby('event_id_str').size().to_dict()
        risks = edf.set_index('event_id_str')[risk_col].to_dict()

        h_parts = {}
        for pname, p_list in [
            ("train", p_tr),
            ("validation", p_va),
            ("calibration", p_ca),
            ("internal_test", p_te),
        ]:
            qualifying = [ev for ev in p_list if ev in events_set]
            lens = [seq_lens[ev] for ev in qualifying]
            crit = sum(1 for ev in qualifying if risks[ev] >= -5.0)
            obs_cnt = sum(lens)
            h_parts[pname] = {
                "events_count": len(qualifying),
                "observations_count": obs_cnt,
                "critical_events_count": crit,
                "mean_sequence_length": float(np.round(np.mean(lens), 4)) if lens else 0.0,
                "median_sequence_length": float(np.median(lens)) if lens else 0.0,
                "min_sequence_length": int(min(lens)) if lens else 0,
                "max_sequence_length": int(max(lens)) if lens else 0,
            }
        
        # Also compute total eligible for this horizon
        total_qual = sum(h_parts[p]["events_count"] for p in h_parts)
        total_obs = sum(h_parts[p]["observations_count"] for p in h_parts)
        total_crit = sum(h_parts[p]["critical_events_count"] for p in h_parts)
        horizon_coverage[h] = {
            "horizon": h,
            "total_eligible_events": total_qual,
            "total_eligible_observations": total_obs,
            "total_eligible_critical_events": total_crit,
            "partitions": h_parts,
        }

    # Manifest dictionary
    manifest_data = {
        "manifest_type": "PHASE_5_EXPERIMENTAL_SPLIT_MANIFEST",
        "description": "4-way chronological event-level disjoint split for Phase 5 probabilistic modeling, validation, conformal calibration, and internal test.",
        "source_dataset": "data/raw/esa/train_data.csv",
        "historical_master_split_manifest": "artifacts/splits/master_split_manifest.json",
        "historical_locked_test_quarantine": {
            "historical_locked_test_events_count": len(hist_test_ids),
            "historical_locked_test_overlap_phase5_train": len(s_tr & s_hist_test),
            "historical_locked_test_overlap_phase5_validation": len(s_va & s_hist_test),
            "historical_locked_test_overlap_phase5_calibration": len(s_ca & s_hist_test),
            "historical_locked_test_overlap_phase5_internal_test": len(s_te & s_hist_test),
            "status": "QUARANTINED_AND_IMMUTABLE",
        },
        "ratios": {
            "train_ratio": r_tr,
            "validation_ratio": r_va,
            "calibration_ratio": r_ca,
            "internal_test_ratio": r_te,
        },
        "counts": {
            "train": len(p_tr),
            "validation": len(p_va),
            "calibration": len(p_ca),
            "internal_test": len(p_te),
            "total_eligible_events": N,
            "historical_quarantined_test_events": len(hist_test_ids),
            "grand_total_events": N + len(hist_test_ids),
        },
        "chronological_boundaries": {
            "train": {
                "start_event_id": p_tr[0],
                "end_event_id": p_tr[-1],
                "event_index_range": [0, n_tr - 1],
            },
            "validation": {
                "start_event_id": p_va[0],
                "end_event_id": p_va[-1],
                "event_index_range": [n_tr, n_tr + n_va - 1],
            },
            "calibration": {
                "start_event_id": p_ca[0],
                "end_event_id": p_ca[-1],
                "event_index_range": [n_tr + n_va, n_tr + n_va + n_ca - 1],
            },
            "internal_test": {
                "start_event_id": p_te[0],
                "end_event_id": p_te[-1],
                "event_index_range": [n_tr + n_va + n_ca, N - 1],
            },
            "historical_quarantined_test": {
                "start_event_id": hist_test_ids[0],
                "end_event_id": hist_test_ids[-1],
                "event_index_range": [N, N + len(hist_test_ids) - 1],
            },
        },
        "horizon_coverage": horizon_coverage,
        "train_event_ids": p_tr,
        "val_event_ids": p_va,
        "cal_event_ids": p_ca,
        "test_event_ids": p_te,
    }

    # Save to artifacts/splits/phase5/
    out_dir = os.path.join(WORKSPACE, "artifacts", "splits", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    out_artifact = os.path.join(out_dir, "phase5_split_manifest.json")
    with open(out_artifact, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Save report manifest copy to reports/phase5/
    rep_dir = os.path.join(WORKSPACE, "reports", "phase5")
    os.makedirs(rep_dir, exist_ok=True)
    out_report = os.path.join(rep_dir, "step2_split_manifest.json")
    with open(out_report, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Calculate SHA256 hashes
    h_artifact = compute_sha256(out_artifact)
    h_report = compute_sha256(out_report)
    print(f"Artifact path: {out_artifact}")
    print(f"Artifact SHA256: {h_artifact}")
    print(f"Report manifest path: {out_report}")
    print(f"Report manifest SHA256: {h_report}")

    return manifest_data

if __name__ == "__main__":
    build_phase5_split()
