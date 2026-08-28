"""
Inspect chronological time boundaries and dates across partitions for the 60/15/10/15 split.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def inspect_chronological_boundaries():
    raw_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    df_raw = pd.read_csv(raw_path)
    df_raw['event_id_str'] = df_raw['event_id'].astype(str)

    master_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(master_path, "r") as f:
        master_manifest = json.load(f)

    hist_test_ids = set(master_manifest["test_event_ids"])
    hist_train_ids = master_manifest["train_event_ids"]
    hist_val_ids = master_manifest["val_event_ids"]
    
    eligible_ordered = hist_train_ids + hist_val_ids
    N = len(eligible_ordered)
    
    # 60/15/10/15 split indices
    r_tr, r_va, r_ca, r_te = 0.60, 0.15, 0.10, 0.15
    n_tr = int(np.floor(N * r_tr))  # 6708
    n_va = int(np.floor(N * r_va))  # 1677
    n_ca = int(np.floor(N * r_ca))  # 1118
    n_te = N - n_tr - n_va - n_ca   # 1677

    p_tr = eligible_ordered[:n_tr]
    p_va = eligible_ordered[n_tr : n_tr + n_va]
    p_ca = eligible_ordered[n_tr + n_va : n_tr + n_va + n_ca]
    p_te = eligible_ordered[n_tr + n_va + n_ca :]

    partitions = [
        ("Phase5_Train", p_tr, 0.60),
        ("Phase5_Validation", p_va, 0.15),
        ("Phase5_Calibration", p_ca, 0.10),
        ("Phase5_InternalTest", p_te, 0.15),
    ]

    # Map each event_id to its chronological order index in raw data
    event_order_map = {ev: i for i, ev in enumerate(eligible_ordered)}

    # Check time ranges in df_raw for each partition
    print(f"Total eligible events: {N}")
    for name, p_ids, ratio in partitions:
        df_p = df_raw[df_raw['event_id_str'].isin(set(p_ids))]
        min_idx = min(event_order_map[ev] for ev in p_ids)
        max_idx = max(event_order_map[ev] for ev in p_ids)
        min_ttca = df_p['time_to_tca'].min()
        max_ttca = df_p['time_to_tca'].max()
        
        # Check OD span / time features
        min_t_lastob_start = df_p['t_time_lastob_start'].min()
        max_t_lastob_start = df_p['t_time_lastob_start'].max()
        
        print(f"\nPartition: {name} (N={len(p_ids)}, target ratio={ratio:.2f})")
        print(f"  Chronological Event Index Range: [{min_idx}, {max_idx}]")
        print(f"  Event ID Sample (first 3, last 3): {p_ids[:3]} ... {p_ids[-3:]}")
        print(f"  Raw Rows: {len(df_p)}")
        print(f"  time_to_tca range: [{min_ttca:.4f}, {max_ttca:.4f}] days")
        print(f"  t_time_lastob_start range: [{min_t_lastob_start:.1f}, {max_t_lastob_start:.1f}]")

    # Historical locked test set
    df_locked = df_raw[df_raw['event_id_str'].isin(hist_test_ids)]
    locked_order_indices = [i for i, ev in enumerate(master_manifest["train_event_ids"] + master_manifest["val_event_ids"] + master_manifest["test_event_ids"]) if ev in hist_test_ids]
    print(f"\nHistorical Locked Test Set (N={len(hist_test_ids)})")
    print(f"  Chronological Global Index Range: [{min(locked_order_indices)}, {max(locked_order_indices)}]")
    print(f"  Raw Rows: {len(df_locked)}")

if __name__ == "__main__":
    inspect_chronological_boundaries()
