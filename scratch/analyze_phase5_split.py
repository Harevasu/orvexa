"""
Phase 5 Step 2: Temporal Distribution and Partition Analysis Script.
"""
import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()

def analyze_event_pool():
    master_split_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(master_split_path, "r") as f:
        master_manifest = json.load(f)

    hist_train_ids = set(master_manifest["train_event_ids"])
    hist_val_ids = set(master_manifest["val_event_ids"])
    hist_test_ids = set(master_manifest["test_event_ids"])

    eligible_ids = hist_train_ids | hist_val_ids
    print(f"Historical Train: {len(hist_train_ids)}, Val: {len(hist_val_ids)}, Test: {len(hist_test_ids)}")
    print(f"Eligible Phase 5 Pool: {len(eligible_ids)} events")

    # Load raw data
    raw_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    df_raw = pd.read_csv(raw_path)
    df_raw['event_id_str'] = df_raw['event_id'].astype(str)

    # Filter to eligible events
    df_eligible = df_raw[df_raw['event_id_str'].isin(eligible_ids)].copy()
    print(f"Eligible raw rows: {len(df_eligible)}")

    # Check event-level ordering in the original dataset or time stamps
    # Let's inspect time features: time_to_tca, t_time_lastob_start, t_time_lastob_end, c_time_lastob_start, c_time_lastob_end, mission_id
    print("Checking time-related columns:")
    for col in ['t_time_lastob_start', 't_time_lastob_end', 'c_time_lastob_start', 'c_time_lastob_end', 'time_to_tca']:
        if col in df_eligible.columns:
            print(f"  {col}: min={df_eligible[col].min()}, max={df_eligible[col].max()}, nulls={df_eligible[col].isnull().sum()}")

    # Check how historical split was ordered
    # In splitting.py, make_chronological_splits uses the order of event_ids as they appear in the dataset.
    # Let's verify the order of unique event_ids in df_raw:
    raw_ordered_event_ids = []
    seen = set()
    for ev in df_raw['event_id_str']:
        if ev not in seen:
            seen.add(ev)
            raw_ordered_event_ids.append(ev)

    print(f"Total ordered unique event IDs in raw: {len(raw_ordered_event_ids)}")
    # Verify how master_manifest was constructed:
    n_total = len(raw_ordered_event_ids)
    n_tr = int(np.floor(n_total * 0.70))
    n_va = int(np.floor(n_total * 0.15))
    expected_tr = raw_ordered_event_ids[:n_tr]
    expected_va = raw_ordered_event_ids[n_tr:n_tr+n_va]
    expected_te = raw_ordered_event_ids[n_tr+n_va:]

    print(f"Match master train: {expected_tr == master_manifest['train_event_ids']}")
    print(f"Match master val: {expected_va == master_manifest['val_event_ids']}")
    print(f"Match master test: {expected_te == master_manifest['test_event_ids']}")

    # The 11,180 eligible event IDs are exactly raw_ordered_event_ids[:11180] (which is expected_tr + expected_va)
    eligible_ordered = raw_ordered_event_ids[:11180]
    print(f"Eligible ordered event IDs count: {len(eligible_ordered)}")

    # Let's inspect per-event summary: final_risk, min_time_to_tca, max_time_to_tca, n_cdms, t_time_lastob_start min/max
    event_summaries = []
    for ev_id, group in df_eligible.groupby('event_id_str', sort=False):
        # target risk is the risk of the final CDM (or min time_to_tca)
        last_cdm = group.sort_values('time_to_tca').iloc[0]
        first_cdm = group.sort_values('time_to_tca', ascending=False).iloc[0]
        final_risk = last_cdm['risk']
        n_cdms = len(group)
        min_ttca = group['time_to_tca'].min()
        max_ttca = group['time_to_tca'].max()
        earliest_time = group['t_time_lastob_start'].min()
        latest_time = group['t_time_lastob_end'].max()
        event_summaries.append({
            'event_id': ev_id,
            'n_cdms': n_cdms,
            'final_risk': final_risk,
            'is_critical': final_risk >= -5.0,
            'min_ttca': min_ttca,
            'max_ttca': max_ttca,
            'earliest_time': earliest_time,
            'latest_time': latest_time,
            'has_h2': max_ttca >= 2.0,
            'has_h3': max_ttca >= 3.0,
            'has_h5': max_ttca >= 5.0,
            'has_h6': max_ttca >= 6.0,
        })

    df_events = pd.DataFrame(event_summaries)
    print(f"\nTotal events summarized: {len(df_events)}")
    print(f"Critical events in eligible pool: {df_events['is_critical'].sum()} / {len(df_events)} ({df_events['is_critical'].mean()*100:.2f}%)")
    print(f"Events by horizon:")
    print(f"  H2: {df_events['has_h2'].sum()}")
    print(f"  H3: {df_events['has_h3'].sum()}")
    print(f"  H5: {df_events['has_h5'].sum()}")
    print(f"  H6: {df_events['has_h6'].sum()}")

    return df_events, eligible_ordered, df_raw

if __name__ == "__main__":
    analyze_event_pool()
