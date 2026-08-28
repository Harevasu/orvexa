"""
Detailed evaluation of candidate split ratios on eligible Phase 5 event pool.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def evaluate_splits():
    raw_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    df_raw = pd.read_csv(raw_path)
    df_raw['event_id_str'] = df_raw['event_id'].astype(str)

    master_split_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(master_split_path, "r") as f:
        master_manifest = json.load(f)

    hist_test_ids = set(master_manifest["test_event_ids"])

    # Extract ordered unique event IDs
    seen = set()
    ordered_events = []
    for ev in df_raw['event_id_str']:
        if ev not in seen:
            seen.add(ev)
            ordered_events.append(ev)

    eligible_ordered = [ev for ev in ordered_events if ev not in hist_test_ids]
    assert len(eligible_ordered) == 11180, f"Expected 11180 eligible, got {len(eligible_ordered)}"

    # Build per-event lookup
    print("Building event lookup...")
    event_data = {}
    for ev_id, group in df_raw.groupby('event_id_str', sort=False):
        if ev_id in hist_test_ids:
            continue
        last_cdm = group.sort_values('time_to_tca').iloc[0]
        final_risk = last_cdm['risk']
        max_ttca = group['time_to_tca'].max()
        cdm_count = len(group)
        # Count CDMs for each horizon
        cdms_h2 = len(group[group['time_to_tca'] >= 2.0])
        cdms_h3 = len(group[group['time_to_tca'] >= 3.0])
        cdms_h5 = len(group[group['time_to_tca'] >= 5.0])
        cdms_h6 = len(group[group['time_to_tca'] >= 6.0])

        event_data[ev_id] = {
            'event_id': ev_id,
            'final_risk': final_risk,
            'is_critical': final_risk >= -5.0,
            'total_cdms': cdm_count,
            'max_ttca': max_ttca,
            'h2': cdms_h2 > 0,
            'h3': cdms_h3 > 0,
            'h5': cdms_h5 > 0,
            'h6': cdms_h6 > 0,
            'cdms_h2': cdms_h2,
            'cdms_h3': cdms_h3,
            'cdms_h5': cdms_h5,
            'cdms_h6': cdms_h6,
        }

    # Evaluate multiple split ratio configs
    configs = [
        ("70/10/10/10", (0.70, 0.10, 0.10, 0.10)),
        ("60/15/10/15", (0.60, 0.15, 0.10, 0.15)),
        ("65/15/10/10", (0.65, 0.15, 0.10, 0.10)),
        ("60/15/15/10", (0.60, 0.15, 0.15, 0.10)),
        ("64/12/12/12", (0.64, 0.12, 0.12, 0.12)),
    ]

    N = len(eligible_ordered)
    for name, (r_tr, r_va, r_ca, r_te) in configs:
        n_tr = int(np.floor(N * r_tr))
        n_va = int(np.floor(N * r_va))
        n_ca = int(np.floor(N * r_ca))
        n_te = N - n_tr - n_va - n_ca

        p_tr = eligible_ordered[:n_tr]
        p_va = eligible_ordered[n_tr : n_tr + n_va]
        p_ca = eligible_ordered[n_tr + n_va : n_tr + n_va + n_ca]
        p_te = eligible_ordered[n_tr + n_va + n_ca :]

        assert len(p_tr) + len(p_va) + len(p_ca) + len(p_te) == N
        assert set(p_tr).isdisjoint(set(p_va))
        assert set(p_tr).isdisjoint(set(p_ca))
        assert set(p_tr).isdisjoint(set(p_te))
        assert set(p_va).isdisjoint(set(p_ca))
        assert set(p_va).isdisjoint(set(p_te))
        assert set(p_ca).isdisjoint(set(p_te))

        print(f"\n==========================================")
        print(f"Config: {name} (N={N}: tr={len(p_tr)}, va={len(p_va)}, ca={len(p_ca)}, te={len(p_te)})")
        print(f"==========================================")

        partitions = [("Train", p_tr), ("Val", p_va), ("Cal", p_ca), ("Test", p_te)]
        for pname, p_ids in partitions:
            p_crit = sum(event_data[ev]['is_critical'] for ev in p_ids)
            print(f"  {pname}: events={len(p_ids)}, critical={p_crit} ({p_crit/len(p_ids)*100:.2f}%)")
            for h in ['h2', 'h3', 'h5', 'h6']:
                h_evs = [ev for ev in p_ids if event_data[ev][h]]
                h_crit = sum(event_data[ev]['is_critical'] for ev in h_evs)
                h_cdms = [event_data[ev][f'cdms_{h}'] for ev in h_evs]
                mean_len = np.mean(h_cdms) if h_cdms else 0
                med_len = np.median(h_cdms) if h_cdms else 0
                print(f"    {h.upper()}: evs={len(h_evs)}, crit={h_crit}, obs={sum(h_cdms)}, mean_len={mean_len:.2f}, med_len={med_len}")

if __name__ == "__main__":
    evaluate_splits()
