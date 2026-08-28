"""
Fast vectorized split analyzer using existing processed event and sequence tables.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def fast_analyze_splits():
    master_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(master_path, "r") as f:
        master_manifest = json.load(f)

    hist_test_ids = set(master_manifest["test_event_ids"])
    hist_train_ids = master_manifest["train_event_ids"]
    hist_val_ids = master_manifest["val_event_ids"]
    
    eligible_ordered = hist_train_ids + hist_val_ids
    assert len(eligible_ordered) == 11180
    assert set(eligible_ordered).isdisjoint(hist_test_ids)
    print(f"Eligible ordered events: {len(eligible_ordered)}")

    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    horizon_data = {}
    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        sfile = os.path.join(events_dir, f"sequences_{h}.csv")
        edf = pd.read_csv(efile)
        sdf = pd.read_csv(sfile)
        edf['event_id_str'] = edf['event_id'].astype(str)
        sdf['event_id_str'] = sdf['event_id'].astype(str)
        
        risk_col = 'final_risk' if 'final_risk' in edf.columns else ('risk' if 'risk' in edf.columns else 'target_risk')
        print(f"{h} columns sample: {list(edf.columns)[:5]}, using risk_col='{risk_col}'")

        seq_lens = sdf.groupby('event_id_str').size().to_dict()
        risks = edf.set_index('event_id_str')[risk_col].to_dict()
        
        horizon_data[h] = {
            'events_set': set(edf['event_id_str']),
            'seq_lens': seq_lens,
            'risks': risks,
        }
        print(f"Loaded {h}: {len(edf)} events, {len(sdf)} observations")

    configs = [
        ("60/15/10/15", (0.60, 0.15, 0.10, 0.15)),
        ("65/15/10/10", (0.65, 0.15, 0.10, 0.10)),
        ("60/15/15/10", (0.60, 0.15, 0.15, 0.10)),
        ("70/10/10/10", (0.70, 0.10, 0.10, 0.10)),
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

        print(f"\n{'='*70}")
        print(f"CONFIGURATION: {name} (Total Eligible: {N})")
        print(f"Train: {len(p_tr)} ({len(p_tr)/N*100:.1f}%), Val: {len(p_va)} ({len(p_va)/N*100:.1f}%), Cal: {len(p_ca)} ({len(p_ca)/N*100:.1f}%), Test: {len(p_te)} ({len(p_te)/N*100:.1f}%)")
        print(f"{'='*70}")

        partitions = [("Train", p_tr), ("Validation", p_va), ("Calibration", p_ca), ("InternalTest", p_te)]
        
        for h in ["H2", "H3", "H5", "H6"]:
            h_info = horizon_data[h]
            print(f"\n  --- Horizon {h} ---")
            for pname, p_ids in partitions:
                qualifying = [ev for ev in p_ids if ev in h_info['events_set']]
                lens = [h_info['seq_lens'][ev] for ev in qualifying]
                crit = sum(1 for ev in qualifying if h_info['risks'][ev] >= -5.0)
                mean_l = np.mean(lens) if lens else 0
                med_l = np.median(lens) if lens else 0
                min_l = min(lens) if lens else 0
                max_l = max(lens) if lens else 0
                obs_count = sum(lens)
                print(f"    {pname:12s}: Events={len(qualifying):5d} | Obs={obs_count:6d} | Crit={crit:2d} | SeqLen: Mean={mean_l:4.2f}, Med={med_l:4.1f}, Range=[{min_l:2d}, {max_l:2d}]")

if __name__ == "__main__":
    fast_analyze_splits()
