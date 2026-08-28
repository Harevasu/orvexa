"""
Check event_id types and split manifest intersection.
"""
import os
import sys
import json
import pandas as pd

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def check_event_ids():
    split_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(split_path, "r") as f:
        manifest = json.load(f)

    tr = set(manifest.get("train_event_ids", []))
    va = set(manifest.get("val_event_ids", []))
    te = set(manifest.get("test_event_ids", []))

    print(f"Sample train IDs from manifest (type={type(list(tr)[0])}): {list(tr)[:5]}")

    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        edf = pd.read_csv(efile)
        # Check event_id type
        print(f"Sample event_ids from {h} (type={type(edf['event_id'].iloc[0])}): {edf['event_id'].iloc[:5].tolist()}")
        # convert edf['event_id'] to string
        h_events_str = set(edf['event_id'].astype(str).unique())
        print(f"Horizon {h} with string conversion: tr={len(h_events_str & tr)}, va={len(h_events_str & va)}, te={len(h_events_str & te)}, total={len(h_events_str)}")

if __name__ == "__main__":
    check_event_ids()
