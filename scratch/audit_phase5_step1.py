"""
Phase 5 Step 1: Comprehensive Read-Only Audit Script
Gathers all data, schema, model, split, physics, and codebase details.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def audit_datasets():
    print("=== DATASET AUDIT ===")
    raw_esa_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    if os.path.exists(raw_esa_path):
        size_mb = os.path.getsize(raw_esa_path) / (1024 * 1024)
        df_head = pd.read_csv(raw_esa_path, nrows=5)
        print(f"Raw ESA train_data.csv: exists, size={size_mb:.2f} MB, columns={len(df_head.columns)}")
        print(f"Raw columns: {list(df_head.columns)}")
    else:
        print("Raw ESA train_data.csv NOT FOUND")

    # Check external raw dirs
    celestrak_dir = os.path.join(WORKSPACE, "data", "raw", "celestrak")
    spacetrack_dir = os.path.join(WORKSPACE, "data", "raw", "space_track")
    print(f"celestrak dir contents: {os.listdir(celestrak_dir) if os.path.exists(celestrak_dir) else 'N/A'}")
    print(f"space_track dir contents: {os.listdir(spacetrack_dir) if os.path.exists(spacetrack_dir) else 'N/A'}")

    # Processed tables
    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        sfile = os.path.join(events_dir, f"sequences_{h}.csv")
        if os.path.exists(efile):
            edf = pd.read_csv(efile)
            print(f"{efile}: rows={len(edf)}, cols={len(edf.columns)}, unique_events={edf['event_id'].nunique() if 'event_id' in edf.columns else 'N/A'}")
        if os.path.exists(sfile):
            sdf = pd.read_csv(sfile)
            print(f"{sfile}: rows={len(sdf)}, cols={len(sdf.columns)}, unique_events={sdf['event_id'].nunique() if 'event_id' in sdf.columns else 'N/A'}")

def audit_manifests():
    print("\n=== MANIFESTS AUDIT ===")
    master_split_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    if os.path.exists(master_split_path):
        with open(master_split_path, "r") as f:
            manifest = json.load(f)
        print(f"Master split manifest keys: {list(manifest.keys())}")
        if "horizons" in manifest:
            for h, info in manifest["horizons"].items():
                print(f"  Horizon {h}: train_events={len(info.get('train_event_ids', []))}, val_events={len(info.get('val_event_ids', []))}, test_events={len(info.get('test_event_ids', []))}")
        elif "train_event_ids" in manifest:
            print(f"  Single split: train={len(manifest.get('train_event_ids', []))}, val={len(manifest.get('val_event_ids', []))}, test={len(manifest.get('test_event_ids', []))}")
    else:
        print("master_split_manifest.json NOT FOUND")

    # H6 dataset manifest
    h6_manifest = os.path.join(WORKSPACE, "reports", "phase4a", "h6_dataset_manifest.json")
    if os.path.exists(h6_manifest):
        with open(h6_manifest, "r") as f:
            h6_man = json.load(f)
        print(f"H6 manifest keys: {list(h6_man.keys())}")

def audit_preprocessors_and_models():
    print("\n=== PREPROCESSORS & MODELS AUDIT ===")
    models_dir = os.path.join(WORKSPACE, "artifacts", "models")
    preproc_dir = os.path.join(WORKSPACE, "artifacts", "preprocessors")
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            p = os.path.join(root, f)
            print(f"Model artifact: {os.path.relpath(p, WORKSPACE)} ({os.path.getsize(p)} bytes)")
    for root, dirs, files in os.walk(preproc_dir):
        for f in files:
            p = os.path.join(root, f)
            print(f"Preproc artifact: {os.path.relpath(p, WORKSPACE)} ({os.path.getsize(p)} bytes)")

if __name__ == "__main__":
    audit_datasets()
    audit_manifests()
    audit_preprocessors_and_models()
