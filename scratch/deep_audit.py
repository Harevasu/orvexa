"""
Comprehensive deep inspection script for Phase 5 Step 1 Audit.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

def inspect_raw_and_processed():
    raw_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    df_raw = pd.read_csv(raw_path)
    print("=== RAW DATASET INFO ===")
    print(f"Total rows: {len(df_raw)}")
    print(f"Unique event_ids: {df_raw['event_id'].nunique()}")
    print(f"Columns ({len(df_raw.columns)}): {list(df_raw.columns)}")
    print(f"time_to_tca min/max: {df_raw['time_to_tca'].min():.6f} to {df_raw['time_to_tca'].max():.6f}")
    print(f"risk min/max: {df_raw['risk'].min():.6f} to {df_raw['risk'].max():.6f}")
    
    # Check mission_id
    if 'mission_id' in df_raw.columns:
        print(f"mission_id counts:\n{df_raw['mission_id'].value_counts()}")
        
    # Check space weather fields
    sw_cols = ['F10', 'F3M', 'SSN', 'AP']
    for c in sw_cols:
        if c in df_raw.columns:
            print(f"Space weather {c}: min={df_raw[c].min()}, max={df_raw[c].max()}, nulls={df_raw[c].isnull().sum()}")

    # Processed tables
    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        sfile = os.path.join(events_dir, f"sequences_{h}.csv")
        edf = pd.read_csv(efile)
        sdf = pd.read_csv(sfile)
        print(f"\n--- HORIZON {h} ---")
        print(f"Events table: rows={len(edf)}, cols={len(edf.columns)}, events={edf['event_id'].nunique()}")
        print(f"Sequences table: rows={len(sdf)}, cols={len(sdf.columns)}, events={sdf['event_id'].nunique()}")
        seq_lens = sdf.groupby('event_id').size()
        print(f"Sequences per event: mean={seq_lens.mean():.2f}, median={seq_lens.median()}, min={seq_lens.min()}, max={seq_lens.max()}")
        print(f"time_to_tca cutoff in seqs: min={sdf['time_to_tca'].min():.4f}, max={sdf['time_to_tca'].max():.4f}")

def inspect_split_manifest():
    print("\n=== SPLIT MANIFEST INSPECTION ===")
    split_path = os.path.join(WORKSPACE, "artifacts", "splits", "master_split_manifest.json")
    with open(split_path, "r") as f:
        manifest = json.load(f)
    print(f"Master split manifest keys: {list(manifest.keys())}")
    tr = set(manifest.get("train_event_ids", []))
    va = set(manifest.get("val_event_ids", []))
    te = set(manifest.get("test_event_ids", []))
    print(f"Global split counts: train={len(tr)}, val={len(va)}, test={len(te)}, total={len(tr)+len(va)+len(te)}")
    print(f"Overlap tr & va: {len(tr & va)}, tr & te: {len(tr & te)}, va & te: {len(va & te)}")

    # Check how each horizon filters this master split
    events_dir = os.path.join(WORKSPACE, "data", "processed", "events")
    for h in ["H2", "H3", "H5", "H6"]:
        efile = os.path.join(events_dir, f"events_{h}.csv")
        edf = pd.read_csv(efile)
        h_events = set(edf['event_id'].unique())
        h_tr = h_events & tr
        h_va = h_events & va
        h_te = h_events & te
        print(f"Horizon {h} split: train={len(h_tr)}, val={len(h_va)}, test={len(h_te)}, total={len(h_events)}")

def inspect_feature_sets():
    print("\n=== FEATURE SETS (M0 - M5) ===")
    from orvexa import phase3b_config
    print("M0 (BASE_FEATURES):", phase3b_config.BASE_FEATURES)
    print(f"  Count: {len(phase3b_config.BASE_FEATURES)}")
    print("M1 (M1_FEATURES):", phase3b_config.M1_FEATURES)
    print(f"  Count: {len(phase3b_config.M1_FEATURES)}")
    print("M4 (M4_FEATURES):", phase3b_config.M4_FEATURES)
    print(f"  Count: {len(phase3b_config.M4_FEATURES)}")
    print("M5 (M5_FEATURES):", phase3b_config.M5_FEATURES)
    print(f"  Count: {len(phase3b_config.M5_FEATURES)}")
    print("LOG_TRANSFORM_FEATURES:", phase3b_config.LOG_TRANSFORM_FEATURES)

def inspect_orbit_and_physics_modules():
    print("\n=== ORBIT & PHYSICS MODULES ===")
    import orvexa.orbit.coordinate_utils as cu
    import orvexa.orbit.omm_parser as op
    import orvexa.orbit.propagation as prop
    import orvexa.orbit.tle_parser as tp
    import orvexa.orbit.validation as ov
    import orvexa.models_physics as mp

    print("coordinate_utils:", [m for m in dir(cu) if not m.startswith("_")])
    print("omm_parser:", [m for m in dir(op) if not m.startswith("_")])
    print("propagation:", [m for m in dir(prop) if not m.startswith("_")])
    print("tle_parser:", [m for m in dir(tp) if not m.startswith("_")])
    print("validation:", [m for m in dir(ov) if not m.startswith("_")])
    print("models_physics:", [m for m in dir(mp) if not m.startswith("_")])

def inspect_models_and_metrics():
    print("\n=== MODELS & METRICS ===")
    import orvexa.models_tcn as mt
    import orvexa.regression_metrics as rm
    import orvexa.ranking_metrics as rkm
    import orvexa.classification_metrics as cm
    import orvexa.calibration as cal
    import orvexa.splitting as sp

    print("models_tcn:", [m for m in dir(mt) if not m.startswith("_")])
    print("regression_metrics:", [m for m in dir(rm) if not m.startswith("_")])
    print("ranking_metrics:", [m for m in dir(rkm) if not m.startswith("_")])
    print("classification_metrics:", [m for m in dir(cm) if not m.startswith("_")])
    print("calibration:", [m for m in dir(cal) if not m.startswith("_")])
    print("splitting:", [m for m in dir(sp) if not m.startswith("_")])

if __name__ == "__main__":
    inspect_raw_and_processed()
    inspect_split_manifest()
    inspect_feature_sets()
    inspect_orbit_and_physics_modules()
    inspect_models_and_metrics()
