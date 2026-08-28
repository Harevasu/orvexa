"""ORVEXA Phase 3B Framework Setup and Preprocessing Manifest Generator.

Generates preprocessor manifests for Phase 3B controlled experiments (M0, M1, M2, M3)
across warning horizons H2, H3, H5 under strict zero-leakage constraints.

DOES NOT TRAIN ANY MODELS.
DOES NOT EVALUATE ON TEST SET.
DOES NOT MODIFY FROZEN BASELINES OR MASTER SPLIT.
"""

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import compute_file_sha256
from orvexa.phase3b_config import (
    ExperimentID,
    Phase3BExperimentConfig,
    get_experiment_config,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.splitting import SplitManifest


def verify_frozen_baselines() -> None:
    """Verify SHA-256 hashes of all canonical datasets and split manifest."""
    expected_hashes = {
        "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
        "data/processed/events/events_H2.csv": "3977a0b8adaaa6eeb29b107381f5ed19856e9e9adb44b1f511574fab547c8dd3",
        "data/processed/events/events_H3.csv": "6840e2c7ffdcdafaec46172b3051bce2063bb51c2a5b22a2064473902090f049",
        "data/processed/events/events_H5.csv": "89c427f05285606da42b2004a2e6175547cf78834e5f900e66d9f22cf859a51a",
        "data/processed/events/sequences_H2.csv": "4ccc7ddc779c53d99ad5d0775ed5e4d87d1b6470062dc0921fbbaeedd3bc8c0c",
        "data/processed/events/sequences_H3.csv": "7901ebbc5b073c31fffe0f967a96ea1ffdea41034771f6acccb2a3d089b9a097",
        "data/processed/events/sequences_H5.csv": "fb5ca14f20d7dfbe07ac74b5d5a772ebd4ed81dd3f234e87b31b4e1099442243",
        "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
    }

    print("\n[Step 1] Verifying frozen baseline SHA-256 hashes...")
    for fpath, expected_h in expected_hashes.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required canonical file: {fpath}")
        actual_h = compute_file_sha256(fpath)
        if actual_h != expected_h:
            raise ValueError(f"Integrity violation on {fpath}! Expected {expected_h}, got {actual_h}")
        print(f"  Verified: {fpath} ({actual_h[:16]}...)")
    print("All canonical files 100% frozen and verified.")


def main() -> None:
    print("=" * 70)
    print("ORVEXA — PHASE 3B EXPERIMENTAL FRAMEWORK SETUP")
    print("=" * 70)

    # 1. Baseline verification
    verify_frozen_baselines()

    # 2. Ensure Phase 3B directories exist
    dirs_to_create = [
        "artifacts/models/phase3b",
        "artifacts/preprocessors/phase3b",
        "data/processed/predictions/phase3b",
        "reports/phase3b",
        "configs",
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}")

    # 3. Save Experiment Matrix Configuration
    configs_dict = {
        exp_id.value: get_experiment_config(exp_id).to_dict()
        for exp_id in [
            ExperimentID.M0,
            ExperimentID.M1,
            ExperimentID.M2,
            ExperimentID.M3,
            ExperimentID.M4,
            ExperimentID.M5,
        ]
    }
    config_file_path = "configs/phase3b_experiments.json"
    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(configs_dict, f, indent=2)
    print(f"\nSaved Phase 3B Experiment Matrix Configuration -> {config_file_path}")

    # 4. Load Master Split Manifest
    manifest_path = "artifacts/splits/master_split_manifest.json"
    manifest = SplitManifest.load(manifest_path)
    train_ids_set = set(manifest.train_event_ids)
    val_ids_set = set(manifest.val_event_ids)
    print(f"\nMaster Split Manifest Loaded: {len(manifest.train_event_ids)} Train, {len(manifest.val_event_ids)} Val, {len(manifest.test_event_ids)} Test")

    # 5. Load Raw CDM Rows grouped by event_id
    raw_path = "data/raw/esa/train_data.csv"
    print(f"\nLoading raw CDM observations from {raw_path}...")
    t0 = time.time()
    raw_rows_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows_by_event[row["event_id"]].append(row)
    print(f"Loaded {len(raw_rows_by_event)} events in {time.time() - t0:.2f}s")

    # 6. Generate Preprocessor Manifests for M0, M1, M2, M3 across H2, H3, H5
    horizons = [2.0, 3.0, 5.0]
    experiments = [ExperimentID.M0, ExperimentID.M1, ExperimentID.M2, ExperimentID.M3]

    print("\n[Step 2] Fitting & serializing Phase 3B Sequence Preprocessors (Train Only)...")

    summary_records = []

    for h in horizons:
        h_str = f"H{int(h)}"
        cutoff = h
        print(f"\n--- Warning Horizon {h_str} (Cutoff = {cutoff} days) ---")

        # Extract events for this horizon
        h_train_events = {eid: raw_rows_by_event[eid] for eid in manifest.train_event_ids if eid in raw_rows_by_event}
        h_val_events = {eid: raw_rows_by_event[eid] for eid in manifest.val_event_ids if eid in raw_rows_by_event}

        for exp_id in experiments:
            exp_config = get_experiment_config(exp_id)
            preprocessor = Phase3BSequencePreprocessor(config=exp_config)

            # Prepare training sequence tensors
            X_tr_raw, mask_tr, y_tr, ids_tr = preprocessor.prepare_sequence_tensors(
                h_train_events, horizon_cutoff=cutoff
            )
            # Prepare validation sequence tensors (for contract verification only, no fitting)
            X_val_raw, mask_val, y_val, ids_val = preprocessor.prepare_sequence_tensors(
                h_val_events, horizon_cutoff=cutoff
            )

            # Fit preprocessor strictly on TRAIN sequence tensors
            preprocessor.fit(X_tr_raw, mask_tr)

            # Transform train and val to verify tensor contract and shapes
            X_tr_norm = preprocessor.transform(X_tr_raw, mask_tr)
            X_val_norm = preprocessor.transform(X_val_raw, mask_val)

            # Verify shapes
            expected_c = exp_config.n_channels
            actual_tr_shape = tuple(X_tr_norm.shape) if hasattr(X_tr_norm, "shape") else (len(X_tr_norm), len(X_tr_norm[0]), len(X_tr_norm[0][0]))
            actual_val_shape = tuple(X_val_norm.shape) if hasattr(X_val_norm, "shape") else (len(X_val_norm), len(X_val_norm[0]), len(X_val_norm[0][0]))

            assert actual_tr_shape[1] == expected_c, f"Channel count mismatch on {exp_id.value} {h_str}: expected {expected_c}, got {actual_tr_shape[1]}"
            assert actual_tr_shape[2] == 23, f"Sequence length mismatch on {exp_id.value} {h_str}: expected 23, got {actual_tr_shape[2]}"
            assert actual_val_shape[1] == expected_c, f"Val channel count mismatch on {exp_id.value} {h_str}"

            # Save manifest
            manifest_out_path = f"artifacts/preprocessors/phase3b/preprocessor_{exp_id.value}_h{h:.1f}.json"
            preprocessor.save(manifest_out_path)

            print(f"  [{exp_id.value}] Channels={expected_c} | Train Shape={actual_tr_shape} | Val Shape={actual_val_shape} -> Saved {manifest_out_path}")

            summary_records.append({
                "horizon": h_str,
                "experiment_id": exp_id.value,
                "n_channels": expected_c,
                "train_samples": actual_tr_shape[0],
                "val_samples": actual_val_shape[0],
                "max_seq_len": actual_tr_shape[2],
                "manifest_path": manifest_out_path,
            })

    # 7. Final Immutability Check
    print("\n[Step 3] Final Immutability Check on Canonical Datasets...")
    verify_frozen_baselines()
    print("\nPhase 3B Framework Setup complete with 0 models trained and 0 test-set touches.")


if __name__ == "__main__":
    main()
