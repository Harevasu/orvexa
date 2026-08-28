"""Scratch script for ORVEXA Phase 4A Step 3: H6 M4 Forensic Audit & Candidate Freeze."""

import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import compute_file_sha256
from orvexa.models_tcn import TCNRiskModel
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest

# 1. Authoritative Hashes
AUTHORITATIVE_HASHES = {
    "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
    "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
    "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
    "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
    "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
    "data/processed/events/events_H6.csv": "bbc4a34ebbc900f6344d3380dc14faa1b691c2180e1ad875586eb031b5d7cee9",
    "data/processed/events/sequences_H6.csv": "ad31bc8e99ec8cf720fd4645fb571d2d906d2e9bd1fb961613c99dee514c8817",
    "reports/phase4a/h6_dataset_manifest.json": "318ff7226e2239f19a382e76e15bd6a6f7c59ded4d795d54c94ad2e20714dbab",
    "artifacts/models/phase4a/tcn_best_M4_h6.0.pt": "9bb5f10b990be67336dd0902ab3943c28b20b21d18855f2ab4e9b4ca31844d30",
    "artifacts/models/phase4a/tcn_best_M4_h6.0.json": "99fc8138f877a26567e2e01388293709f8a1f8364ad76c3342274b7b5d25784b",
    "artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json": "e2c5e17769efb93358bfe9308591aefbf30f2b10045cbd970ae0b2c5b0fc5a62",
    "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv": "ff90ec2732eae47d4032a92b99139fd766c10cadfc038326a5eec41a9bdc1e8f",
}


def audit_frozen_hashes():
    print("=== 1. VERIFYING AUTHORITATIVE HASHES ===")
    for path, expected in AUTHORITATIVE_HASHES.items():
        assert os.path.exists(path), f"File missing: {path}"
        actual = compute_file_sha256(path)
        assert actual == expected, f"Hash mismatch on {path}!\n  Exp: {expected}\n  Act: {actual}"
        print(f"  [PASS] {path} -> {actual[:16]}...")
    print("All authoritative hashes match 100%.")


def audit_splits():
    print("\n=== 2. AUDITING H6 SPLIT INHERITANCE ===")
    manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
    tr_set = set(manifest.train_event_ids)
    va_set = set(manifest.val_event_ids)
    te_set = set(manifest.test_event_ids)

    with open("data/processed/events/events_H6.csv", "r", encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    eids = [r["event_id"] for r in events]
    assert len(eids) == 8426, f"Expected 8,426 events, got {len(eids)}"

    h6_tr = [e for e in eids if e in tr_set]
    h6_va = [e for e in eids if e in va_set]
    h6_te = [e for e in eids if e in te_set]

    print(f"  H6 Train Events: {len(h6_tr):,} (Expected 5,883)")
    print(f"  H6 Val Events:   {len(h6_va):,} (Expected 1,264)")
    print(f"  H6 Test Events:  {len(h6_te):,} (Expected 1,279)")

    assert len(h6_tr) == 5883
    assert len(h6_va) == 1264
    assert len(h6_te) == 1279
    assert len(set(h6_tr).intersection(set(h6_va))) == 0
    assert len(set(h6_tr).intersection(set(h6_te))) == 0
    assert len(set(h6_va).intersection(set(h6_te))) == 0
    print("Split inheritance and disjointness: 100% PASS.")


def audit_preprocessor():
    print("\n=== 3. AUDITING PREPROCESSOR CONFIGURATION ===")
    prep_path = "artifacts/preprocessors/phase4a/preprocessor_M4_h6.0.json"
    with open(prep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["n_channels"] == 37, f"Expected 37 channels, got {data['n_channels']}"
    assert len(data["feature_columns"]) == 37
    assert len(data["channel_stats"]) == 37

    # Check one-hot channels
    cat_expected = [
        "c_object_type__DEBRIS",
        "c_object_type__PAYLOAD",
        "c_object_type__ROCKET_BODY",
        "c_object_type__UNKNOWN",
    ]
    for ch in cat_expected:
        assert ch in data["feature_columns"], f"Missing categorical channel {ch}"

    # Check log10 channels
    log10_expected = [
        "t_sigma_r",
        "t_sigma_t",
        "t_sigma_n",
        "c_sigma_r",
        "c_sigma_t",
        "c_sigma_n",
        "mahalanobis_distance",
    ]
    for ch in log10_expected:
        assert ch in data["channel_stats"], f"Missing channel stat for {ch}"
        assert data["channel_stats"][ch]["is_log10"] is True, f"Expected {ch} to have is_log10=True"

    print(f"  Preprocessor channel count: {data['n_channels']}")
    print(f"  Categorical one-hot channels (4): {cat_expected}")
    print(f"  Log10 scaled channels (7): {log10_expected}")
    print("Preprocessor audit: 100% PASS.")


def audit_model_architecture():
    print("\n=== 4. AUDITING MODEL ARCHITECTURE & CHECKPOINT ===")
    model_json_path = "artifacts/models/phase4a/tcn_best_M4_h6.0.json"
    model_pt_path = "artifacts/models/phase4a/tcn_best_M4_h6.0.pt"

    with open(model_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "masked_causal_tcn"
    assert meta["in_features"] == 37
    assert meta["channels"] == [64, 64, 128]
    assert meta["kernel_size"] == 3
    assert meta["dilations"] == [1, 2, 4]
    assert meta["dropout"] == 0.15
    assert meta["max_seq_len"] == 23
    assert meta["seed"] == 42
    assert meta["config"]["huber_delta"] == 1.0
    assert meta["config"]["experiment_id"] == "M4"
    assert meta["config"]["horizon"] == 6.0

    ckpt = torch.load(model_pt_path, map_location="cpu")
    assert "network.0.conv1.conv.weight" in ckpt
    conv_w = ckpt["network.0.conv1.conv.weight"]
    assert conv_w.shape == (64, 37, 3)

    # Check for NaN / Inf in weights
    for k, v in ckpt.items():
        assert not torch.isnan(v).any(), f"NaN in weight {k}"
        assert not torch.isinf(v).any(), f"Inf in weight {k}"

    print(f"  Input Features: {meta['in_features']}")
    print(f"  Channel Progression: {meta['channels']}")
    print(f"  Kernel: {meta['kernel_size']}, Dilations: {meta['dilations']}, Dropout: {meta['dropout']}")
    print(f"  Weight Tensor Shape: {conv_w.shape}")
    print(f"  Weight Integrity: 0 NaN, 0 Inf across {len(ckpt)} parameter tensors.")
    print("Model architecture audit: 100% PASS.")


def audit_validation_metrics_reproduction():
    print("\n=== 5. INDEPENDENT REPRODUCTION OF VALIDATION METRICS ===")
    pred_csv_path = "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv"
    with open(pred_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_samples = len(rows)
    assert n_samples == 1264, f"Expected 1,264 validation rows, got {n_samples}"

    y_true = [float(r["final_risk"]) for r in rows]
    y_pred = [float(r["predicted_risk"]) for r in rows]

    reg = compute_regression_metrics(y_true, y_pred)
    rank = compute_ranking_metrics(y_true, y_pred, threshold_log10=-5.0)

    # Tail metrics
    tail_yt = [yt for yt, yp in zip(y_true, y_pred) if yt >= -5.0]
    tail_yp = [yp for yt, yp in zip(y_true, y_pred) if yt >= -5.0]
    residuals = [yp - yt for yt, yp in zip(tail_yt, tail_yp)]

    tail_mae = float(np.mean([abs(r) for r in residuals]))
    tail_rmse = float(np.sqrt(np.mean([r**2 for r in residuals])))
    tail_mean_res = float(np.mean(residuals))
    tail_median_res = float(np.median(residuals))

    # Read stored metrics
    with open("reports/phase4a_step2_metrics.csv", "r", encoding="utf-8") as f:
        m_rows = list(csv.DictReader(f))
    stored_m4 = [r for r in m_rows if r["model"] == "M4"][0]

    reproduced = {
        "val_mae": reg["mae"],
        "val_rmse": reg["rmse"],
        "val_r2": reg["r2"],
        "val_pearson": reg["pearson_correlation"],
        "val_spearman": reg["spearman_correlation"],
        "val_recall_top1pct": rank["budget_pct_1"]["recall"],
        "val_precision_top1pct": rank["budget_pct_1"]["precision"],
        "val_recall_top5pct": rank["budget_pct_5"]["recall"],
        "val_precision_top5pct": rank["budget_pct_5"]["precision"],
        "val_recall_top10pct": rank["budget_pct_10"]["recall"],
        "val_precision_top10pct": rank["budget_pct_10"]["precision"],
        "val_missed_events_top10pct": rank["budget_pct_10"]["missed_high_risk"],
        "tail_mean_residual": round(tail_mean_res, 5),
        "tail_median_residual": round(tail_median_res, 5),
    }

    print("Comparing Reproduced vs Stored Metrics (Tolerance = 1e-6):")
    for k, v in reproduced.items():
        stored_v = float(stored_m4[k])
        diff = abs(v - stored_v)
        assert diff <= 1e-6, f"Discrepancy on {k}: Reproduced={v}, Stored={stored_v}, Diff={diff}"
        print(f"  {k:28s} | Reproduced: {v:10.5f} | Stored: {stored_v:10.5f} | Diff: {diff:.8f} [MATCH]")
    print("Validation metrics reproduction: 100% PASS.")


def audit_paired_bootstrap():
    print("\n=== 6. VERIFYING PAIRED BOOTSTRAP ANALYSIS (2,000 RESAMPLES) ===")
    m0_pred_path = "data/processed/predictions/phase4a/tcn_M0_h6.0_val_predictions.csv"
    m4_pred_path = "data/processed/predictions/phase4a/tcn_M4_h6.0_val_predictions.csv"

    with open(m0_pred_path, "r", encoding="utf-8") as f0, open(m4_pred_path, "r", encoding="utf-8") as f4:
        r0 = list(csv.DictReader(f0))
        r4 = list(csv.DictReader(f4))

    assert len(r0) == len(r4) == 1264

    # Verify event ID alignment
    for row0, row4 in zip(r0, r4):
        assert row0["event_id"] == row4["event_id"]

    yt = np.array([float(r["final_risk"]) for r in r0], dtype=np.float64)
    yp0 = np.array([float(r["predicted_risk"]) for r in r0], dtype=np.float64)
    yp4 = np.array([float(r["predicted_risk"]) for r in r4], dtype=np.float64)

    rng = np.random.default_rng(42)
    n = len(yt)
    n_boot = 2000

    boot_d_mae = np.zeros(n_boot)
    boot_d_rmse = np.zeros(n_boot)
    boot_d_r2 = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub_yt = yt[idx]
        sub_p0 = yp0[idx]
        sub_p4 = yp4[idx]

        # MAE
        m0 = np.mean(np.abs(sub_yt - sub_p0))
        m4 = np.mean(np.abs(sub_yt - sub_p4))
        boot_d_mae[b] = m4 - m0

        # RMSE
        r0_val = np.sqrt(np.mean((sub_yt - sub_p0)**2))
        r4_val = np.sqrt(np.mean((sub_yt - sub_p4)**2))
        boot_d_rmse[b] = r4_val - r0_val

        # R2
        sub_mean = np.mean(sub_yt)
        ss_tot = np.sum((sub_yt - sub_mean)**2)
        if ss_tot > 1e-12:
            r2_0 = 1.0 - np.sum((sub_yt - sub_p0)**2) / ss_tot
            r2_4 = 1.0 - np.sum((sub_yt - sub_p4)**2) / ss_tot
            boot_d_r2[b] = r2_4 - r2_0

    d_mae_pt = float(np.mean(np.abs(yt - yp4)) - np.mean(np.abs(yt - yp0)))
    d_rmse_pt = float(np.sqrt(np.mean((yt - yp4)**2)) - np.sqrt(np.mean((yt - yp0)**2)))
    ss_tot_all = np.sum((yt - np.mean(yt))**2)
    r2_0_all = 1.0 - np.sum((yt - yp0)**2) / ss_tot_all
    r2_4_all = 1.0 - np.sum((yt - yp4)**2) / ss_tot_all
    d_r2_pt = float(r2_4_all - r2_0_all)

    ci_mae = np.percentile(boot_d_mae, [2.5, 97.5])
    ci_rmse = np.percentile(boot_d_rmse, [2.5, 97.5])
    ci_r2 = np.percentile(boot_d_r2, [2.5, 97.5])

    print(f"  Delta MAE:  {d_mae_pt:.4f} [95% CI: {ci_mae[0]:.4f}, {ci_mae[1]:.4f}]")
    print(f"  Delta RMSE: {d_rmse_pt:.4f} [95% CI: {ci_rmse[0]:.4f}, {ci_rmse[1]:.4f}]")
    print(f"  Delta R2:   {d_r2_pt:.4f} [95% CI: {ci_r2[0]:.4f}, {ci_r2[1]:.4f}]")

    assert abs(d_mae_pt - (-1.07868)) < 1e-4
    assert abs(d_rmse_pt - (-1.23379)) < 1e-4
    assert abs(d_r2_pt - (0.26180)) < 1e-4
    print("Paired bootstrap audit: 100% PASS.")


def audit_convergence():
    print("\n=== 7. AUDITING TRAINING CONVERGENCE ===")
    with open("reports/phase4a/step2_diagnostics_summary.json", "r", encoding="utf-8") as f:
        diag = json.load(f)

    m4_diag = diag["models"]["M4"]["training_diagnostics"]
    print(f"  Best Epoch: {m4_diag['best_epoch']} (Total Epochs: {m4_diag['total_epochs_run']})")
    print(f"  Best Val Loss (Huber): {m4_diag['best_val_loss']:.5f}")
    print(f"  Final Train Loss: {m4_diag['final_train_loss']:.5f}")
    print(f"  Final Val Loss:   {m4_diag['final_val_loss']:.5f}")

    assert m4_diag["best_epoch"] == 46
    assert m4_diag["total_epochs_run"] == 50
    assert abs(m4_diag["best_val_loss"] - 4.64681) < 1e-4
    assert abs(m4_diag["final_train_loss"] - 3.10520) < 1e-4
    assert abs(m4_diag["final_val_loss"] - 4.72309) < 1e-4
    print("Convergence audit: 100% PASS.")


if __name__ == "__main__":
    audit_frozen_hashes()
    audit_splits()
    audit_preprocessor()
    audit_model_architecture()
    audit_validation_metrics_reproduction()
    audit_paired_bootstrap()
    audit_convergence()
    print("\n==================================================")
    print("ALL PHASE 4A STEP 3 FORENSIC AUDITS PASSED!")
    print("==================================================")
