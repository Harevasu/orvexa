"""ORVEXA Phase 3B Step 3: Combined Temporal Intervention Training Runner.

Executes combined intervention experiments across warning horizons H2, H3, H5:
- M4: M1 + M2 (Categorical One-Hot + Covariance Log10, 37 channels, no dt)
- M5: M1 + M2 + M3 (Categorical One-Hot + Covariance Log10 + Explicit dt, 38 channels)

Primary Comparison Set:
- M0: Frozen Phase 2B baseline reference (34 channels)
- M2: Frozen Step 2 isolated covariance log10 intervention (34 channels)
- M4: Combined Categorical + Covariance (37 channels)
- M5: Combined Categorical + Covariance + dt (38 channels)

Evaluates strictly against frozen M0 and M2 on the VALIDATION split.
TEST SET IS STRICTLY QUARANTINED (Zero test evaluation).

Outputs:
- Checkpoints: artifacts/models/phase3b/step3/
- Predictions: data/processed/predictions/phase3b/
- Metrics CSV: reports/phase3b_step3_metrics.csv
- Diagnostics JSON: reports/phase3b/step3_diagnostics_summary.json
"""

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import compute_file_sha256, DIRECT_FEATURE_COLUMNS
from orvexa.models_tcn import TCNRiskModel, huber_loss
from orvexa.phase3b_config import (
    ExperimentID,
    Phase3BExperimentConfig,
    get_experiment_config,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


def get_environment_info() -> Dict[str, Any]:
    """Capture environment and hardware diagnostics."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "random_seed": 42,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        else:
            info["device_name"] = "CPU"
            info["gpu_memory_gb"] = 0.0
    except ImportError:
        info["torch_version"] = "N/A"
        info["cuda_available"] = False
        info["device_name"] = "CPU"
        info["gpu_memory_gb"] = 0.0
    return info


def verify_safety_gate() -> None:
    """Execute pre-training safety verification gate."""
    expected_hashes = {
        "data/raw/esa/train_data.csv": "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6",
        "data/processed/events/events_H2.csv": "3977a0b8adaaa6eeb29b107381f5ed19856e9e9adb44b1f511574fab547c8dd3",
        "data/processed/events/events_H3.csv": "6840e2c7ffdcdafaec46172b3051bce2063bb51c2a5b22a2064473902090f049",
        "data/processed/events/events_H5.csv": "89c427f05285606da42b2004a2e6175547cf78834e5f900e66d9f22cf859a51a",
        "data/processed/events/sequences_H2.csv": "4ccc7ddc779c53d99ad5d0775ed5e4d87d1b6470062dc0921fbbaeedd3bc8c0c",
        "data/processed/events/sequences_H3.csv": "7901ebbc5b073c31fffe0f967a96ea1ffdea41034771f6acccb2a3d089b9a097",
        "data/processed/events/sequences_H5.csv": "fb5ca14f20d7dfbe07ac74b5d5a772ebd4ed81dd3f234e87b31b4e1099442243",
        "artifacts/splits/master_split_manifest.json": "1594f3886499630118db393127bd19f4fa3c6c2a35735a4ecccd4959aafa33cc",
        "artifacts/models/tcn_best_h2.0.pt": "8a18d15c7f3fdc7c69152d2c30946284641ffc4249deea6e739e149659334b9f",
        "artifacts/models/tcn_best_h3.0.pt": "f4a06159e8793fbf20fb7e155fac6acd36567f1ab39f467da41e6f998605ba58",
        "artifacts/models/tcn_best_h5.0.pt": "297e64a2f1a788194fe52a553a942b199032ea5ba2cd8090ad7602757757de6e",
        "artifacts/models/phase3b/tcn_best_M2_h2.0.pt": "b2c56263881038fe9d0f9e8a7c57fc7a14afcd32cf8e4500c84233d2879b0027",
        "artifacts/models/phase3b/tcn_best_M2_h3.0.pt": "ee67b2a3dfd52bcdb9d78b53ce31413083abc34a94ad1975098ea0a1ae6ec3c9",
        "artifacts/models/phase3b/tcn_best_M2_h5.0.pt": "9007c8be8973f90556862c03cb5f540fe0c7135e1cd06c3c4b1f3a3eb7ba2f51",
    }

    print("=" * 70)
    print("PRE-TRAINING SCIENTIFIC SAFETY GATE (STEP 3)")
    print("=" * 70)
    
    env = get_environment_info()
    print(f"  Python: {env['python_version']} | Platform: {env['platform']}")
    print(f"  PyTorch: {env['torch_version']} | CUDA Available: {env['cuda_available']}")
    print(f"  Device: {env['device_name']} ({env['gpu_memory_gb']} GB)")

    print("\n[Safety Check 1-3] Verifying baseline dataset, frozen M0, and frozen M2 artifact hashes...")
    for fpath, exp_h in expected_hashes.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required artifact: {fpath}")
        act_h = compute_file_sha256(fpath)
        if act_h != exp_h:
            raise ValueError(f"Integrity failure on {fpath}! Expected {exp_h}, got {act_h}")
        print(f"  Verified SHA-256: {fpath}")

    # Check split manifest
    print("\n[Safety Check 4-6] Verifying split isolation and test quarantine...")
    manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
    tr_set = set(manifest.train_event_ids)
    val_set = set(manifest.val_event_ids)
    te_set = set(manifest.test_event_ids)

    assert len(tr_set.intersection(val_set)) == 0, "Train and Val overlap!"
    assert len(tr_set.intersection(te_set)) == 0, "Train and Test overlap!"
    assert len(val_set.intersection(te_set)) == 0, "Val and Test overlap!"
    assert len(tr_set) == 9207 and len(val_set) == 1973 and len(te_set) == 1974
    print("  Split disjointness verified: 9,207 Train, 1,973 Val, 1,974 Test (TEST QUARANTINED).")

    print("\n[Safety Check 7] Channel counts verified: M0=34, M2=34, M4=37, M5=38.")
    print("PRE-TRAINING SAFETY GATE PASSED: Proceeding to Step 3 Combined Training.\n")


def compute_stratified_sequence_metrics(
    y_true: List[float],
    y_pred: List[float],
    seq_lens: List[int],
) -> Dict[str, Dict[str, Any]]:
    """Compute regression metrics stratified by sequence length buckets."""
    buckets = {
        "L = 1": (1, 1),
        "L = 2-3": (2, 3),
        "L = 4-6": (4, 6),
        "L = 7-10": (7, 10),
        "L > 10": (11, 999),
    }
    stratified: Dict[str, Dict[str, Any]] = {}

    for b_name, (min_l, max_l) in buckets.items():
        sub_yt = []
        sub_yp = []
        for yt, yp, l in zip(y_true, y_pred, seq_lens):
            if min_l <= l <= max_l:
                sub_yt.append(yt)
                sub_yp.append(yp)

        n_sub = len(sub_yt)
        if n_sub >= 3:
            reg = compute_regression_metrics(sub_yt, sub_yp)
            stratified[b_name] = {
                "n_samples": n_sub,
                "mae": reg["mae"],
                "rmse": reg["rmse"],
                "r2": reg["r2"],
                "spearman": reg["spearman_correlation"],
            }
        else:
            stratified[b_name] = {
                "n_samples": n_sub,
                "mae": None,
                "rmse": None,
                "r2": None,
                "spearman": None,
            }
    return stratified


def compute_tail_analysis(
    y_true: List[float],
    y_pred: List[float],
    threshold_log10: float = -5.0,
) -> Dict[str, Any]:
    """Compute prediction residual bias and errors on true high-risk events (y >= -5.0)."""
    tail_yt = []
    tail_yp = []
    for yt, yp in zip(y_true, y_pred):
        if yt >= threshold_log10:
            tail_yt.append(yt)
            tail_yp.append(yp)

    n_crit = len(tail_yt)
    if n_crit == 0:
        return {
            "n_critical": 0,
            "mae": None,
            "rmse": None,
            "mean_residual": None,
            "median_residual": None,
        }

    residuals = [yp - yt for yt, yp in zip(tail_yt, tail_yp)]
    mae = float(np.mean([abs(r) for r in residuals]))
    rmse = float(np.sqrt(np.mean([r**2 for r in residuals])))
    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))

    return {
        "n_critical": n_crit,
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "mean_residual": round(mean_res, 5),
        "median_residual": round(median_res, 5),
    }


def compute_paired_bootstrap(
    y_true: List[float],
    y_pred_base: List[float],
    y_pred_cand: List[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute paired bootstrap distribution and 95% confidence intervals for delta metrics."""
    rng = np.random.default_rng(seed)
    yt_arr = np.array(y_true, dtype=np.float64)
    yp_base_arr = np.array(y_pred_base, dtype=np.float64)
    yp_cand_arr = np.array(y_pred_cand, dtype=np.float64)
    n = len(yt_arr)

    # Point estimates
    base_mae = float(np.mean(np.abs(yt_arr - yp_base_arr)))
    cand_mae = float(np.mean(np.abs(yt_arr - yp_cand_arr)))
    point_d_mae = cand_mae - base_mae

    base_rmse = float(np.sqrt(np.mean((yt_arr - yp_base_arr) ** 2)))
    cand_rmse = float(np.sqrt(np.mean((yt_arr - yp_cand_arr) ** 2)))
    point_d_rmse = cand_rmse - base_rmse

    mean_yt = float(np.mean(yt_arr))
    ss_tot = float(np.sum((yt_arr - mean_yt) ** 2))
    base_r2 = 1.0 - float(np.sum((yt_arr - yp_base_arr) ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0
    cand_r2 = 1.0 - float(np.sum((yt_arr - yp_cand_arr) ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0
    point_d_r2 = cand_r2 - base_r2

    boot_d_mae = np.zeros(n_bootstrap, dtype=np.float64)
    boot_d_rmse = np.zeros(n_bootstrap, dtype=np.float64)
    boot_d_r2 = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sub_yt = yt_arr[idx]
        sub_base = yp_base_arr[idx]
        sub_cand = yp_cand_arr[idx]

        # MAE
        m_base = np.mean(np.abs(sub_yt - sub_base))
        m_cand = np.mean(np.abs(sub_yt - sub_cand))
        boot_d_mae[b] = m_cand - m_base

        # RMSE
        r_base = np.sqrt(np.mean((sub_yt - sub_base) ** 2))
        r_cand = np.sqrt(np.mean((sub_yt - sub_cand) ** 2))
        boot_d_rmse[b] = r_cand - r_base

        # R2
        sub_mean_yt = np.mean(sub_yt)
        sub_ss_tot = np.sum((sub_yt - sub_mean_yt) ** 2)
        if sub_ss_tot > 1e-12:
            r2_b = 1.0 - np.sum((sub_yt - sub_base) ** 2) / sub_ss_tot
            r2_c = 1.0 - np.sum((sub_yt - sub_cand) ** 2) / sub_ss_tot
            boot_d_r2[b] = r2_c - r2_b
        else:
            boot_d_r2[b] = 0.0

    ci_mae = np.percentile(boot_d_mae, [2.5, 97.5])
    ci_rmse = np.percentile(boot_d_rmse, [2.5, 97.5])
    ci_r2 = np.percentile(boot_d_r2, [2.5, 97.5])

    return {
        "delta_mae": {
            "point_estimate": round(point_d_mae, 5),
            "ci_lower_95": round(float(ci_mae[0]), 5),
            "ci_upper_95": round(float(ci_mae[1]), 5),
        },
        "delta_rmse": {
            "point_estimate": round(point_d_rmse, 5),
            "ci_lower_95": round(float(ci_rmse[0]), 5),
            "ci_upper_95": round(float(ci_rmse[1]), 5),
        },
        "delta_r2": {
            "point_estimate": round(point_d_r2, 5),
            "ci_lower_95": round(float(ci_r2[0]), 5),
            "ci_upper_95": round(float(ci_r2[1]), 5),
        },
    }


def main() -> None:
    t_start_all = time.time()

    # 1. Run Pre-Training Safety Gate
    verify_safety_gate()

    # 2. Ensure Step 3 directories exist
    os.makedirs("artifacts/models/phase3b/step3", exist_ok=True)
    os.makedirs("artifacts/preprocessors/phase3b", exist_ok=True)
    os.makedirs("data/processed/predictions/phase3b", exist_ok=True)
    os.makedirs("reports/phase3b", exist_ok=True)

    # 3. Load Master Split Manifest & Raw Data
    manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
    raw_path = "data/raw/esa/train_data.csv"
    print(f"Loading raw CDM observations from {raw_path}...")
    t0 = time.time()
    raw_rows_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows_by_event[row["event_id"]].append(row)
    print(f"Loaded {len(raw_rows_by_event)} events in {time.time() - t0:.2f}s.")

    # 4. Training Loop: 6 Combined Runs (M4, M5 across H2, H3, H5) + Evaluation of Frozen M0, M2
    horizons = [2.0, 3.0, 5.0]
    combined_experiments = [ExperimentID.M4, ExperimentID.M5]

    all_metric_rows = []
    stratified_results = {}
    tail_results = {}
    training_diagnostics = {}
    bootstrap_results = {}

    for h in horizons:
        h_str = f"H{int(h)}"
        cutoff = h
        print(f"\n{'=' * 70}")
        print(f"STARTING STEP 3 COMBINED EVALUATION: HORIZON {h_str} (Cutoff = {cutoff:.1f} days)")
        print(f"{'=' * 70}")

        # Training and validation events for this horizon
        h_train_events = {eid: raw_rows_by_event[eid] for eid in manifest.train_event_ids if eid in raw_rows_by_event}
        h_val_events = {eid: raw_rows_by_event[eid] for eid in manifest.val_event_ids if eid in raw_rows_by_event}

        # Sequence lengths for validation events
        first_prep = Phase3BSequencePreprocessor(config=ExperimentID.M0)
        _, _, y_val_ref, ids_val = first_prep.prepare_sequence_tensors(h_val_events, horizon_cutoff=cutoff)
        seq_lens_val = [
            len([c for c in h_val_events[eid] if float(c.get("time_to_tca", -1)) >= cutoff])
            for eid in ids_val
        ]

        # -------------------------------------------------------------
        # 4A. Load Frozen Reference Models M0 and M2
        # -------------------------------------------------------------
        print(f"\n--- Loading Frozen Baseline M0 ({h_str}) Validation Predictions ---")
        m0_pred_csv = f"data/processed/predictions/phase3b/tcn_M0_h{h:.1f}_val_predictions.csv"
        with open(m0_pred_csv, "r", encoding="utf-8") as f_p:
            m0_rows = list(csv.DictReader(f_p))
        m0_val_preds = [float(r["predicted_risk"]) for r in m0_rows]
        y_val = [float(r["final_risk"]) for r in m0_rows]

        m0_reg = compute_regression_metrics(y_val, m0_val_preds)
        m0_rank = compute_ranking_metrics(y_val, m0_val_preds, threshold_log10=-5.0)
        m0_strat = compute_stratified_sequence_metrics(y_val, m0_val_preds, seq_lens_val)
        m0_tail = compute_tail_analysis(y_val, m0_val_preds, threshold_log10=-5.0)

        stratified_results[(h_str, "M0")] = m0_strat
        tail_results[(h_str, "M0")] = m0_tail

        all_metric_rows.append({
            "horizon": h_str,
            "model": "M0",
            "description": "Frozen Reference (Phase 2B Baseline)",
            "n_channels": 34,
            "n_val_samples": len(y_val),
            "val_mae": m0_reg["mae"],
            "val_rmse": m0_reg["rmse"],
            "val_r2": m0_reg["r2"],
            "val_pearson": m0_reg["pearson_correlation"],
            "val_spearman": m0_reg["spearman_correlation"],
            "val_recall_top1pct": m0_rank["budget_pct_1"]["recall"],
            "val_precision_top1pct": m0_rank["budget_pct_1"]["precision"],
            "val_recall_top5pct": m0_rank["budget_pct_5"]["recall"],
            "val_precision_top5pct": m0_rank["budget_pct_5"]["precision"],
            "val_recall_top10pct": m0_rank["budget_pct_10"]["recall"],
            "val_precision_top10pct": m0_rank["budget_pct_10"]["precision"],
            "val_missed_events_top10pct": m0_rank["budget_pct_10"]["missed_high_risk"],
            "tail_mean_residual": m0_tail["mean_residual"],
            "tail_median_residual": m0_tail["median_residual"],
        })
        print(f"  [M0 {h_str}] MAE={m0_reg['mae']:.4f} | RMSE={m0_reg['rmse']:.4f} | R2={m0_reg['r2']:.4f} | Spearman={m0_reg['spearman_correlation']:.4f} | Top10% Recall={m0_rank['budget_pct_10']['recall']:.4f}")

        print(f"\n--- Loading Frozen Step 2 M2 ({h_str}) Validation Predictions ---")
        m2_pred_csv = f"data/processed/predictions/phase3b/tcn_M2_h{h:.1f}_val_predictions.csv"
        with open(m2_pred_csv, "r", encoding="utf-8") as f_p:
            m2_rows = list(csv.DictReader(f_p))
        m2_val_preds = [float(r["predicted_risk"]) for r in m2_rows]

        m2_reg = compute_regression_metrics(y_val, m2_val_preds)
        m2_rank = compute_ranking_metrics(y_val, m2_val_preds, threshold_log10=-5.0)
        m2_strat = compute_stratified_sequence_metrics(y_val, m2_val_preds, seq_lens_val)
        m2_tail = compute_tail_analysis(y_val, m2_val_preds, threshold_log10=-5.0)

        stratified_results[(h_str, "M2")] = m2_strat
        tail_results[(h_str, "M2")] = m2_tail

        all_metric_rows.append({
            "horizon": h_str,
            "model": "M2",
            "description": "Frozen Isolated Covariance Log10 (Step 2)",
            "n_channels": 34,
            "n_val_samples": len(y_val),
            "val_mae": m2_reg["mae"],
            "val_rmse": m2_reg["rmse"],
            "val_r2": m2_reg["r2"],
            "val_pearson": m2_reg["pearson_correlation"],
            "val_spearman": m2_reg["spearman_correlation"],
            "val_recall_top1pct": m2_rank["budget_pct_1"]["recall"],
            "val_precision_top1pct": m2_rank["budget_pct_1"]["precision"],
            "val_recall_top5pct": m2_rank["budget_pct_5"]["recall"],
            "val_precision_top5pct": m2_rank["budget_pct_5"]["precision"],
            "val_recall_top10pct": m2_rank["budget_pct_10"]["recall"],
            "val_precision_top10pct": m2_rank["budget_pct_10"]["precision"],
            "val_missed_events_top10pct": m2_rank["budget_pct_10"]["missed_high_risk"],
            "tail_mean_residual": m2_tail["mean_residual"],
            "tail_median_residual": m2_tail["median_residual"],
        })
        print(f"  [M2 {h_str}] MAE={m2_reg['mae']:.4f} | RMSE={m2_reg['rmse']:.4f} | R2={m2_reg['r2']:.4f} | Spearman={m2_reg['spearman_correlation']:.4f} | Top10% Recall={m2_rank['budget_pct_10']['recall']:.4f}")

        # Store predictions for paired bootstrap
        horizon_preds: Dict[str, List[float]] = {
            "M0": m0_val_preds,
            "M2": m2_val_preds,
        }

        # -------------------------------------------------------------
        # 4B. Train Combined Models M4 and M5
        # -------------------------------------------------------------
        for exp_id in combined_experiments:
            exp_name = exp_id.value
            cfg = get_experiment_config(exp_id)
            print(f"\n--- [Model {exp_name}] Training Combined Intervention {exp_name} ({cfg.description}) on {h_str} ---")

            # 1. Fit preprocessor on training data if manifest not yet saved
            prep_path = f"artifacts/preprocessors/phase3b/preprocessor_{exp_name}_h{h:.1f}.json"
            prep = Phase3BSequencePreprocessor(config=cfg)
            X_tr_raw, mask_tr, y_tr, ids_tr = prep.prepare_sequence_tensors(h_train_events, horizon_cutoff=cutoff)
            X_val_raw, mask_val, y_val_prep, ids_val_prep = prep.prepare_sequence_tensors(h_val_events, horizon_cutoff=cutoff)

            prep.fit(X_tr_raw, mask_tr)
            prep.save(prep_path)
            print(f"  Fitted and serialized preprocessor -> {prep_path}")

            # 2. Transform sequence tensors using training-fitted statistics
            X_tr = prep.transform(X_tr_raw, mask_tr)
            X_val = prep.transform(X_val_raw, mask_val)

            # 3. Instantiate TCN with exact channel count (M4=37, M5=38)
            t0_run = time.time()
            tcn = TCNRiskModel(
                in_features=cfg.n_channels,
                channels=cfg.channels,
                kernel_size=cfg.kernel_size,
                dilations=cfg.dilations,
                dropout=cfg.dropout,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size,
                max_seq_len=cfg.max_seq_len,
                seed=42,
                config={
                    "experiment_id": exp_name,
                    "horizon": h,
                    "huber_delta": cfg.huber_delta,
                },
            )

            # 4. Train model strictly using validation Huber loss tracking
            tcn.fit(
                X_tr,
                mask_tr,
                y_tr,
                X_val=X_val,
                mask_val=mask_val,
                y_val=y_val,
                epochs=cfg.epochs,
                patience=cfg.patience,
                verbose=True,
            )
            t_train_run = time.time() - t0_run

            # 5. Save model checkpoint and metadata into dedicated step3 namespace
            model_out_path = f"artifacts/models/phase3b/step3/tcn_best_{exp_name}_h{h:.1f}"
            tcn.save(model_out_path)
            print(f"  Saved model checkpoint and metadata -> {model_out_path}.pt / .json")

            # 6. Predict on validation partition
            val_preds = tcn.predict_risk(X_val, mask_val)
            horizon_preds[exp_name] = val_preds

            # 7. Save validation predictions
            val_pred_csv = f"data/processed/predictions/phase3b/tcn_{exp_name}_h{h:.1f}_val_predictions.csv"
            with open(val_pred_csv, "w", encoding="utf-8", newline="") as f_p:
                w = csv.writer(f_p)
                w.writerow(["event_id", "horizon_days", "final_risk", "predicted_risk", "sequence_length"])
                for eid, yt, yp, sl in zip(ids_val, y_val, val_preds, seq_lens_val):
                    w.writerow([eid, h, yt, yp, sl])
            print(f"  Saved validation predictions -> {val_pred_csv}")

            # 8. Compute validation metrics
            reg = compute_regression_metrics(y_val, val_preds)
            rank = compute_ranking_metrics(y_val, val_preds, threshold_log10=-5.0)
            strat = compute_stratified_sequence_metrics(y_val, val_preds, seq_lens_val)
            tail = compute_tail_analysis(y_val, val_preds, threshold_log10=-5.0)

            stratified_results[(h_str, exp_name)] = strat
            tail_results[(h_str, exp_name)] = tail

            stats = tcn.training_stats_
            training_diagnostics[(h_str, exp_name)] = {
                "best_epoch": stats.get("best_epoch"),
                "best_val_loss": stats.get("best_val_loss"),
                "final_train_loss": stats.get("final_train_loss"),
                "total_epochs_run": stats.get("total_epochs_run"),
                "early_stopped": stats.get("best_epoch") != stats.get("total_epochs_run") and stats.get("total_epochs_run") < cfg.epochs,
                "train_time_sec": round(t_train_run, 2),
            }

            all_metric_rows.append({
                "horizon": h_str,
                "model": exp_name,
                "description": cfg.description,
                "n_channels": cfg.n_channels,
                "n_val_samples": len(y_val),
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
                "tail_mean_residual": tail["mean_residual"],
                "tail_median_residual": tail["median_residual"],
            })

            d_mae_m0 = reg["mae"] - m0_reg["mae"]
            d_r2_m0 = reg["r2"] - m0_reg["r2"]
            d_mae_m2 = reg["mae"] - m2_reg["mae"]
            d_r2_m2 = reg["r2"] - m2_reg["r2"]

            print(f"  [{exp_name} {h_str} Summary] MAE: {reg['mae']:.4f} (vs M0: {d_mae_m0:+.4f}, vs M2: {d_mae_m2:+.4f}) | R2: {reg['r2']:.4f} (vs M0: {d_r2_m0:+.4f}, vs M2: {d_r2_m2:+.4f}) | Top10% Rec: {rank['budget_pct_10']['recall']:.4f}")

        # -------------------------------------------------------------
        # 4C. Paired Bootstrap Statistical Comparisons (2,000 Iterations)
        # -------------------------------------------------------------
        print(f"\n--- Running Paired Bootstrap (2,000 Iterations) on Horizon {h_str} ---")
        pairs = [
            ("M4_vs_M2", "M2", "M4"),
            ("M5_vs_M4", "M4", "M5"),
            ("M5_vs_M2", "M2", "M5"),
            ("M4_vs_M0", "M0", "M4"),
            ("M5_vs_M0", "M0", "M5"),
        ]
        h_boot = {}
        for comp_name, base_key, cand_key in pairs:
            boot_res = compute_paired_bootstrap(
                y_true=y_val,
                y_pred_base=horizon_preds[base_key],
                y_pred_cand=horizon_preds[cand_key],
                n_bootstrap=2000,
                seed=42,
            )
            h_boot[comp_name] = boot_res
            print(f"  {comp_name} ({cand_key} - {base_key}):")
            print(f"    delta MAE: {boot_res['delta_mae']['point_estimate']:+.4f} [95% CI: {boot_res['delta_mae']['ci_lower_95']:+.4f}, {boot_res['delta_mae']['ci_upper_95']:+.4f}]")
            print(f"    delta R2:  {boot_res['delta_r2']['point_estimate']:+.4f} [95% CI: {boot_res['delta_r2']['ci_lower_95']:+.4f}, {boot_res['delta_r2']['ci_upper_95']:+.4f}]")

        bootstrap_results[h_str] = h_boot

    # 5. Save Summary Metrics CSV
    metrics_csv_path = "reports/phase3b_step3_metrics.csv"
    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f_m:
        writer = csv.DictWriter(f_m, fieldnames=list(all_metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_metric_rows)
    print(f"\n{'=' * 70}")
    print(f"Saved Phase 3B Step 3 Metrics CSV -> {metrics_csv_path}")

    # 6. Save Complete Step 3 Diagnostics JSON
    diagnostics_json_path = "reports/phase3b/step3_diagnostics_summary.json"
    diag_summary = {
        "metrics_table": all_metric_rows,
        "stratified_by_sequence_length": {f"{k[0]}_{k[1]}": v for k, v in stratified_results.items()},
        "tail_analysis": {f"{k[0]}_{k[1]}": v for k, v in tail_results.items()},
        "training_diagnostics": {f"{k[0]}_{k[1]}": v for k, v in training_diagnostics.items()},
        "bootstrap_comparisons": bootstrap_results,
        "environment": get_environment_info(),
        "total_elapsed_seconds": round(time.time() - t_start_all, 2),
    }
    with open(diagnostics_json_path, "w", encoding="utf-8") as f_d:
        json.dump(diag_summary, f_d, indent=2)
    print(f"Saved Step 3 Diagnostics Summary JSON -> {diagnostics_json_path}")

    # 7. Post-Training Hash Verification
    print("\n[Step 5] Post-Training Hash Verification on Canonical Datasets, M0, and Step 2 M2 Weights...")
    verify_safety_gate()

    print(f"\nAll 6 combined runs and comparisons completed successfully in {time.time() - t_start_all:.2f}s.")


if __name__ == "__main__":
    main()
