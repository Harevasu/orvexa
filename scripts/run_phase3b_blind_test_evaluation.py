"""ORVEXA Phase 3B Step 4B: Blind Test Evaluation of Frozen M4 Candidate.

IMPORTANT GOVERNANCE RULES:
- FINAL BLIND EVALUATION.
- Test set accessed for prediction and evaluation ONLY.
- Zero retraining, zero fine-tuning, zero hyperparameter search, zero threshold tuning.
- All preprocessors operated in STRICT TRANSFORM-ONLY mode using frozen training parameters.
- Checkpoint SHA-256 hashes verified before and after evaluation.
- Independent recomputation routine verifies stored prediction CSV metrics.
"""

from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import compute_file_sha256
from orvexa.models_tcn import TCNRiskModel
from orvexa.phase3b_config import ExperimentID, get_experiment_config
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest

# Expected Frozen SHA-256 Hashes
EXPECTED_HASHES = {
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
    "artifacts/models/phase3b/step3/tcn_best_M4_h2.0.pt": "00df58e3108c4857d0e8256a23bb9ee3a0ba424682d83c16f75727adb9d0d2b2",
    "artifacts/models/phase3b/step3/tcn_best_M4_h3.0.pt": "7a0e906018fb052217064b1f5caed4a170bce27a7d48dc8d201daf9b0b8ba372",
    "artifacts/models/phase3b/step3/tcn_best_M4_h5.0.pt": "c5a5d0ebbfd7551d89c4f1456b357b3a265a272a9d76e8990608f3189e2f1091",
}


def verify_cryptographic_hashes(tag: str = "Pre-Evaluation") -> None:
    """Verify all frozen baseline and M4 checkpoint hashes."""
    print(f"\n======================================================================")
    print(f"[{tag}] Cryptographic SHA-256 Hash Verification")
    print(f"======================================================================")
    for fpath, exp_h in EXPECTED_HASHES.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required artifact: {fpath}")
        act_h = compute_file_sha256(fpath)
        if act_h != exp_h:
            raise ValueError(f"HASH MISMATCH for {fpath}!\nExpected: {exp_h}\nActual:   {act_h}")
        print(f"  [VERIFIED] {fpath}")
    print(f"[{tag}] All {len(EXPECTED_HASHES)} artifacts match expected SHA-256 hashes.")


def load_raw_dataset_events() -> Dict[str, List[Dict[str, Any]]]:
    """Load raw ESA training dataset grouped by event_id."""
    raw_path = "data/raw/esa/train_data.csv"
    events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            events[r["event_id"]].append(r)
    return events


def compute_tail_analysis(y_true: List[float], y_pred: List[float], threshold_log10: float = -5.0) -> Dict[str, Any]:
    """Compute high-risk tail residual diagnostics."""
    tail_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt >= threshold_log10]
    n_crit = len(tail_pairs)
    if n_crit == 0:
        return {
            "n_critical": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "mean_residual": 0.0,
            "median_residual": 0.0,
        }
    y_true_crit = [p[0] for p in tail_pairs]
    y_pred_crit = [p[1] for p in tail_pairs]
    residuals = [yp - yt for yt, yp in zip(y_true_crit, y_pred_crit)]
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(math.sqrt(np.mean(np.square(residuals))))
    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    return {
        "n_critical": n_crit,
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "mean_residual": round(mean_res, 5),
        "median_residual": round(median_res, 5),
    }


def compute_prediction_distribution(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Compute distribution statistics for target and predictions."""
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    yt_mean = float(np.mean(yt))
    yt_std = float(np.std(yt))
    yp_mean = float(np.mean(yp))
    yp_std = float(np.std(yp))
    mean_bias = float(np.mean(yp - yt))
    std_ratio = float(yp_std / yt_std) if yt_std > 1e-6 else 0.0
    yp_median = float(np.median(yp))
    return {
        "target_mean": round(yt_mean, 5),
        "target_std": round(yt_std, 5),
        "prediction_mean": round(yp_mean, 5),
        "prediction_std": round(yp_std, 5),
        "mean_bias": round(mean_bias, 5),
        "prediction_target_std_ratio": round(std_ratio, 5),
        "prediction_median": round(yp_median, 5),
    }


def run_blind_test_evaluation() -> None:
    """Execute final blind test evaluation of frozen M4 across H2, H3, H5."""
    t_start = time.time()

    print("=" * 70)
    print("ORVEXA — PHASE 3B STEP 4B: BLIND TEST EVALUATION (FROZEN M4)")
    print("=" * 70)

    # 1. Pre-Evaluation Cryptographic Gate
    verify_cryptographic_hashes(tag="Pre-Evaluation Gate")

    # 2. Load Split Manifest
    manifest_path = "artifacts/splits/master_split_manifest.json"
    manifest = SplitManifest.load(manifest_path)
    test_event_ids = set(manifest.test_event_ids)
    print(f"\n[Test Split] Master Partition Sizes: Train={len(manifest.train_event_ids)}, Val={len(manifest.val_event_ids)}, Test={len(test_event_ids)}")
    assert len(test_event_ids) == 1974, f"Expected 1,974 test events, got {len(test_event_ids)}"

    # 3. Load Raw Dataset Events
    raw_events = load_raw_dataset_events()
    test_events = {eid: raw_events[eid] for eid in manifest.test_event_ids if eid in raw_events}
    print(f"[Dataset Loaded] Filtered {len(test_events)} raw test events.")

    # 4. Prepare Output Directories
    out_pred_dir = Path("data/processed/predictions/phase3b/blind_test")
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    horizons = [2.0, 3.0, 5.0]
    all_metrics_rows = []
    all_summary_data = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "candidate_model": "M4",
        "candidate_description": "Categorical One-Hot (4 channels) + Covariance Log10 Scaling (37 channels, no dt)",
        "master_split": {
            "train_n": len(manifest.train_event_ids),
            "val_n": len(manifest.val_event_ids),
            "test_n": len(manifest.test_event_ids),
        },
        "horizons": {},
    }

    # 5. Evaluate Each Horizon
    for h in horizons:
        h_str = f"H{int(h)}"
        cutoff = h
        print(f"\n" + "=" * 70)
        print(f"BLIND TEST EVALUATION: HORIZON {h_str} (Cutoff = {cutoff:.1f} days)")
        print(f"=" * 70)

        # A. Load Frozen Preprocessor Manifest
        prep_manifest_path = f"artifacts/preprocessors/phase3b/preprocessor_M4_h{h:.1f}.json"
        print(f"  Loading Frozen Preprocessor: {prep_manifest_path}")
        prep = Phase3BSequencePreprocessor.load(prep_manifest_path)
        assert prep.is_fitted_, "Frozen preprocessor must be pre-fitted!"
        assert prep.config.n_channels == 37, f"Expected 37 channels, got {prep.config.n_channels}"

        # B. Prepare Raw Test Sequence Tensors
        X_test_raw, mask_test, y_test, test_ids = prep.prepare_sequence_tensors(test_events, horizon_cutoff=cutoff)
        n_test = len(y_test)
        print(f"  Qualifying Test Sequences ({h_str}): N = {n_test} (out of {len(test_event_ids)} total test events)")

        # Verify tensor shape
        X_test_np = X_test_raw.numpy() if hasattr(X_test_raw, "numpy") else np.array(X_test_raw)
        mask_test_np = mask_test.numpy() if hasattr(mask_test, "numpy") else np.array(mask_test)
        assert X_test_np.shape[1] == 37, f"Expected 37 input channels, got {X_test_np.shape[1]}"
        assert X_test_np.shape[2] == 23, f"Expected 23 max sequence length, got {X_test_np.shape[2]}"

        # C. Transform Sequence Tensors (STRICT TRANSFORM-ONLY MODE)
        X_test_norm = prep.transform(X_test_raw, mask_test)

        # D. Load Frozen M4 Model Checkpoint
        model_base = f"artifacts/models/phase3b/step3/tcn_best_M4_h{h:.1f}"
        print(f"  Loading Frozen Model Checkpoint: {model_base}.pt")
        model = TCNRiskModel.load(model_base)
        assert model.in_features == 37, f"Model in_features must be 37, got {model.in_features}"

        # E. Generate Blind Test Predictions
        test_preds = model.predict_risk(X_test_norm, mask_test)
        assert len(test_preds) == n_test, "Prediction count mismatch!"
        assert all(math.isfinite(p) for p in test_preds), "Found NaN or Inf in test predictions!"

        # F. Compute Sequence Lengths for Prediction Export
        seq_lens = [int(np.sum(mask_test_np[i])) for i in range(n_test)]

        # G. Save Blind Test Prediction CSV
        pred_csv_path = out_pred_dir / f"tcn_M4_h{h:.1f}_test_predictions.csv"
        with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "horizon", "final_risk", "predicted_risk", "sequence_length"])
            for eid, yt, yp, sl in zip(test_ids, y_test, test_preds, seq_lens):
                writer.writerow([eid, f"{h:.1f}", f"{yt:.6f}", f"{yp:.6f}", sl])
        print(f"  Saved Blind Test Predictions -> {pred_csv_path}")

        # H. Compute Metrics
        reg = compute_regression_metrics(y_test, test_preds)
        dist = compute_prediction_distribution(y_test, test_preds)
        rank = compute_ranking_metrics(y_test, test_preds, threshold_log10=-5.0)
        tail = compute_tail_analysis(y_test, test_preds, threshold_log10=-5.0)

        print(f"\n  [{h_str} Blind Test Results (M4)]")
        print(f"    Continuous Metrics:  MAE = {reg['mae']:.4f} | RMSE = {reg['rmse']:.4f} | R2 = {reg['r2']:.4f} | Spearman = {reg['spearman_correlation']:.4f}")
        print(f"    Distribution Stats:  Target Mean = {dist['target_mean']:.4f} | Pred Mean = {dist['prediction_mean']:.4f} | Bias = {dist['mean_bias']:+.4f} | Std Ratio = {dist['prediction_target_std_ratio']:.4f}")
        print(f"    Operational Alert:   Recall@1% = {rank['budget_pct_1']['recall']:.4f} | Recall@5% = {rank['budget_pct_5']['recall']:.4f} | Recall@10% = {rank['budget_pct_10']['recall']:.4f}")
        print(f"    Critical Events:     N Critical (y >= -5.0) = {tail['n_critical']} | Missed @ 10% = {rank['budget_pct_10']['missed_high_risk']}")
        print(f"    Tail Diagnostics:    Mean Residual = {tail['mean_residual']:+.4f} | Median Residual = {tail['median_residual']:+.4f}")

        row = {
            "horizon": h_str,
            "model": "M4",
            "description": "Frozen M4 Candidate (Categorical + Covariance Log10, 37 channels)",
            "n_channels": 37,
            "n_test_samples": n_test,
            "master_test_partition_n": len(test_event_ids),
            "test_mae": reg["mae"],
            "test_rmse": reg["rmse"],
            "test_r2": reg["r2"],
            "test_pearson": reg["pearson_correlation"],
            "test_spearman": reg["spearman_correlation"],
            "target_mean": dist["target_mean"],
            "target_std": dist["target_std"],
            "prediction_mean": dist["prediction_mean"],
            "prediction_std": dist["prediction_std"],
            "mean_bias": dist["mean_bias"],
            "prediction_target_std_ratio": dist["prediction_target_std_ratio"],
            "prediction_median": dist["prediction_median"],
            "n_critical_events": tail["n_critical"],
            "test_recall_top1pct": rank["budget_pct_1"]["recall"],
            "test_precision_top1pct": rank["budget_pct_1"]["precision"],
            "test_recall_top5pct": rank["budget_pct_5"]["recall"],
            "test_precision_top5pct": rank["budget_pct_5"]["precision"],
            "test_recall_top10pct": rank["budget_pct_10"]["recall"],
            "test_precision_top10pct": rank["budget_pct_10"]["precision"],
            "test_missed_events_top10pct": rank["budget_pct_10"]["missed_high_risk"],
            "tail_mean_residual": tail["mean_residual"],
            "tail_median_residual": tail["median_residual"],
        }
        all_metrics_rows.append(row)

        all_summary_data["horizons"][h_str] = {
            "horizon_days": h,
            "qualifying_test_n": n_test,
            "regression_metrics": reg,
            "distribution_statistics": dist,
            "operational_ranking": rank,
            "tail_diagnostics": tail,
        }

    # 6. Save Machine-Readable Outputs
    metrics_csv_path = "reports/phase3b_step4b_blind_test_metrics.csv"
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics_rows)
    print(f"\nSaved Blind Test Metrics CSV -> {metrics_csv_path}")

    summary_json_path = "reports/phase3b/step4b_blind_test_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_summary_data, f, indent=2)
    print(f"Saved Blind Test Diagnostics JSON -> {summary_json_path}")

    # 7. Independent Metric Recomputation Verification Routine
    print("\n======================================================================")
    print("[Verification] Independent Metric Recomputation from Stored CSV Files")
    print("======================================================================")
    recomp_errors = 0
    for row in all_metrics_rows:
        h_str = row["horizon"]
        h_val = 2.0 if h_str == "H2" else (3.0 if h_str == "H3" else 5.0)
        csv_file = out_pred_dir / f"tcn_M4_h{h_val:.1f}_test_predictions.csv"
        with open(csv_file, "r", encoding="utf-8") as f:
            disk_rows = list(csv.DictReader(f))

        y_t_disk = [float(r["final_risk"]) for r in disk_rows]
        y_p_disk = [float(r["predicted_risk"]) for r in disk_rows]

        reg_disk = compute_regression_metrics(y_t_disk, y_p_disk)
        rank_disk = compute_ranking_metrics(y_t_disk, y_p_disk, threshold_log10=-5.0)
        tail_disk = compute_tail_analysis(y_t_disk, y_p_disk, threshold_log10=-5.0)

        # Check tolerances
        diff_mae = abs(reg_disk["mae"] - row["test_mae"])
        diff_r2 = abs(reg_disk["r2"] - row["test_r2"])
        diff_rec = abs(rank_disk["budget_pct_10"]["recall"] - row["test_recall_top10pct"])
        diff_tail = abs(tail_disk["mean_residual"] - row["tail_mean_residual"])

        if diff_mae > 1e-5 or diff_r2 > 1e-5 or diff_rec > 1e-5 or diff_tail > 1e-5:
            print(f"  [RECOMP ERROR] Discrepancy on {h_str}: MAE diff={diff_mae}, R2 diff={diff_r2}")
            recomp_errors += 1
        else:
            print(f"  [EXACT MATCH] {h_str}: MAE={reg_disk['mae']:.5f}, R2={reg_disk['r2']:.5f}, Recall@10%={rank_disk['budget_pct_10']['recall']:.5f}")

    if recomp_errors == 0:
        print("[Verification Passed] Independent recomputation verified with 100% precision (0 errors).")
    else:
        raise RuntimeError(f"Found {recomp_errors} discrepancies during metric recomputation!")

    # 8. Post-Evaluation Cryptographic Gate
    verify_cryptographic_hashes(tag="Post-Evaluation Gate")

    elapsed = time.time() - t_start
    print(f"\nBlind Test Evaluation of Frozen M4 completed successfully in {elapsed:.2f}s.")


if __name__ == "__main__":
    run_blind_test_evaluation()
