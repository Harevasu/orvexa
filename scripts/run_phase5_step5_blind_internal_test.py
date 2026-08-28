"""ORVEXA Phase 5 Step 5: Final Blind Internal-Test Evaluation Runner.

Evaluates the formally frozen candidate:
Candidate C: Quantile M4 Causal TCN + Conformalized Quantile Regression (CQR)

Strict Governance & Quarantine:
- First and only authorized evaluation on Phase 5 Internal Test (1,677 events, indices [9503, 11179]).
- ZERO retraining, fine-tuning, threshold tuning, or calibration refitting.
- Historical Master Test (1,974 events, indices [11180, 13153]) remains permanently quarantined.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.bootstrap import compute_event_bootstrap_ci
from orvexa.conformal import SplitConformalCalibrator
from orvexa.losses_quantile import (
    compute_quantile_evaluation_metrics,
    multi_quantile_pinball_loss_numpy,
)
from orvexa.models_probabilistic import (
    DEFAULT_QUANTILES,
    QuantileTCNRiskModel,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import Phase5SplitManifest


def compute_file_sha256(file_path: str) -> str:
    """Deterministically compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_events_by_id(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load raw dataset grouped by event_id, sorted oldest to newest (time_to_tca descending)."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])

    events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    records = df.to_dict(orient="records")
    for r in records:
        ev_id = str(r["event_id"])
        events[ev_id].append(r)
    return dict(events)


# Metric wrapper callables for bootstrap
def metric_mae(yt: List[float], yp: List[float]) -> float:
    return float(np.mean(np.abs(np.array(yt) - np.array(yp))))


def metric_rmse(yt: List[float], yp: List[float]) -> float:
    return float(np.sqrt(np.mean((np.array(yt) - np.array(yp)) ** 2)))


def metric_r2(yt: List[float], yp: List[float]) -> float:
    yt_a = np.array(yt)
    yp_a = np.array(yp)
    ss_tot = np.sum((yt_a - np.mean(yt_a)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    ss_res = np.sum((yt_a - yp_a) ** 2)
    return float(1.0 - (ss_res / ss_tot))


def metric_spearman(yt: List[float], yp: List[float]) -> float:
    from orvexa.regression_metrics import _spearman_rank_correlation
    return float(_spearman_rank_correlation(yt, yp))


def run_phase5_step5_blind_internal_test() -> Dict[str, Any]:
    print("=" * 80)
    print("ORVEXA PHASE 5 STEP 5: FINAL BLIND INTERNAL-TEST EVALUATION")
    print("=" * 80)

    # 1. Verification Gate: Split Manifest & Candidate Freeze Manifest
    split_manifest_path = "artifacts/splits/phase5/phase5_split_manifest.json"
    freeze_manifest_path = "artifacts/models/phase5/candidate_freeze_manifest.json"

    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Missing Phase 5 split manifest: {split_manifest_path}")
    if not os.path.exists(freeze_manifest_path):
        raise FileNotFoundError(f"Missing Candidate freeze manifest: {freeze_manifest_path}")

    # Verify Split Manifest Hash
    act_split_hash = compute_file_sha256(split_manifest_path)
    exp_split_hash = "c304cb88c48b1f2c066c65a69aa947fb5902f8892633b0244f6746d1c37b15d2"
    if act_split_hash != exp_split_hash:
        raise ValueError(f"Split manifest hash corrupted! Expected {exp_split_hash}, got {act_split_hash}")

    split_manifest = Phase5SplitManifest.load(split_manifest_path)
    print(f"Loaded and verified Phase 5 Split Manifest:")
    print(f"  Internal Test Events: {len(split_manifest.test_event_ids):5d} (AUTHORIZED FOR EVALUATION)")
    print(f"  Historical Test:      {len(split_manifest.quarantined_test_event_ids):5d} (PERMANENTLY QUARANTINED)")

    # Load Freeze Manifest
    with open(freeze_manifest_path, "r", encoding="utf-8") as f:
        freeze_data = json.load(f)

    print(f"\nVerifying frozen candidate integrity across all horizons...")
    for h_key, h_artifacts in freeze_data["frozen_artifacts"].items():
        for key_prefix in ["model_config", "model_weights", "cqr_calibrator", "preprocessor"]:
            rel_path = h_artifacts[key_prefix]
            exp_hash = h_artifacts[f"{key_prefix}_sha256"]
            act_hash = compute_file_sha256(rel_path)
            if act_hash != exp_hash:
                raise ValueError(f"Frozen artifact integrity failure on {rel_path}! Expected {exp_hash}, got {act_hash}")
            print(f"  Verified {h_key} {key_prefix}: {rel_path} ({act_hash[:16]}...)")

    # Load raw events
    raw_csv = "data/raw/esa/train_data.csv"
    all_events = load_raw_events_by_id(raw_csv)
    test_events = {ev: all_events[ev] for ev in split_manifest.test_event_ids if ev in all_events}

    os.makedirs("data/processed/predictions/phase5/blind_test", exist_ok=True)
    os.makedirs("reports/phase5", exist_ok=True)

    horizons = [2.0, 3.0, 5.0, 6.0]
    test_results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for h in horizons:
        h_key = f"H{int(h)}" if h.is_integer() else f"H{h}"
        h_num_str = str(int(h)) if h.is_integer() else str(h)
        print(f"\n{'='*70}")
        print(f"EVALUATING HORIZON {h_key} ON PHASE 5 INTERNAL TEST (Cutoff: {h:.1f} days / {h*24:.0f} hours)")
        print(f"{'='*70}")

        # 1. Load Frozen Preprocessor (NO REFITTING)
        preproc_path = f"artifacts/preprocessors/phase5/preprocessor_M4_h{h}.json"
        preprocessor = Phase3BSequencePreprocessor.load(preproc_path)

        # Prepare Internal Test Sequence Tensors
        X_test_raw, mask_test, y_test, test_ids = preprocessor.prepare_sequence_tensors(
            test_events, horizon_cutoff=h
        )
        X_test_norm = preprocessor.transform(X_test_raw, mask_test)

        print(f"  [Tensors] Internal Test qualifying events: N = {len(test_ids)}")

        # 2. Load Frozen Quantile Model (NO RETRAINING)
        model_path = f"artifacts/models/phase5/tcn_quantile_M4_h{h_num_str}"
        model = QuantileTCNRiskModel.load(model_path)
        quant_test_preds = model.predict_quantiles(X_test_norm, mask_test)

        # 3. Load Frozen CQR Calibrator (NO REFITTING)
        cqr_calibrator_path = f"artifacts/models/phase5/cqr_calibrator_h{h}.json"
        cqr_calibrator = SplitConformalCalibrator.load(cqr_calibrator_path)

        idx_05 = DEFAULT_QUANTILES.index(0.05)
        idx_50 = DEFAULT_QUANTILES.index(0.50)
        idx_95 = DEFAULT_QUANTILES.index(0.95)

        # Extract Median Point Prediction
        q50_test_preds = [float(x) for x in quant_test_preds[:, idx_50]]

        # Generate CQR Prediction Intervals
        cqr_lower, cqr_upper = cqr_calibrator.predict_intervals(
            (quant_test_preds[:, idx_05], quant_test_preds[:, idx_95]), alpha=0.10
        )
        cqr_covered = (np.array(y_test) >= cqr_lower) & (np.array(y_test) <= cqr_upper)
        cqr_widths = cqr_upper - cqr_lower

        # Save Prediction CSV
        pred_csv_path = f"data/processed/predictions/phase5/blind_test/tcn_quantile_M4_cqr_h{h_num_str}_internal_test_predictions.csv"
        with open(pred_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "event_id",
                "y_true",
                "q_05",
                "q_10",
                "q_25",
                "q_50",
                "q_75",
                "q_90",
                "q_95",
                "cqr_lower_90",
                "cqr_upper_90",
                "cqr_covered_90",
                "cqr_width_90",
            ]
            writer.writerow(header)
            for i in range(len(test_ids)):
                row = [
                    test_ids[i],
                    round(float(y_test[i]), 6),
                    round(float(quant_test_preds[i, 0]), 6),
                    round(float(quant_test_preds[i, 1]), 6),
                    round(float(quant_test_preds[i, 2]), 6),
                    round(float(quant_test_preds[i, 3]), 6),
                    round(float(quant_test_preds[i, 4]), 6),
                    round(float(quant_test_preds[i, 5]), 6),
                    round(float(quant_test_preds[i, 6]), 6),
                    round(float(cqr_lower[i]), 6),
                    round(float(cqr_upper[i]), 6),
                    int(cqr_covered[i]),
                    round(float(cqr_widths[i]), 6),
                ]
                writer.writerow(row)

        print(f"  [Predictions] Saved {len(test_ids)} test predictions to: {pred_csv_path}")

        # 4. Central Regression Evaluation & Bootstrap CIs (1,000 iterations)
        mae_ci = compute_event_bootstrap_ci(test_ids, y_test, q50_test_preds, metric_mae, n_iterations=1000, seed=42)
        rmse_ci = compute_event_bootstrap_ci(test_ids, y_test, q50_test_preds, metric_rmse, n_iterations=1000, seed=42)
        r2_ci = compute_event_bootstrap_ci(test_ids, y_test, q50_test_preds, metric_r2, n_iterations=1000, seed=42)
        spear_ci = compute_event_bootstrap_ci(test_ids, y_test, q50_test_preds, metric_spearman, n_iterations=1000, seed=42)

        reg_metrics = compute_regression_metrics(y_test, q50_test_preds)

        # 5. Multi-Quantile Evaluation
        quant_metrics = compute_quantile_evaluation_metrics(y_test, quant_test_preds, DEFAULT_QUANTILES)

        # 6. CQR Evaluation
        cqr_eval = cqr_calibrator.evaluate_coverage(
            y_test, (quant_test_preds[:, idx_05], quant_test_preds[:, idx_95]), alpha=0.10
        )

        # 7. Operational Ranking / Alert Budget Evaluation
        rank_metrics = compute_ranking_metrics(y_test, q50_test_preds, threshold_log10=-5.0)

        # 8. Critical Tail Diagnostics (y >= -5.0)
        y_test_np = np.array(y_test)
        crit_mask = y_test_np >= -5.0
        n_crit = int(np.sum(crit_mask))
        tail_diag: Dict[str, Any] = {"critical_events_count": n_crit}
        if n_crit > 0:
            y_crit = y_test_np[crit_mask]
            q50_crit = np.array(q50_test_preds)[crit_mask]
            res_crit = q50_crit - y_crit
            tail_diag["tail_mean_residual"] = round(float(np.mean(res_crit)), 5)
            tail_diag["tail_median_residual"] = round(float(np.median(res_crit)), 5)
            tail_diag["tail_mae"] = round(float(np.mean(np.abs(res_crit))), 5)
            tail_diag["tail_rmse"] = round(float(np.sqrt(np.mean(res_crit ** 2))), 5)
            tail_diag["cqr_tail_coverage"] = round(float(np.mean(cqr_covered[crit_mask])), 5)
            tail_diag["cqr_tail_covered_count"] = int(np.sum(cqr_covered[crit_mask]))

        # 9. Generalization Delta: Test vs Validation
        val_audit_h = freeze_data["audit_validation_results"][h_key]
        val_r2 = val_audit_h["candidate_b_quantile_raw"]["q50_r2"]["point_estimate"]
        val_mae = val_audit_h["candidate_b_quantile_raw"]["q50_mae"]["point_estimate"]
        val_rmse = val_audit_h["candidate_b_quantile_raw"]["q50_rmse"]["point_estimate"]
        val_spearman = val_audit_h["candidate_b_quantile_raw"]["q50_spearman_rho"]["point_estimate"]
        val_cqr_cov = val_audit_h["candidate_c_cqr"]["empirical_coverage"]
        val_cqr_width = val_audit_h["candidate_c_cqr"]["mean_interval_width"]

        delta_gen = {
            "delta_r2": round(r2_ci["point_estimate"] - val_r2, 5),
            "delta_mae": round(mae_ci["point_estimate"] - val_mae, 5),
            "delta_rmse": round(rmse_ci["point_estimate"] - val_rmse, 5),
            "delta_spearman": round(spear_ci["point_estimate"] - val_spearman, 5),
            "delta_cqr_coverage": round(cqr_eval["empirical_coverage"] - val_cqr_cov, 5),
            "delta_cqr_width": round(cqr_eval["mean_interval_width"] - val_cqr_width, 5),
        }

        print(f"  [Test Results] N = {len(test_ids)}:")
        print(f"    Point Prediction q0.50: MAE = {mae_ci['point_estimate']:.5f} | RMSE = {rmse_ci['point_estimate']:.5f} | R2 = {r2_ci['point_estimate']:+.5f} [{r2_ci['ci_lower']:+.5f}, {r2_ci['ci_upper']:+.5f}] | Spear = {spear_ci['point_estimate']:.5f}")
        print(f"    Pinball Loss:           Mean = {quant_metrics['mean_pinball_loss']:.5f} | Crossing Violations = {quant_metrics['quantile_crossing_violations']}")
        print(f"    CQR 90% Conformal:      Coverage = {cqr_eval['empirical_coverage']*100:.2f}% | Mean Width = {cqr_eval['mean_interval_width']:.2f} | Median Width = {cqr_eval['median_interval_width']:.2f}")
        print(f"    Tail Risk (y >= -5.0):  N_crit = {n_crit} | CQR Tail Cov = {tail_diag.get('cqr_tail_coverage', 'N/A')} ({tail_diag.get('cqr_tail_covered_count', 0)}/{n_crit})")
        print(f"    Generalization Delta:   Delta R2 = {delta_gen['delta_r2']:+.5f} | Delta Cov = {delta_gen['delta_cqr_coverage']*100:+.2f}% | Delta Width = {delta_gen['delta_cqr_width']:+.2f}")

        test_results[h_key] = {
            "horizon_days": h,
            "lead_time_hours": h * 24.0,
            "n_test_samples": len(test_ids),
            "prediction_csv": pred_csv_path,
            "prediction_csv_sha256": compute_file_sha256(pred_csv_path),
            "point_prediction_q50": {
                "mae": mae_ci,
                "rmse": rmse_ci,
                "r2": r2_ci,
                "spearman_rho": spear_ci,
                "regression_summary": reg_metrics,
            },
            "quantile_metrics": quant_metrics,
            "cqr_evaluation": cqr_eval,
            "operational_ranking": rank_metrics,
            "tail_diagnostics": tail_diag,
            "generalization_delta": delta_gen,
        }

        csv_rows.append({
            "horizon": h_key,
            "horizon_days": h,
            "n_test_samples": len(test_ids),
            "q50_mae": mae_ci["point_estimate"],
            "q50_mae_ci_lower": mae_ci["ci_lower"],
            "q50_mae_ci_upper": mae_ci["ci_upper"],
            "q50_rmse": rmse_ci["point_estimate"],
            "q50_rmse_ci_lower": rmse_ci["ci_lower"],
            "q50_rmse_ci_upper": rmse_ci["ci_upper"],
            "q50_r2": r2_ci["point_estimate"],
            "q50_r2_ci_lower": r2_ci["ci_lower"],
            "q50_r2_ci_upper": r2_ci["ci_upper"],
            "q50_spearman_rho": spear_ci["point_estimate"],
            "q50_spearman_ci_lower": spear_ci["ci_lower"],
            "q50_spearman_ci_upper": spear_ci["ci_upper"],
            "mean_pinball_loss": quant_metrics["mean_pinball_loss"],
            "quantile_crossing_violations": quant_metrics["quantile_crossing_violations"],
            "raw_90pct_coverage": quant_metrics["intervals"]["90pct_interval"]["empirical_coverage"],
            "raw_90pct_mean_width": quant_metrics["intervals"]["90pct_interval"]["mean_width"],
            "cqr_shift_qhat": cqr_eval["conformal_quantile_q_hat"],
            "cqr_90pct_coverage": cqr_eval["empirical_coverage"],
            "cqr_mean_width": cqr_eval["mean_interval_width"],
            "cqr_median_width": cqr_eval["median_interval_width"],
            "critical_events_count": n_crit,
            "cqr_tail_coverage": tail_diag.get("cqr_tail_coverage", ""),
            "val_r2": val_r2,
            "delta_r2": delta_gen["delta_r2"],
            "val_cqr_coverage": val_cqr_cov,
            "delta_cqr_coverage": delta_gen["delta_cqr_coverage"],
        })

    # Save summary JSON
    summary_path = "reports/phase5/step5_blind_internal_test_summary.json"
    summary_data = {
        "report_type": "ORVEXA_PHASE5_STEP5_BLIND_INTERNAL_TEST_SUMMARY",
        "phase": "PHASE_5",
        "step": "STEP_5",
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate_id": freeze_data["selected_candidate"]["candidate_id"],
        "candidate_name": freeze_data["selected_candidate"]["candidate_name"],
        "split_manifest_path": split_manifest_path,
        "split_manifest_sha256": act_split_hash,
        "freeze_manifest_path": freeze_manifest_path,
        "freeze_manifest_sha256": compute_file_sha256(freeze_manifest_path),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "horizons": test_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save Metrics CSV
    metrics_csv_path = "reports/phase5_step5_blind_internal_test_metrics.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"\n{'='*80}")
    print(f"Evaluation Complete. Artifacts saved:")
    print(f"  Summary JSON: {summary_path} (SHA-256: {compute_file_sha256(summary_path)})")
    print(f"  Metrics CSV:  {metrics_csv_path} (SHA-256: {compute_file_sha256(metrics_csv_path)})")
    print(f"{'='*80}")

    return summary_data


if __name__ == "__main__":
    run_phase5_step5_blind_internal_test()
