"""ORVEXA Phase 5 Step 3: Probabilistic Modeling & Conformal Uncertainty Infrastructure Runner.

Executes:
1. Fresh deterministic M4 reference training on Phase 5 Train (6,708 events), evaluated on Phase 5 Validation (1,677 events).
2. Multi-quantile M4 training with Pinball loss and monotonic non-crossing parameterization on Phase 5 Train, evaluated on Phase 5 Validation.
3. Split Conformal Calibration on Phase 5 Calibration (1,118 events), evaluated on Phase 5 Validation.
4. Comprehensive multi-horizon evaluation across H2, H3, H5, and H6.

Strict Quarantine Enforcement:
- Historical Test (1,974 events) is NEVER loaded or evaluated.
- Phase 5 Internal Test (1,677 events) is NEVER loaded or evaluated.
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
import pandas as pd

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.conformal import SplitConformalCalibrator
from orvexa.losses_quantile import (
    compute_quantile_evaluation_metrics,
    multi_quantile_pinball_loss_numpy,
)
from orvexa.models_probabilistic import (
    DEFAULT_QUANTILES,
    QuantileTCNRiskModel,
)
from orvexa.models_tcn import TCNRiskModel
from orvexa.phase3b_config import (
    ExperimentID,
    Phase3BExperimentConfig,
    get_experiment_config,
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
    # Sort by event_id and time_to_tca descending (chronological order of CDM reception)
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])

    events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    records = df.to_dict(orient="records")
    for r in records:
        ev_id = str(r["event_id"])
        events[ev_id].append(r)
    return dict(events)


def run_phase5_step3_pipeline() -> Dict[str, Any]:
    print("=" * 80)
    print("ORVEXA PHASE 5 STEP 3: PROBABILISTIC MODELING & CONFORMAL INFRASTRUCTURE")
    print("=" * 80)

    # 1. Verify split manifest & quarantine
    split_manifest_path = "artifacts/splits/phase5/phase5_split_manifest.json"
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Missing Phase 5 split manifest: {split_manifest_path}")

    split_manifest = Phase5SplitManifest.load(split_manifest_path)
    print(f"Loaded Phase 5 Split Manifest:")
    print(f"  Train Events:       {len(split_manifest.train_event_ids):5d}")
    print(f"  Validation Events:  {len(split_manifest.val_event_ids):5d}")
    print(f"  Calibration Events: {len(split_manifest.cal_event_ids):5d}")
    print(f"  Internal Test:      {len(split_manifest.test_event_ids):5d} (QUARANTINED)")

    # Load raw events
    raw_csv = "data/raw/esa/train_data.csv"
    print(f"\nLoading canonical dataset: {raw_csv} ...")
    all_events = load_raw_events_by_id(raw_csv)
    print(f"Loaded {len(all_events)} total unique events.")

    # Partition events for pipeline (Test partition is NOT processed)
    train_events = {ev: all_events[ev] for ev in split_manifest.train_event_ids if ev in all_events}
    val_events = {ev: all_events[ev] for ev in split_manifest.val_event_ids if ev in all_events}
    cal_events = {ev: all_events[ev] for ev in split_manifest.cal_event_ids if ev in all_events}

    # Ensure output directories exist
    os.makedirs("artifacts/models/phase5", exist_ok=True)
    os.makedirs("artifacts/preprocessors/phase5", exist_ok=True)
    os.makedirs("data/processed/predictions/phase5", exist_ok=True)
    os.makedirs("reports/phase5", exist_ok=True)

    m4_config = get_experiment_config(ExperimentID.M4)
    horizons = [2.0, 3.0, 5.0, 6.0]
    results_by_horizon: Dict[str, Any] = {}

    for h in horizons:
        h_key = f"H{int(h)}" if h.is_integer() else f"H{h}"
        print(f"\n{'='*70}")
        print(f"EXECUTING HORIZON {h_key} (Cutoff: {h:.1f} days / {h*24:.0f} hours)")
        print(f"{'='*70}")

        # 1. Fit Preprocessor strictly on Phase 5 Train events
        preprocessor = Phase3BSequencePreprocessor(config=m4_config)
        X_train_raw, mask_train, y_train, train_ids = preprocessor.prepare_sequence_tensors(
            train_events, horizon_cutoff=h
        )
        preprocessor.fit(X_train_raw, mask_train)
        X_train_norm = preprocessor.transform(X_train_raw, mask_train)

        # Save preprocessor artifact
        preproc_path = f"artifacts/preprocessors/phase5/preprocessor_M4_h{h}.json"
        preprocessor.save(preproc_path)
        print(f"  [Preproc] Fitted on Train (N={len(train_ids)} events). Saved to {preproc_path}")

        # 2. Transform Validation and Calibration tensors
        X_val_raw, mask_val, y_val, val_ids = preprocessor.prepare_sequence_tensors(
            val_events, horizon_cutoff=h
        )
        X_val_norm = preprocessor.transform(X_val_raw, mask_val)

        X_cal_raw, mask_cal, y_cal, cal_ids = preprocessor.prepare_sequence_tensors(
            cal_events, horizon_cutoff=h
        )
        X_cal_norm = preprocessor.transform(X_cal_raw, mask_cal)

        print(f"  [Tensors] Qualifying counts: Train={len(train_ids)}, Val={len(val_ids)}, Cal={len(cal_ids)}")

        # ---------------------------------------------------------
        # 3. Train Fresh Deterministic M4 Reference
        # ---------------------------------------------------------
        print(f"\n  [1/3] Training Fresh Deterministic M4 Reference...")
        t0 = time.time()
        det_model = TCNRiskModel(
            in_features=preprocessor.config.n_channels,
            channels=preprocessor.config.channels,
            kernel_size=preprocessor.config.kernel_size,
            dilations=preprocessor.config.dilations,
            dropout=preprocessor.config.dropout,
            learning_rate=preprocessor.config.learning_rate,
            weight_decay=preprocessor.config.weight_decay,
            batch_size=preprocessor.config.batch_size,
            max_seq_len=preprocessor.config.max_seq_len,
            seed=42,
        )
        det_model.fit(
            X_train=X_train_norm,
            mask_train=mask_train,
            y_train=y_train,
            X_val=X_val_norm,
            mask_val=mask_val,
            y_val=y_val,
            epochs=preprocessor.config.epochs,
            patience=preprocessor.config.patience,
            verbose=False,
        )
        det_time = time.time() - t0

        # Save deterministic model
        det_model_path = f"artifacts/models/phase5/tcn_deterministic_M4_h{h}"
        det_model.save(det_model_path)

        # Predict on Validation
        det_val_preds = det_model.predict_risk(X_val_norm, mask_val)
        det_val_reg = compute_regression_metrics(y_val, det_val_preds)
        det_val_rank = compute_ranking_metrics(y_val, det_val_preds)

        # Predict on Calibration (for conformal calibration)
        det_cal_preds = det_model.predict_risk(X_cal_norm, mask_cal)

        # Save deterministic validation predictions
        det_pred_csv = f"data/processed/predictions/phase5/tcn_deterministic_M4_h{h}_val_predictions.csv"
        with open(det_pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "y_true", "y_pred_deterministic"])
            for ev, yt, yp in zip(val_ids, y_val, det_val_preds):
                writer.writerow([ev, round(yt, 6), round(yp, 6)])

        print(f"    Deterministic M4 Val Results ({det_time:.1f}s, Best Ep {det_model.training_stats_['best_epoch']}):")
        print(f"      MAE: {det_val_reg['mae']:.5f} | RMSE: {det_val_reg['rmse']:.5f} | R2: {det_val_reg['r2']:+.5f} | Spearman: {det_val_reg['spearman_correlation']:.5f}")

        # ---------------------------------------------------------
        # 4. Train Quantile M4 Model
        # ---------------------------------------------------------
        print(f"\n  [2/3] Training Quantile M4 Model (Non-crossing parameterization)...")
        t0 = time.time()
        quant_model = QuantileTCNRiskModel(
            in_features=preprocessor.config.n_channels,
            quantiles=DEFAULT_QUANTILES,
            channels=preprocessor.config.channels,
            kernel_size=preprocessor.config.kernel_size,
            dilations=preprocessor.config.dilations,
            dropout=preprocessor.config.dropout,
            learning_rate=preprocessor.config.learning_rate,
            weight_decay=preprocessor.config.weight_decay,
            batch_size=preprocessor.config.batch_size,
            max_seq_len=preprocessor.config.max_seq_len,
            enforce_monotonic=True,
            seed=42,
        )
        quant_model.fit(
            X_train=X_train_norm,
            mask_train=mask_train,
            y_train=y_train,
            X_val=X_val_norm,
            mask_val=mask_val,
            y_val=y_val,
            epochs=preprocessor.config.epochs,
            patience=preprocessor.config.patience,
            verbose=False,
        )
        quant_time = time.time() - t0

        # Save quantile model
        quant_model_path = f"artifacts/models/phase5/tcn_quantile_M4_h{h}"
        quant_model.save(quant_model_path)

        # Predict quantiles on Validation
        quant_val_preds = quant_model.predict_quantiles(X_val_norm, mask_val)
        quant_val_metrics = compute_quantile_evaluation_metrics(
            y_val, quant_val_preds, DEFAULT_QUANTILES
        )

        # Continuous point evaluation using median quantile (tau=0.50)
        idx_med = DEFAULT_QUANTILES.index(0.50)
        q50_val_preds = [float(x) for x in quant_val_preds[:, idx_med]]
        q50_val_reg = compute_regression_metrics(y_val, q50_val_preds)
        q50_val_rank = compute_ranking_metrics(y_val, q50_val_preds)

        # Predict quantiles on Calibration
        quant_cal_preds = quant_model.predict_quantiles(X_cal_norm, mask_cal)

        # Save quantile validation predictions
        quant_pred_csv = f"data/processed/predictions/phase5/tcn_quantile_M4_h{h}_val_predictions.csv"
        with open(quant_pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["event_id", "y_true"] + [f"q_{int(q*100):02d}" for q in DEFAULT_QUANTILES]
            writer.writerow(header)
            for ev, yt, qp_row in zip(val_ids, y_val, quant_val_preds):
                writer.writerow([ev, round(yt, 6)] + [round(float(v), 6) for v in qp_row])

        print(f"    Quantile M4 Val Results ({quant_time:.1f}s, Best Ep {quant_model.training_stats_['best_epoch']}):")
        print(f"      Mean Pinball Loss: {quant_val_metrics['mean_pinball_loss']:.5f}")
        print(f"      Crossing Rate:     {quant_val_metrics['quantile_crossing_rate']:.6f} (Violations: {quant_val_metrics['quantile_crossing_violations']})")
        print(f"      90% Int Coverage:  {quant_val_metrics['intervals']['90pct_interval']['empirical_coverage']:.4f} (Mean Width: {quant_val_metrics['intervals']['90pct_interval']['mean_width']:.2f})")
        print(f"      80% Int Coverage:  {quant_val_metrics['intervals']['80pct_interval']['empirical_coverage']:.4f} (Mean Width: {quant_val_metrics['intervals']['80pct_interval']['mean_width']:.2f})")
        print(f"      50% Int Coverage:  {quant_val_metrics['intervals']['50pct_interval']['empirical_coverage']:.4f} (Mean Width: {quant_val_metrics['intervals']['50pct_interval']['mean_width']:.2f})")
        print(f"      q50 Point R2:      {q50_val_reg['r2']:+.5f} | Spearman: {q50_val_reg['spearman_correlation']:.5f}")

        # ---------------------------------------------------------
        # 5. Split Conformal Calibration
        # ---------------------------------------------------------
        print(f"\n  [3/3] Executing Inductive Split Conformal Calibration on Calibration Partition...")
        # Standard split conformal on deterministic point predictions
        conformal_calibrator = SplitConformalCalibrator(default_alpha=0.10, score_type="absolute_residual")
        conformal_calibrator.fit(y_cal, det_cal_preds)
        conformal_calibrator_path = f"artifacts/models/phase5/conformal_calibrator_h{h}.json"
        conformal_calibrator.save(conformal_calibrator_path)

        # Evaluate conformal intervals on Validation
        conf_val_90 = conformal_calibrator.evaluate_coverage(y_val, det_val_preds, alpha=0.10)
        conf_val_95 = conformal_calibrator.evaluate_coverage(y_val, det_val_preds, alpha=0.05)

        # Conformalized Quantile Regression (CQR) on Quantile bounds [q0.05, q0.95]
        q_low_idx = DEFAULT_QUANTILES.index(0.05)
        q_high_idx = DEFAULT_QUANTILES.index(0.95)
        cqr_calibrator = SplitConformalCalibrator(default_alpha=0.10, score_type="cqr")
        cqr_calibrator.fit(y_cal, (quant_cal_preds[:, q_low_idx], quant_cal_preds[:, q_high_idx]))
        cqr_calibrator_path = f"artifacts/models/phase5/cqr_calibrator_h{h}.json"
        cqr_calibrator.save(cqr_calibrator_path)

        cqr_val_90 = cqr_calibrator.evaluate_coverage(
            y_val, (quant_val_preds[:, q_low_idx], quant_val_preds[:, q_high_idx]), alpha=0.10
        )

        print(f"    Split Conformal Validation Coverage (Residual Score, N_cal={len(cal_ids)}):")
        print(f"      90% Nominal (a=0.10): Empirical Cov = {conf_val_90['empirical_coverage']:.4f} | Width = {conf_val_90['mean_interval_width']:.2f} (q_hat={conf_val_90['conformal_quantile_q_hat']:.3f})")
        print(f"      95% Nominal (a=0.05): Empirical Cov = {conf_val_95['empirical_coverage']:.4f} | Width = {conf_val_95['mean_interval_width']:.2f} (q_hat={conf_val_95['conformal_quantile_q_hat']:.3f})")
        print(f"    CQR Conformalized Quantile Validation Coverage (a=0.10):")
        print(f"      Empirical Cov = {cqr_val_90['empirical_coverage']:.4f} | Width = {cqr_val_90['mean_interval_width']:.2f} (q_hat={cqr_val_90['conformal_quantile_q_hat']:+.3f})")

        # Compile horizon result record
        results_by_horizon[h_key] = {
            "horizon_days": h,
            "lead_time_hours": h * 24.0,
            "n_train_samples": len(train_ids),
            "n_val_samples": len(val_ids),
            "n_cal_samples": len(cal_ids),
            "deterministic_m4": {
                "training_time_sec": round(det_time, 2),
                "best_epoch": det_model.training_stats_["best_epoch"],
                "validation_regression": det_val_reg,
                "validation_ranking": det_val_rank,
            },
            "quantile_m4": {
                "training_time_sec": round(quant_time, 2),
                "best_epoch": quant_model.training_stats_["best_epoch"],
                "validation_quantile_metrics": quant_val_metrics,
                "q50_validation_regression": q50_val_reg,
                "q50_validation_ranking": q50_val_rank,
            },
            "split_conformal_residual": {
                "val_coverage_90pct": conf_val_90,
                "val_coverage_95pct": conf_val_95,
            },
            "cqr_conformal_quantile": {
                "val_coverage_90pct": cqr_val_90,
            },
        }

    # Save summary report JSON
    summary_path = "reports/phase5/step3_probabilistic_metrics.json"
    summary_data = {
        "step": "PHASE_5_STEP3",
        "objective": "Probabilistic Modeling & Conformal Uncertainty Infrastructure",
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_manifest_path": split_manifest_path,
        "split_manifest_sha256": compute_file_sha256(split_manifest_path),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "horizons": results_by_horizon,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Summary metrics written to: {summary_path}")
    print(f"{'='*80}")
    return summary_data


if __name__ == "__main__":
    run_phase5_step3_pipeline()
