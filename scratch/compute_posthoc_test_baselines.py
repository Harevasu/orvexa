"""Post-hoc comparison script to evaluate frozen M0 (Phase 2B) and frozen M2 (Step 2) on the test split."""

import csv
import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np

# Ensure src is discoverable
sys.path.insert(0, os.path.abspath("src"))

from orvexa.models_tcn import TCNRiskModel
from orvexa.phase3b_config import ExperimentID, get_experiment_config
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.ranking_metrics import compute_ranking_metrics
from orvexa.regression_metrics import compute_regression_metrics
from orvexa.splitting import SplitManifest


def main():
    manifest = SplitManifest.load("artifacts/splits/master_split_manifest.json")
    raw_path = "data/raw/esa/train_data.csv"
    from collections import defaultdict
    raw_events = defaultdict(list)
    with open(raw_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw_events[r["event_id"]].append(r)

    test_events = {eid: raw_events[eid] for eid in manifest.test_event_ids if eid in raw_events}

    horizons = [2.0, 3.0, 5.0]
    results = {}

    for h in horizons:
        h_str = f"H{int(h)}"
        # Evaluate M0
        prep_m0 = Phase3BSequencePreprocessor.load(f"artifacts/preprocessors/phase3b/preprocessor_M0_h{h:.1f}.json")
        X_m0, mask_m0, y_test, test_ids = prep_m0.prepare_sequence_tensors(test_events, horizon_cutoff=h)
        X_m0_norm = prep_m0.transform(X_m0, mask_m0)
        model_m0 = TCNRiskModel.load(f"artifacts/models/tcn_best_h{h:.1f}")
        preds_m0 = model_m0.predict_risk(X_m0_norm, mask_m0)
        reg_m0 = compute_regression_metrics(y_test, preds_m0)
        rank_m0 = compute_ranking_metrics(y_test, preds_m0, threshold_log10=-5.0)

        # Evaluate M2
        prep_m2 = Phase3BSequencePreprocessor.load(f"artifacts/preprocessors/phase3b/preprocessor_M2_h{h:.1f}.json")
        X_m2, mask_m2, _, _ = prep_m2.prepare_sequence_tensors(test_events, horizon_cutoff=h)
        X_m2_norm = prep_m2.transform(X_m2, mask_m2)
        model_m2 = TCNRiskModel.load(f"artifacts/models/phase3b/tcn_best_M2_h{h:.1f}")
        preds_m2 = model_m2.predict_risk(X_m2_norm, mask_m2)
        reg_m2 = compute_regression_metrics(y_test, preds_m2)
        rank_m2 = compute_ranking_metrics(y_test, preds_m2, threshold_log10=-5.0)

        results[f"{h_str}_M0"] = {"reg": reg_m0, "rank": rank_m0}
        results[f"{h_str}_M2"] = {"reg": reg_m2, "rank": rank_m2}

        print(f"\n[{h_str} Post-Hoc Test Baselines]")
        print(f"  M0: MAE={reg_m0['mae']:.4f}, RMSE={reg_m0['rmse']:.4f}, R2={reg_m0['r2']:.4f}, Recall@10%={rank_m0['budget_pct_10']['recall']:.4f} (Missed: {rank_m0['budget_pct_10']['missed_high_risk']})")
        print(f"  M2: MAE={reg_m2['mae']:.4f}, RMSE={reg_m2['rmse']:.4f}, R2={reg_m2['r2']:.4f}, Recall@10%={rank_m2['budget_pct_10']['recall']:.4f} (Missed: {rank_m2['budget_pct_10']['missed_high_risk']})")

    with open("reports/phase3b/step4b_posthoc_test_baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
