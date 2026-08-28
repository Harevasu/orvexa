"""Benchmarks service providing verified Phase 5 Internal Test metrics and candidate comparison results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.schemas import (
    BenchmarksResponse,
    HorizonBenchmarkItem,
    RankedAlertItem,
    RankedAlertsResponse,
)


class BenchmarksService:
    """Provides frozen scientific benchmark data for ORVEXA."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent
        self.metrics_csv_path = (
            self.workspace_root / "reports" / "phase5_step5_blind_internal_test_metrics.csv"
        )
        self.freeze_manifest_path = (
            self.workspace_root / "artifacts" / "models" / "phase5" / "candidate_freeze_manifest.json"
        )
        self.val_pred_paths = {
            "H2": self.workspace_root / "data" / "processed" / "predictions" / "phase5" / "tcn_quantile_M4_h2.0_val_predictions.csv",
            "H3": self.workspace_root / "data" / "processed" / "predictions" / "phase5" / "tcn_quantile_M4_h3.0_val_predictions.csv",
            "H5": self.workspace_root / "data" / "processed" / "predictions" / "phase5" / "tcn_quantile_M4_h5.0_val_predictions.csv",
            "H6": self.workspace_root / "data" / "processed" / "predictions" / "phase5" / "tcn_quantile_M4_h6.0_val_predictions.csv",
        }

    def get_benchmarks(self) -> BenchmarksResponse:
        """Return structured benchmark metrics across all 4 operational warning horizons."""
        if not self.metrics_csv_path.exists():
            raise FileNotFoundError(f"Missing benchmark metrics CSV: {self.metrics_csv_path}")

        df = pd.read_csv(self.metrics_csv_path)
        horizons_list: List[HorizonBenchmarkItem] = []

        for _, row in df.iterrows():
            h_name = str(row["horizon"])
            h_days = float(row["horizon_days"])
            lead_h = h_days * 24.0

            if h_name == "H6":
                point_notes = (
                    "Negative R² (-0.16651) observed on Phase 5 Internal Test. "
                    "At 6-day lead times, physical ephemeris uncertainty dominates point regressions, "
                    "making point estimates noisy. However, CQR interval coverage remains empirically valid "
                    "at 90.20% (meeting the 90.0% nominal target), providing reliable risk bounds."
                )
            elif h_name == "H5":
                point_notes = (
                    "Moderate point accuracy (R² = +0.18967, Spearman rho = 0.52281). "
                    "CQR coverage achieves 92.96% with mean width 20.83 log-units."
                )
            elif h_name == "H3":
                point_notes = (
                    "Strong point accuracy (R² = +0.48416, Spearman rho = 0.68016). "
                    "CQR coverage achieves 92.58% with mean width 17.95 log-units."
                )
            else:
                point_notes = (
                    "Superior point accuracy (R² = +0.58497, Spearman rho = 0.71880). "
                    "CQR coverage achieves 92.15% with tight mean width 12.54 log-units."
                )

            item = HorizonBenchmarkItem(
                horizon=h_name,
                horizon_days=h_days,
                lead_time_hours=lead_h,
                n_test_samples=int(row["n_test_samples"]),
                q50_mae=round(float(row["q50_mae"]), 4),
                q50_mae_ci=[round(float(row["q50_mae_ci_lower"]), 4), round(float(row["q50_mae_ci_upper"]), 4)],
                q50_rmse=round(float(row["q50_rmse"]), 4),
                q50_rmse_ci=[round(float(row["q50_rmse_ci_lower"]), 4), round(float(row["q50_rmse_ci_upper"]), 4)],
                q50_r2=round(float(row["q50_r2"]), 5),
                q50_r2_ci=[round(float(row["q50_r2_ci_lower"]), 5), round(float(row["q50_r2_ci_upper"]), 5)],
                q50_spearman_rho=round(float(row["q50_spearman_rho"]), 5),
                q50_spearman_ci=[round(float(row["q50_spearman_ci_lower"]), 5), round(float(row["q50_spearman_ci_upper"]), 5)],
                mean_pinball_loss=round(float(row["mean_pinball_loss"]), 5),
                quantile_crossing_violations=int(row["quantile_crossing_violations"]),
                raw_90pct_coverage=round(float(row["raw_90pct_coverage"]) * 100.0, 2),
                raw_90pct_mean_width=round(float(row["raw_90pct_mean_width"]), 2),
                cqr_shift_qhat=round(float(row["cqr_shift_qhat"]), 4),
                cqr_90pct_coverage=round(float(row["cqr_90pct_coverage"]) * 100.0, 2),
                cqr_mean_width=round(float(row["cqr_mean_width"]), 2),
                cqr_median_width=round(float(row["cqr_median_width"]), 2),
                critical_events_count=int(row["critical_events_count"]),
                cqr_tail_coverage=(
                    round(float(row["cqr_tail_coverage"]) * 100.0, 2)
                    if pd.notna(row.get("cqr_tail_coverage"))
                    else None
                ),
                point_prediction_notes=point_notes,
            )
            horizons_list.append(item)

        return BenchmarksResponse(
            benchmark_name="Phase 5 Blind Internal Test (1,677 events)",
            quarantined_test_evaluation_completed=True,
            evaluation_date="2026-08-28T18:47:00Z",
            horizons=horizons_list,
            h6_degradation_explanation=(
                "Scientific Reality: Long-lead conjunction prediction at H6 (144h / 6 days) experiences "
                "epistemic state degradation due to orbital drag variability and atmospheric fluctuations. "
                "While point predictors yield negative R² (-0.16651), Inductive Split Conformal Prediction (CQR) "
                "maintains valid empirical coverage (90.20% >= 90.0% nominal) and prevents overconfident false certainty."
            ),
        )

    def get_candidate_manifest(self) -> Dict[str, Any]:
        """Return the Candidate C freeze manifest."""
        if not self.freeze_manifest_path.exists():
            raise FileNotFoundError(f"Missing freeze manifest: {self.freeze_manifest_path}")

        with open(self.freeze_manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_ranked_alerts(
        self,
        horizon: str = "H2",
        min_risk: Optional[float] = None,
        limit: int = 100,
    ) -> RankedAlertsResponse:
        """Return ranked list of conjunction alerts using allowed validation predictions."""
        h_key = horizon.upper()
        if h_key not in self.val_pred_paths:
            raise ValueError(f"Invalid horizon '{horizon}'. Must be one of H2, H3, H5, H6.")

        path = self.val_pred_paths[h_key]
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")

        # Load CQR q_hat for this horizon
        manifest = self.get_candidate_manifest()
        frozen_h = manifest["frozen_artifacts"][h_key]
        cal_path = self.workspace_root / frozen_h["cqr_calibrator"]

        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        q_hat = float(cal_data["calibration_score_summary"]["p90"])

        df = pd.read_csv(path)
        # Sort descending by q_50 (highest predicted risk first)
        df = df.sort_values("q_50", ascending=False)

        alerts: List[RankedAlertItem] = []
        rank = 1

        for _, row in df.iterrows():
            q50 = float(row["q_50"])
            q05 = float(row["q_05"])
            q95 = float(row["q_95"])

            cqr_low = q05 - q_hat
            cqr_high = q95 + q_hat
            cqr_w = cqr_high - cqr_low
            y_t = float(row["y_true"])

            if min_risk is not None and q50 < min_risk:
                continue

            # Categorization
            if q50 >= -4.0:
                r_level = "CRITICAL / HIGH RISK"
            elif q50 >= -6.0:
                r_level = "MODERATE RISK"
            elif q50 >= -15.0:
                r_level = "LOW RISK"
            else:
                r_level = "NEGLIGIBLE RISK"

            alerts.append(
                RankedAlertItem(
                    rank=rank,
                    event_id=str(row["event_id"]),
                    horizon=h_key,
                    split="validation",
                    total_cdms=23,  # Standard window
                    median_risk_q50=round(q50, 4),
                    cqr_lower=round(cqr_low, 4),
                    cqr_upper=round(cqr_high, 4),
                    cqr_width=round(cqr_w, 4),
                    risk_level=r_level,
                    target_final_risk=round(y_t, 4),
                )
            )
            rank += 1
            if len(alerts) >= limit:
                break

        return RankedAlertsResponse(
            horizon=h_key,
            total_events=len(alerts),
            alerts=alerts,
        )


# Global singleton instance
benchmarks_service = BenchmarksService()
