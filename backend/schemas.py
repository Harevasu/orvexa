"""Pydantic data schemas for ORVEXA FastAPI backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str = "ORVEXA Conjunction Risk Prioritization API"
    version: str = "1.0.0"
    governance_candidate: str = "Candidate C (Quantile M4 Causal TCN + CQR)"
    supported_horizons: List[str] = ["H2", "H3", "H5", "H6"]
    disclaimer: str = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


class QuantilesOutput(BaseModel):
    q05: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q95: float


class CQRIntervalOutput(BaseModel):
    lower: float
    upper: float
    width: float
    confidence_level: float
    target_alpha: float
    conformal_shift_qhat: float


class RiskClassificationOutput(BaseModel):
    level: str  # "CRITICAL", "HIGH", "MODERATE", "LOW", "NEGLIGIBLE"
    description: str
    log10_threshold: str
    disclaimer: str = "Research classification estimate only."


class ModelArtifactInfo(BaseModel):
    candidate_id: str
    architecture: str
    horizon: str
    lead_time_hours: float
    model_weights_sha256: str
    preprocessor_sha256: str
    cqr_calibrator_sha256: str


class CDMRecord(BaseModel):
    time_to_tca: float
    risk: Optional[float] = None
    c_object_type: Optional[str] = None
    miss_distance: Optional[float] = None
    relative_speed: Optional[float] = None
    relative_position_r: Optional[float] = None
    relative_position_t: Optional[float] = None
    relative_position_n: Optional[float] = None
    relative_velocity_r: Optional[float] = None
    relative_velocity_t: Optional[float] = None
    relative_velocity_n: Optional[float] = None
    mahalanobis_distance: Optional[float] = None
    t_sigma_r: Optional[float] = None
    t_sigma_t: Optional[float] = None
    t_sigma_n: Optional[float] = None
    c_sigma_r: Optional[float] = None
    c_sigma_t: Optional[float] = None
    c_sigma_n: Optional[float] = None
    t_obs_used: Optional[float] = None
    c_obs_used: Optional[float] = None
    F10: Optional[float] = None
    AP: Optional[float] = None
    SSN: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class EventSummary(BaseModel):
    event_id: str
    split: str  # "validation" or "calibration"
    total_cdms: int
    earliest_time_to_tca: float
    latest_time_to_tca: float
    target_final_risk: float
    primary_object_type: str
    min_miss_distance: Optional[float] = None
    max_mahalanobis: Optional[float] = None
    qualifies_h2: bool
    qualifies_h3: bool
    qualifies_h5: bool
    qualifies_h6: bool


class EventListResponse(BaseModel):
    total_count: int
    page: int
    page_size: int
    events: List[EventSummary]


class EventDetailResponse(BaseModel):
    event_id: str
    split: str
    total_cdms: int
    target_final_risk: float
    primary_object_type: str
    cdms: List[CDMRecord]
    disclaimer: str = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


class InferenceRequest(BaseModel):
    event_id: str
    horizon: str = Field("H2", description="Horizon key: H2, H3, H5, or H6")
    alpha: float = Field(0.10, ge=0.01, le=0.50, description="Significance level (default 0.10 for 90% coverage)")


class InferenceResponse(BaseModel):
    event_id: str
    horizon: str
    horizon_days: float
    lead_time_hours: float
    qualifying_cdms_count: int
    total_cdms_count: int
    quantiles: QuantilesOutput
    cqr_interval: CQRIntervalOutput
    risk_assessment: RiskClassificationOutput
    model_info: ModelArtifactInfo
    disclaimer: str = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


class MultiHorizonInferenceResponse(BaseModel):
    event_id: str
    horizons: Dict[str, Optional[InferenceResponse]]
    disclaimer: str = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


class RankedAlertItem(BaseModel):
    rank: int
    event_id: str
    horizon: str
    split: str
    total_cdms: int
    median_risk_q50: float
    cqr_lower: float
    cqr_upper: float
    cqr_width: float
    risk_level: str
    target_final_risk: float


class RankedAlertsResponse(BaseModel):
    horizon: str
    total_events: int
    alerts: List[RankedAlertItem]
    disclaimer: str = "Research estimate only. ORVEXA is not an operational collision-avoidance authority."


class HorizonBenchmarkItem(BaseModel):
    horizon: str
    horizon_days: float
    lead_time_hours: float
    n_test_samples: int
    q50_mae: float
    q50_mae_ci: List[float]
    q50_rmse: float
    q50_rmse_ci: List[float]
    q50_r2: float
    q50_r2_ci: List[float]
    q50_spearman_rho: float
    q50_spearman_ci: List[float]
    mean_pinball_loss: float
    quantile_crossing_violations: int
    raw_90pct_coverage: float
    raw_90pct_mean_width: float
    cqr_shift_qhat: float
    cqr_90pct_coverage: float
    cqr_mean_width: float
    cqr_median_width: float
    critical_events_count: int
    cqr_tail_coverage: Optional[float]
    point_prediction_notes: str


class BenchmarksResponse(BaseModel):
    benchmark_name: str
    quarantined_test_evaluation_completed: bool
    evaluation_date: str
    horizons: List[HorizonBenchmarkItem]
    h6_degradation_explanation: str
    disclaimer: str = "Historical Internal Test benchmarks. Research estimate only."


class OrbitalTrajectoryPoint(BaseModel):
    jd: float
    timestamp_offset_sec: float
    x_km: float
    y_km: float
    z_km: float
    vx_km_s: float
    vy_km_s: float
    vz_km_s: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float


class OrbitalPropagationResponse(BaseModel):
    satellite_name: str
    norad_id: Optional[str]
    epoch_jd: float
    duration_hours: float
    step_seconds: float
    trajectory: List[OrbitalTrajectoryPoint]
    notes: str
    disclaimer: str = "Auxiliary orbital ephemeris tool. Decoupled from ESA ML risk scoring."
