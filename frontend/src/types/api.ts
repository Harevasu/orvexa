export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  governance_candidate: string;
  supported_horizons: string[];
  disclaimer: string;
}

export interface QuantilesOutput {
  q05: number;
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
  q95: number;
}

export interface CQRIntervalOutput {
  lower: number;
  upper: number;
  width: number;
  confidence_level: number;
  target_alpha: number;
  conformal_shift_qhat: number;
}

export interface RiskClassificationOutput {
  level: string; // "CRITICAL / HIGH RISK", "MODERATE RISK", "LOW RISK", "NEGLIGIBLE / SAFE"
  description: string;
  log10_threshold: string;
  disclaimer: string;
}

export interface ModelArtifactInfo {
  candidate_id: string;
  architecture: string;
  horizon: string;
  lead_time_hours: number;
  model_weights_sha256: string;
  preprocessor_sha256: string;
  cqr_calibrator_sha256: string;
}

export interface InferenceResponse {
  event_id: string;
  horizon: string;
  horizon_days: number;
  lead_time_hours: number;
  qualifying_cdms_count: number;
  total_cdms_count: number;
  quantiles: QuantilesOutput;
  cqr_interval: CQRIntervalOutput;
  risk_assessment: RiskClassificationOutput;
  model_info: ModelArtifactInfo;
  disclaimer: string;
}

export interface MultiHorizonInferenceResponse {
  event_id: string;
  horizons: Record<string, InferenceResponse | null>;
  disclaimer: string;
}

export interface EventSummary {
  event_id: string;
  split: string;
  total_cdms: number;
  earliest_time_to_tca: number;
  latest_time_to_tca: number;
  target_final_risk: number;
  primary_object_type: string;
  min_miss_distance: number | null;
  max_mahalanobis: number | null;
  qualifies_h2: boolean;
  qualifies_h3: boolean;
  qualifies_h5: boolean;
  qualifies_h6: boolean;
}

export interface EventListResponse {
  total_count: number;
  page: number;
  page_size: number;
  events: EventSummary[];
}

export interface CDMRecord {
  time_to_tca: number;
  risk: number | null;
  c_object_type: string | null;
  miss_distance: number | null;
  relative_speed: number | null;
  relative_position_r: number | null;
  relative_position_t: number | null;
  relative_position_n: number | null;
  relative_velocity_r: number | null;
  relative_velocity_t: number | null;
  relative_velocity_n: number | null;
  mahalanobis_distance: number | null;
  t_sigma_r: number | null;
  t_sigma_t: number | null;
  t_sigma_n: number | null;
  c_sigma_r: number | null;
  c_sigma_t: number | null;
  c_sigma_n: number | null;
  t_obs_used: number | null;
  c_obs_used: number | null;
  F10: number | null;
  AP: number | null;
  SSN: number | null;
}

export interface EventDetailResponse {
  event_id: string;
  split: string;
  total_cdms: number;
  target_final_risk: number;
  primary_object_type: string;
  cdms: CDMRecord[];
  disclaimer: string;
}

export interface RankedAlertItem {
  rank: number;
  event_id: string;
  horizon: string;
  split: string;
  total_cdms: number;
  median_risk_q50: number;
  cqr_lower: number;
  cqr_upper: number;
  cqr_width: number;
  risk_level: string;
  target_final_risk: number;
}

export interface RankedAlertsResponse {
  horizon: string;
  total_events: number;
  alerts: RankedAlertItem[];
  disclaimer: string;
}

export interface HorizonBenchmarkItem {
  horizon: string;
  horizon_days: number;
  lead_time_hours: number;
  n_test_samples: number;
  q50_mae: number;
  q50_mae_ci: number[];
  q50_rmse: number;
  q50_rmse_ci: number[];
  q50_r2: number;
  q50_r2_ci: number[];
  q50_spearman_rho: number;
  q50_spearman_ci: number[];
  mean_pinball_loss: number;
  quantile_crossing_violations: number;
  raw_90pct_coverage: number;
  raw_90pct_mean_width: number;
  cqr_shift_qhat: number;
  cqr_90pct_coverage: number;
  cqr_mean_width: number;
  cqr_median_width: number;
  critical_events_count: number;
  cqr_tail_coverage: number | null;
  point_prediction_notes: string;
}

export interface BenchmarksResponse {
  benchmark_name: string;
  quarantined_test_evaluation_completed: boolean;
  evaluation_date: string;
  horizons: HorizonBenchmarkItem[];
  h6_degradation_explanation: string;
  disclaimer: string;
}

export interface OrbitalTrajectoryPoint {
  jd: number;
  timestamp_offset_sec: number;
  x_km: number;
  y_km: number;
  z_km: number;
  vx_km_s: number;
  vy_km_s: number;
  vz_km_s: number;
  latitude_deg: number;
  longitude_deg: number;
  altitude_km: number;
}

export interface OrbitalPropagationResponse {
  satellite_name: string;
  norad_id: string | null;
  epoch_jd: number;
  duration_hours: number;
  step_seconds: number;
  trajectory: OrbitalTrajectoryPoint[];
  notes: string;
  disclaimer: string;
}
