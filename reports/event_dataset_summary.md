# ORVEXA Event-Level Dataset Construction Summary

## 1. Dataset Overview & Integrity

- **Source Dataset**: `data/raw/esa/train_data.csv`
- **Source SHA-256**: `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6` (Verified Untouched)
- **Total Raw Events**: `13,154`
- **Total Raw CDMs**: `162,634`
- **Direct Features Used**: `34`
- **Excluded `NEEDS_HUMAN_REVIEW` Features**: `23`
- **Excluded `NOT_AVAILABLE` Features**: `8`
- **Excluded Identifiers**: `event_id` (grouping key), `mission_id` (identity leakage control)
- **Target**: `risk` -> `final_risk` (excluded from model input features)

---

## 2. Horizon Eligibility & Sequence Statistics

| Horizon | Eligible Events | % Total Events | Total CDMs | Min Len | Max Len | Mean Len | Median Len | p25 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **H2 (2d)** | **11,942** | 90.79% | 111,939 | 1 | 17 | 9.3736 | 10.0 | 4.0 | 15.0 | 15.0 | 15.0 | 16.0 |
| **H3 (3d)** | **11,273** | 85.7% | 88,237 | 1 | 14 | 7.8273 | 9.0 | 4.0 | 12.0 | 12.0 | 12.0 | 13.0 |
| **H5 (5d)** | **9,484** | 72.1% | 42,524 | 1 | 8 | 4.4838 | 5.0 | 3.0 | 6.0 | 6.0 | 6.0 | 7.0 |

*Note on H = 7 days: Verified in audit that max(time_to_tca) = 6.9938 days, so 7-day cutoff produces exactly 0 eligible events and was not created.*

---

## 3. Target Distribution (`final_risk`)

| Horizon | Valid Targets | Missing Targets | Min | Max | Mean | Median | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **H2** | 11,942 | 0 | -30.0000 | -1.6847 | -23.1986 | -30.0000 | -30.0000 | -30.0000 | -30.0000 | -13.5879 | -7.4323 | -6.5439 | -5.3334 |
| **H3** | 11,273 | 0 | -30.0000 | -1.6847 | -23.2019 | -30.0000 | -30.0000 | -30.0000 | -30.0000 | -13.5095 | -7.4085 | -6.5353 | -5.3555 |
| **H5** | 9,484 | 0 | -30.0000 | -1.6847 | -23.4023 | -30.0000 | -30.0000 | -30.0000 | -30.0000 | -13.7805 | -7.4344 | -6.5771 | -5.3586 |

---

## 4. Missing Values per DIRECT Feature by Horizon

| # | Feature Name | H=2d Missing (%) | H=3d Missing (%) | H=5d Missing (%) |
|---|---|---|---|---|
| 1 | `time_to_tca` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | `c_object_type` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 3 | `max_risk_estimate` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 4 | `max_risk_scaling` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 5 | `miss_distance` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 6 | `relative_speed` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 7 | `relative_position_r` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 8 | `relative_position_t` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 9 | `relative_position_n` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 10 | `relative_velocity_r` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 11 | `relative_velocity_t` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 12 | `relative_velocity_n` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 13 | `mahalanobis_distance` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 14 | `geocentric_latitude` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 15 | `azimuth` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 16 | `elevation` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 17 | `t_sigma_r` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 18 | `t_sigma_t` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 19 | `t_sigma_n` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 20 | `c_sigma_r` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 21 | `c_sigma_t` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 22 | `c_sigma_n` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 23 | `t_obs_available` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 24 | `t_obs_used` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 25 | `c_obs_available` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 26 | `c_obs_used` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 27 | `t_residuals_accepted` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 28 | `c_residuals_accepted` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 29 | `t_weighted_rms` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 30 | `c_weighted_rms` | 11 (0.0098%) | 11 (0.0125%) | 0 (0.0%) |
| 31 | `F10` | 4,856 (4.3381%) | 3,859 (4.3734%) | 1,916 (4.5057%) |
| 32 | `F3M` | 4,856 (4.3381%) | 3,859 (4.3734%) | 1,916 (4.5057%) |
| 33 | `AP` | 4,856 (4.3381%) | 3,859 (4.3734%) | 1,916 (4.5057%) |
| 34 | `SSN` | 4,856 (4.3381%) | 3,859 (4.3734%) | 1,916 (4.5057%) |

---

## 5. Excluded Features Registry

### Excluded `NEEDS_HUMAN_REVIEW` Features (18 items)
The following features remain excluded pending human domain sign-off:
- `t_a`
- `t_sma`
- `t_e`
- `t_eccentricity`
- `t_i`
- `t_inclination`
- `t_perigee`
- `t_apogee`
- `c_a`
- `c_sma`
- `c_e`
- `c_eccentricity`
- `c_i`
- `c_inclination`
- `c_perigee`
- `c_apogee`
- `covariance_determinant`
- `t_od_span`
- `c_od_span`
- `t_rms`
- `c_rms`
- `t_residual`
- `c_residual`

### Excluded `NOT_AVAILABLE` Features (8 items)
- `t_argp`
- `t_raan`
- `t_mean_anomaly`
- `c_argp`
- `c_raan`
- `c_mean_anomaly`
- `miss_distance_sigma`
- `covariance_correlation`

### Excluded Identifiers & Target Leakage Controls
- `event_id`: Grouping identifier only
- `mission_id`: Excluded from feature representation
- `risk`: Excluded from inputs; used strictly as label source (`final_risk`)
