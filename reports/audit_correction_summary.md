# ORVEXA Audit Correction Summary

This document summarizes the corrections, verifications, and precision enhancements made to the ESA dataset audit for `data/raw/esa/train_data.csv`.

---

## 1. What Was Corrected

1. **True Final-CDM Risk Calculation**: Separated the aggregate row-level risk statistics from the event-level final-CDM risk targets. Evaluated the final CDM (minimum `time_to_tca`) for every event independently.
2. **Precision Time-Order Validation**: Audited adjacent time differences ($time\_to\_tca[i] > time\_to\_tca[i+1]$) across all 13,154 events to test for ties, non-monotonicity, or reversals.
3. **Deterministic Duplicate Detection**: Replaced standard Python hashing with cryptographic SHA-256 row-content hashing.
4. **Horizon Eligibility Modeling**: Added multi-horizon eligibility analysis ($H \in \{2, 3, 5, 7\}$ days) without building training datasets.
5. **Explicit Feature Mapping Registry**: Created a categorized `# Approved Feature Mapping` matrix with strict status tags (`DIRECT`, `APPROVED_EQUIVALENT`, `NOT_AVAILABLE`, `NEEDS_HUMAN_REVIEW`) to eliminate silent substitutions.
6. **Raw File Integrity Verification**: Verified that the SHA-256 checksum of `data/raw/esa/train_data.csv` was completely unchanged before and after script execution.

---

## 2. Final-CDM Risk Verification

Every conjunction event's final CDM was identified by selecting the qualifying record with the minimum `time_to_tca`.

- **Total Events**: `13,154`
- **Events with Valid Final Risk**: `13,154` (**100.0%**)
- **Events without Valid Final Risk**: `0` (**0.0%**)
- **Minimum Final Risk**: `-30.0000`
- **Maximum Final Risk**: `-1.6847` ($\approx 2.07\%$)
- **Mean Final Risk**: `-23.2960`
- **Median Final Risk**: `-30.0000`

### Final-CDM Risk Percentiles
| Percentile | Final-CDM Risk ($\log_{10}$) |
|---|---|
| **p1** | `-30.0000` |
| **p5** | `-30.0000` |
| **p10** | `-30.0000` |
| **p25** | `-30.0000` |
| **p50 (Median)** | `-30.0000` |
| **p75** | `-13.9371` |
| **p90** | `-7.4742` |
| **p95** | `-6.5458` |
| **p99** | `-5.3078` |

*Finding: In more than 50% of events, the final CDM risk resolves to the ESA operational floor of $-30.0$.*

---

## 3. Horizon Eligibility Statistics

Evaluated for warning horizons $H \in \{2, 3, 5, 7\}$ days based on events having $\ge 1$ CDM with $time\_to\_tca \ge H$:

| Horizon ($H$) | Eligible Events | Eligible % | Median Eligible CDMs | Maximum Eligible CDMs | Mean Eligible CDMs |
|---|---|---|---|---|---|
| **$H = 2$ days** | **11,942** | **90.79%** | **10.0** | **17** | **9.37** |
| **$H = 3$ days** | **11,273** | **85.70%** | **9.0** | **14** | **7.83** |
| **$H = 5$ days** | **9,484** | **72.10%** | **5.0** | **8** | **4.48** |
| **$H = 7$ days** | **0** | **0.00%** | **0** | **0** | **0.00** |

*Finding: Because the maximum `time_to_tca` in the raw CSV is `6.9938` days, zero events qualify for a 7-day horizon. Future experiment configs must restrict evaluation to $H \le 5$ days.*

---

## 4. Time-Order Validation

Every adjacent pair of rows in all 13,154 events was tested:

- **Strictly Decreasing Events ($t_i > t_{i+1}$)**: **13,154 (100.0%)**
- **Non-Increasing Events with Ties ($t_i \ge t_{i+1}$ with ties)**: **0 (0.0%)**
- **Increasing Events**: **0 (0.0%)**
- **Non-Monotonic Events**: **0 (0.0%)**

*Conclusion: The raw dataset contains zero timestamp ties; time ordering is strictly monotonic decreasing.*

---

## 5. Feature Mapping Decisions

To prevent unauthorized renaming or assumptions, candidate column alignments are explicitly tagged:

### Direct Matches (`DIRECT` — 24 features + 6 controls)
- All mandatory IDs/targets/controls (`event_id`, `mission_id`, `risk`, `time_to_tca`, `c_object_type`, `max_risk_estimate`, `max_risk_scaling`)
- Encounter geometry (`miss_distance`, `relative_speed`, `relative_position_*`, `relative_velocity_*`, `mahalanobis_distance`)
- Encounter location (`geocentric_latitude`, `azimuth`, `elevation`)
- Position covariances (`t_sigma_r`, `t_sigma_t`, `t_sigma_n`, `c_sigma_r`, `c_sigma_t`, `c_sigma_n`)
- Observation counts & residuals (`t_obs_available`, `t_obs_used`, `c_obs_available`, `c_obs_used`, `t_residuals_accepted`, `c_residuals_accepted`, `t_weighted_rms`, `c_weighted_rms`)
- Space weather (`F10`, `F3M`, `AP`, `SSN`)

### Features Requiring Review (`NEEDS_HUMAN_REVIEW` — 18 candidates)
- Orbital elements: `t_j2k_sma` (J2000 SMA km), `t_j2k_ecc`, `t_j2k_inc`, `t_h_per` (altitude km), `t_h_apo` (altitude km), and chaser equivalents (`c_j2k_sma`, `c_j2k_ecc`, `c_j2k_inc`, `c_h_per`, `c_h_apo`).
- Covariances: `t_position_covariance_det`, `c_position_covariance_det`.
- OD spans: `t_actual_od_span`, `c_actual_od_span` vs `*_recommended_od_span`.

### Features Not Available (`NOT_AVAILABLE` — 7 items)
- Orbital angles: `t_argp`, `t_raan`, `t_mean_anomaly`, `c_argp`, `c_raan`, `c_mean_anomaly`.
- Derived scalar covariances: `miss_distance_sigma`, `covariance_correlation`.

---

## 6. Duplicate Validation

- **Exact Duplicate Rows**: `0`
- **Duplicate Percentage**: `0.00%`
- **Validation Technique**: Deterministic SHA-256 content hashing across all 162,634 rows.

---

## 7. Raw-File Checksum Verification

- **SHA-256 (Before)**: `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6`
- **SHA-256 (After)**:  `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6`
- **Status**: Verified identical; raw data was completely untouched.

---

## 8. Remaining Unknowns

1. **Approval of Orbital Aliases**: Whether to formally adopt `t_j2k_sma`, `t_j2k_ecc`, `t_j2k_inc`, `t_h_per`, and `t_h_apo` (and chaser counterparts) for the orbital feature group.
2. **Approval of Position Covariance Determinants**: Whether to include both target (`t_position_covariance_det`) and chaser (`c_position_covariance_det`) 3x3 determinants.
3. **OD Span Selection**: Whether to include `actual_od_span` or both `actual` and `recommended` spans.

---

## 9. Readiness for Event-Level Construction

**Yes.** With strict time ordering validated, exact final-CDM risk verified across 100% of events, horizon eligibility established ($H \in \{2, 3, 5\}$ days), and duplicate rows proven absent, the dataset audit is complete and ready for event-level prefix construction upon resolution of the feature mapping items marked `NEEDS_HUMAN_REVIEW`.
