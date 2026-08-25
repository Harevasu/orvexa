# ORVEXA ESA Dataset Audit

## 1. Dataset Identity

- **File Path**: `data/raw/esa/train_data.csv`
- **File Integrity (SHA-256 Before Audit)**: `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6`
- **File Integrity (SHA-256 After Audit)**: `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6`
- **Integrity Status**: VERIFIED UNCHANGED (Strictly Read-Only)
- **Source Context**: ESA Collision Avoidance Challenge (Kelvins / Zenodo `4463683`)
- **File Format**: Standard Comma-Separated Values (CSV, UTF-8)

---

## 2. File Statistics

- **File Size**: `233,600,296` bytes (`222.78 MB`)
- **Total Row Count**: `162,634` rows (excluding single header row)
- **Total Column Count**: `103` columns
- **Duplicate Rows**: `0` exact duplicate rows (`0.0%`) validated using deterministic SHA-256 row-content hashing.

---

## 3. Exact Schema

The dataset contains 103 columns: 101 continuous numeric/integer fields, 1 categorical string field (`c_object_type`), and 1 event identifier (`event_id`).

### Complete Column Inventory (103 Columns)

| # | Column Name | Inferred Dtype | Missing Count | Missing % | Observed Min | Observed Max | Mean |
|---|---|---|---|---|---|---|---|
| 1 | `event_id` | int64 | 0 | 0.00% | 0 | 13153 | 6566.87 |
| 2 | `time_to_tca` | float64 | 0 | 0.00% | -0.1498 | 6.9938 | 3.3502 |
| 3 | `mission_id` | int64 | 0 | 0.00% | 1 | 24 | 7.12 |
| 4 | `risk` | float64 | 0 | 0.00% | -30.0000 | -1.4429 | -19.3406 |
| 5 | `max_risk_estimate` | float64 | 0 | 0.00% | -30.0000 | -0.7300 | -17.5135 |
| 6 | `max_risk_scaling` | float64 | 0 | 0.00% | -25.2974 | 22.7303 | -0.1983 |
| 7 | `miss_distance` | float64 | 0 | 0.00% | 0.0 | 2580715.0 | 25964.38 |
| 8 | `relative_speed` | float64 | 0 | 0.00% | 0.2 | 15467.8 | 10834.78 |
| 9 | `relative_position_r` | float64 | 0 | 0.00% | -21995.6 | 21545.9 | 1.83 |
| 10 | `relative_position_t` | float64 | 0 | 0.00% | -2560803.0 | 2580665.0 | 851.68 |
| 11 | `relative_position_n` | float64 | 0 | 0.00% | -108990.2 | 136279.7 | -71.28 |
| 12 | `relative_velocity_r` | float64 | 0 | 0.00% | -14364.5 | 14352.0 | 1.13 |
| 13 | `relative_velocity_t` | float64 | 0 | 0.00% | -15467.8 | 12903.0 | -5474.33 |
| 14 | `relative_velocity_n` | float64 | 0 | 0.00% | -15239.3 | 15291.6 | 137.95 |
| 15 | `t_time_lastob_start` | float64 | 0 | 0.00% | -42.8229 | 13.9111 | 4.3168 |
| 16 | `t_time_lastob_end` | float64 | 0 | 0.00% | -0.1498 | 6.9938 | 3.3934 |
| 17 | `t_recommended_od_span` | float64 | 0 | 0.00% | 1.0 | 10.0 | 3.65 |
| 18 | `t_actual_od_span` | float64 | 0 | 0.00% | 0.0 | 43.08 | 3.35 |
| 19 | `t_obs_available` | int64 | 0 | 0.00% | 0 | 7990 | 185.06 |
| 20 | `t_obs_used` | int64 | 0 | 0.00% | 0 | 7887 | 182.16 |
| 21 | `t_residuals_accepted` | float64 | 0 | 0.00% | 0.0 | 100.0 | 97.43 |
| 22 | `t_weighted_rms` | float64 | 0 | 0.00% | 0.0 | 3.864 | 0.7711 |
| 23 | `t_rcs_estimate` | float64 | 3277 | 2.02% | 0.0001 | 18.0699 | 1.5542 |
| 24 | `t_cd_area_over_mass` | float64 | 0 | 0.00% | 0.0 | 0.5898 | 0.0152 |
| 25 | `t_cr_area_over_mass` | float64 | 0 | 0.00% | 0.0 | 0.1691 | 0.0101 |
| 26 | `t_sedr` | float64 | 0 | 0.00% | 0.0 | 0.0029 | 0.000007 |
| 27 | `t_j2k_sma` | float64 | 0 | 0.00% | 6608.2 | 7847.6 | 7119.50 |
| 28 | `t_j2k_ecc` | float64 | 0 | 0.00% | 0.00007 | 0.0883 | 0.0051 |
| 29 | `t_j2k_inc` | float64 | 0 | 0.00% | 40.00 | 98.72 | 97.59 |
| 30 | `t_ct_r` | float64 | 0 | 0.00% | -1.0 | 1.0 | -0.1582 |
| 31 | `t_cn_r` | float64 | 0 | 0.00% | -0.9996 | 0.9996 | -0.0069 |
| 32 | `t_cn_t` | float64 | 0 | 0.00% | -0.9997 | 0.9997 | 0.0076 |
| 33 | `t_crdot_r` | float64 | 9230 | 5.68% | -1.0 | 1.0 | 0.1565 |
| 34 | `t_crdot_t` | float64 | 9230 | 5.68% | -1.0 | 1.0 | -0.9576 |
| 35 | `t_crdot_n` | float64 | 9230 | 5.68% | -0.9997 | 0.9996 | -0.0074 |
| 36 | `t_ctdot_r` | float64 | 9230 | 5.68% | -1.0 | 1.0 | 0.9904 |
| 37 | `t_ctdot_t` | float64 | 9230 | 5.68% | -1.0 | 1.0 | -0.1741 |
| 38 | `t_ctdot_n` | float64 | 9230 | 5.68% | -0.9996 | 0.9996 | -0.0075 |
| 39 | `t_ctdot_rdot` | float64 | 9230 | 5.68% | -1.0 | 1.0 | 0.1738 |
| 40 | `t_cndot_r` | float64 | 9230 | 5.68% | -0.9998 | 0.9998 | 0.0039 |
| 41 | `t_cndot_t` | float64 | 9230 | 5.68% | -0.9996 | 0.9996 | -0.0074 |
| 42 | `t_cndot_n` | float64 | 9230 | 5.68% | -1.0 | 1.0 | 0.0163 |
| 43 | `t_cndot_rdot` | float64 | 9230 | 5.68% | -0.9997 | 0.9998 | 0.0072 |
| 44 | `t_cndot_tdot` | float64 | 9230 | 5.68% | -0.9998 | 0.9998 | 0.0038 |
| 45 | `c_object_type` | string (cat) | 0 | 0.00% | N/A | N/A | N/A |
| 46 | `c_time_lastob_start` | float64 | 11 | 0.01% | -74.9658 | 18.7302 | 3.5186 |
| 47 | `c_time_lastob_end` | float64 | 11 | 0.01% | -0.1498 | 6.9938 | 3.4283 |
| 48 | `c_recommended_od_span` | float64 | 11 | 0.01% | 0.0 | 10.0 | 3.32 |
| 49 | `c_actual_od_span` | float64 | 11 | 0.01% | 0.0 | 75.05 | 2.50 |
| 50 | `c_obs_available` | float64 | 11 | 0.01% | 0 | 1968 | 24.37 |
| 51 | `c_obs_used` | float64 | 11 | 0.01% | 0 | 1968 | 24.18 |
| 52 | `c_residuals_accepted` | float64 | 11 | 0.01% | 0.0 | 100.0 | 97.43 |
| 53 | `c_weighted_rms` | float64 | 11 | 0.01% | 0.0 | 6.938 | 1.1353 |
| 54 | `c_rcs_estimate` | float64 | 52841 | 32.49% | 0.0001 | 18.0699 | 0.7028 |
| 55 | `c_cd_area_over_mass` | float64 | 0 | 0.00% | 0.0 | 0.4435 | 0.0135 |
| 56 | `c_cr_area_over_mass` | float64 | 0 | 0.00% | 0.0 | 0.0543 | 0.0009 |
| 57 | `c_sedr` | float64 | 0 | 0.00% | 0.0 | 0.0029 | 0.000007 |
| 58 | `c_j2k_sma` | float64 | 0 | 0.00% | 6586.2 | 10471.1 | 7132.89 |
| 59 | `c_j2k_ecc` | float64 | 0 | 0.00% | 0.00003 | 0.3540 | 0.0076 |
| 60 | `c_j2k_inc` | float64 | 0 | 0.00% | 0.0 | 144.60 | 85.06 |
| 61 | `c_ct_r` | float64 | 11 | 0.01% | -1.0 | 1.0 | -0.6698 |
| 62 | `c_cn_r` | float64 | 11 | 0.01% | -0.9996 | 0.9996 | -0.0051 |
| 63 | `c_cn_t` | float64 | 11 | 0.01% | -0.9997 | 0.9997 | 0.0069 |
| 64 | `c_crdot_r` | float64 | 9241 | 5.68% | -1.0 | 1.0 | 0.5843 |
| 65 | `c_crdot_t` | float64 | 9241 | 5.68% | -1.0 | 1.0 | -0.9634 |
| 66 | `c_crdot_n` | float64 | 9241 | 5.68% | -0.9997 | 0.9996 | -0.0073 |
| 67 | `c_ctdot_r` | float64 | 9241 | 5.68% | -1.0 | 1.0 | 0.9798 |
| 68 | `c_ctdot_t` | float64 | 9241 | 5.68% | -1.0 | 1.0 | -0.6847 |
| 69 | `c_ctdot_n` | float64 | 9241 | 5.68% | -0.9996 | 0.9996 | -0.0065 |
| 70 | `c_ctdot_rdot` | float64 | 9241 | 5.68% | -1.0 | 1.0 | 0.6015 |
| 71 | `c_cndot_r` | float64 | 9241 | 5.68% | -0.9998 | 0.9998 | 0.0039 |
| 72 | `c_cndot_t` | float64 | 9241 | 5.68% | -0.9996 | 0.9996 | -0.0067 |
| 73 | `c_cndot_n` | float64 | 9241 | 5.68% | -1.0 | 1.0 | 0.0159 |
| 74 | `c_cndot_rdot` | float64 | 9241 | 5.68% | -0.9997 | 0.9998 | 0.0067 |
| 75 | `c_cndot_tdot` | float64 | 9241 | 5.68% | -0.9998 | 0.9998 | 0.0039 |
| 76 | `t_span` | float64 | 0 | 0.00% | 0.0 | 43.08 | 3.35 |
| 77 | `c_span` | float64 | 0 | 0.00% | 0.0 | 75.05 | 2.50 |
| 78 | `t_h_apo` | float64 | 0 | 0.00% | 239.9 | 1548.8 | 777.62 |
| 79 | `t_h_per` | float64 | 0 | 0.00% | 224.2 | 1407.4 | 704.97 |
| 80 | `c_h_apo` | float64 | 0 | 0.00% | 210.8 | 6524.5 | 794.75 |
| 81 | `c_h_per` | float64 | 0 | 0.00% | 204.9 | 1475.4 | 686.72 |
| 82 | `geocentric_latitude` | float64 | 0 | 0.00% | -82.68 | 82.68 | 0.65 |
| 83 | `azimuth` | float64 | 0 | 0.00% | 0.0 | 360.0 | 163.63 |
| 84 | `elevation` | float64 | 0 | 0.00% | -89.47 | 89.28 | -0.57 |
| 85 | `mahalanobis_distance` | float64 | 0 | 0.00% | 0.0 | 10000000.0 | 1797.10 |
| 86 | `t_position_covariance_det` | float64 | 0 | 0.00% | 0.0 | 2.87e+17 | 1.83e+13 |
| 87 | `c_position_covariance_det` | float64 | 0 | 0.00% | 0.0 | 1.34e+22 | 8.85e+15 |
| 88 | `t_sigma_r` | float64 | 0 | 0.00% | 0.014 | 1481.5 | 5.37 |
| 89 | `c_sigma_r` | float64 | 11 | 0.01% | 0.023 | 240430.8 | 49.30 |
| 90 | `t_sigma_t` | float64 | 0 | 0.00% | 0.50 | 141705.5 | 504.60 |
| 91 | `c_sigma_t` | float64 | 11 | 0.01% | 0.53 | 1109968.0 | 1342.34 |
| 92 | `t_sigma_n` | float64 | 0 | 0.00% | 0.08 | 1581.5 | 6.74 |
| 93 | `c_sigma_n` | float64 | 11 | 0.01% | 0.07 | 4514.8 | 12.39 |
| 94 | `t_sigma_rdot` | float64 | 9230 | 5.68% | 0.0005 | 161.4 | 0.57 |
| 95 | `c_sigma_rdot` | float64 | 9241 | 5.68% | 0.0006 | 1256.7 | 1.51 |
| 96 | `t_sigma_tdot` | float64 | 9230 | 5.68% | 0.00001 | 1.70 | 0.0062 |
| 97 | `c_sigma_tdot` | float64 | 9241 | 5.68% | 0.00002 | 269.4 | 0.055 |
| 98 | `t_sigma_ndot` | float64 | 9230 | 5.68% | 0.00008 | 1.83 | 0.0078 |
| 99 | `c_sigma_ndot` | float64 | 9241 | 5.68% | 0.00008 | 5.16 | 0.0142 |
| 100 | `F10` | float64 | 6822 | 4.19% | 65.0 | 254.0 | 88.08 |
| 101 | `F3M` | float64 | 6822 | 4.19% | 67.2 | 215.1 | 89.28 |
| 102 | `SSN` | float64 | 6822 | 4.19% | 0.0 | 206.0 | 38.64 |
| 103 | `AP` | float64 | 6822 | 4.19% | 0.0 | 149.0 | 9.77 |

---

## 4. Missing Values

### Summary of Missingness
- **Columns with zero missing values**: `53` columns (51.46%)
- **Columns with missing values**: `50` columns (48.54%)
- **Total cells**: `16,751,302`
- **Total missing cells**: `336,666` (`2.0098%` overall missingness)

### Missing Value Breakdown by Frequency Tier

#### High Missingness (> 10%)
1. `c_rcs_estimate`: `52,841` missing (`32.49%`) — Radar cross-section is frequently uncatalogued for debris/unknown chasers.

#### Moderate Missingness (4% – 10%)
2. Velocity Covariance / Sigma cross-correlation fields (`9,241` missing, `5.68%`):
   - Chaser terms: `c_crdot_r`, `c_crdot_t`, `c_crdot_n`, `c_ctdot_r`, `c_ctdot_t`, `c_ctdot_n`, `c_ctdot_rdot`, `c_cndot_r`, `c_cndot_t`, `c_cndot_n`, `c_cndot_rdot`, `c_cndot_tdot`, `c_sigma_rdot`, `c_sigma_tdot`, `c_sigma_ndot`
3. Target Velocity Covariance / Sigma terms (`9,230` missing, `5.68%`):
   - Target terms: `t_crdot_r`, `t_crdot_t`, `t_crdot_n`, `t_ctdot_r`, `t_ctdot_t`, `t_ctdot_n`, `t_ctdot_rdot`, `t_cndot_r`, `t_cndot_t`, `t_cndot_n`, `t_cndot_rdot`, `t_cndot_tdot`, `t_sigma_rdot`, `t_sigma_tdot`, `t_sigma_ndot`
4. Space Weather Contextual Indices (`6,822` missing, `4.19%`):
   - `F10`, `F3M`, `SSN`, `AP`

#### Low Missingness (1% – 4%)
5. `t_rcs_estimate`: `3,277` missing (`2.02%`)

#### Trace Missingness (< 0.1%)
6. Chaser Observation / OD Quality and Position Covariance (`11` missing, `0.0068%` / `0.01%`):
   - `c_time_lastob_start`, `c_time_lastob_end`, `c_recommended_od_span`, `c_actual_od_span`, `c_obs_available`, `c_obs_used`, `c_residuals_accepted`, `c_weighted_rms`, `c_ct_r`, `c_cn_r`, `c_cn_t`, `c_sigma_r`, `c_sigma_t`, `c_sigma_n`

---

## 5. Event Structure & Sequence Length

- **Total Unique Events (`event_id`)**: `13,154`
- **Missing `event_id`**: `0` (`0.0%`)
- **Range of `event_id`**: `0` to `13,153` (continuous integer indexing)
- **Multi-CDM Events**: `12,929` events (`98.29%`) contain multiple CDMs ($\ge 2$).
- **Single-CDM Events**: `225` events (`1.71%`) contain exactly $1$ CDM.

### CDM Count Per Event Distribution
- **Minimum CDMs per event**: `1`
- **Maximum CDMs per event**: `23`
- **Mean CDMs per event**: `12.3638`
- **Median CDMs per event**: `13.0`

#### Selected Percentiles (CDM Count per Event)
- **p1**: `1.0`
- **p5**: `2.0`
- **p10**: `2.0`
- **p25**: `5.0`
- **p50 (Median)**: `13.0`
- **p75**: `20.0`
- **p90**: `21.0`
- **p95**: `21.0`
- **p99**: `22.0`

### Horizon Eligibility Statistics

For each warning horizon $H \in \{2, 3, 5, 7\}$ days, an event is eligible if it contains at least one incoming CDM satisfying `time_to_tca >= H`.

| Horizon ($H$) | Eligible Events | % of Total Events | Median Eligible CDMs/Event | Maximum Eligible CDMs/Event | Mean Eligible CDMs/Event |
|---|---|---|---|---|---|
| **$H = 2$ days** | **11,942** | **90.79%** | **10.0** | **17** | **9.37** |
| **$H = 3$ days** | **11,273** | **85.70%** | **9.0** | **14** | **7.83** |
| **$H = 5$ days** | **9,484** | **72.10%** | **5.0** | **8** | **4.48** |
| **$H = 7$ days** | **0** | **0.00%** | **0** | **0** | **0.00** |

*Note on $H = 7$ days: The maximum `time_to_tca` in the entire dataset is `6.9938` days (< 7.0 days). Thus, exactly zero events qualify for a strict 7-day cutoff. Research protocols must use $H \le 5$ days or define a 7-day proxy.*

---

## 6. Time and Ordering Fields

### Time-to-TCA Ordering Precision Check

We evaluated every adjacent pair of rows within each event to test whether $time\_to\_tca[i] > time\_to\_tca[i+1]$ strictly holds:

| Classification | Event Count | Percentage |
|---|---|---|
| **Strictly Decreasing Events** ($t_i > t_{i+1}$) | **13,154** | **100.00%** |
| **Non-Increasing Events with Ties** ($t_i \ge t_{i+1}$ with $\ge 1$ tie) | **0** | **0.00%** |
| **Strictly Increasing Events** ($t_i < t_{i+1}$) | **0** | **0.00%** |
| **Non-Monotonic Events** | **0** | **0.00%** |

**Conclusion**: The raw CSV guarantees strict monotonic decreasing order of `time_to_tca` for every multi-CDM event without a single timestamp collision or tie.

### Available Temporal Fields
1. `time_to_tca`: Continuous numeric warning time in days prior to Time of Closest Approach.
   - **Range**: `-0.1498` days to `+6.9938` days.
   - **Mean**: `3.3502` days.
2. Observation Window Spans:
   - `t_time_lastob_start`, `t_time_lastob_end`, `t_recommended_od_span`, `t_actual_od_span`, `t_span`
   - `c_time_lastob_start`, `c_time_lastob_end`, `c_recommended_od_span`, `c_actual_od_span`, `c_span`

---

## 7. Target and Risk Analysis

### Target Identity & Dtype
- **Target Field**: `risk`
- **Data Type**: `float64` (Continuous logarithmic risk score: $\log_{10}(\text{Risk})$)
- **Total Conjunction Events**: `13,154`
- **Events with Valid Final Risk**: `13,154` (**100.0%**)
- **Events without Valid Final Risk**: `0` (**0.0%**)

### True Final-CDM Risk Distribution (13,154 Final CDM Targets)
Identified strictly by selecting the row with minimum `time_to_tca` for each event:

- **Minimum**: `-30.0000`
- **Maximum**: `-1.6847` ($\approx 2.07\%$)
- **Mean**: `-23.2960`
- **Median**: `-30.0000`

#### Final-CDM Risk Percentiles
- **p1**: `-30.0000`
- **p5**: `-30.0000`
- **p10**: `-30.0000`
- **p25**: `-30.0000`
- **p50 (Median)**: `-30.0000`
- **p75**: `-13.9371`
- **p90**: `-7.4742`
- **p95**: `-6.5458`
- **p99**: `-5.3078`

### All-Row Risk Distribution (162,634 CDM Rows)
- **Minimum**: `-30.0000`
- **Maximum**: `-1.4429`
- **Mean**: `-19.3406`
- **Median**: `-17.8706`
- **Percentiles**: p1: `-30.0000`, p5: `-30.0000`, p10: `-30.0000`, p25: `-30.0000`, p50: `-17.8706`, p75: `-9.1733`, p90: `-6.5761`, p95: `-5.7331`, p99: `-4.4808`

---

## 8. Approved Feature Mapping

The safe feature policy enforces explicit classification. Every configured feature from `docs/ORVEXA_VIBE_CODING_SPEC.md` and `configs/features.yaml` is classified below.

### Status Definitions
- `DIRECT`: Exact column name exists in raw CSV.
- `APPROVED_EQUIVALENT`: Documented semantic equivalent established in project specification.
- `NOT_AVAILABLE`: Not present in raw CSV; no direct single column exists.
- `NEEDS_HUMAN_REVIEW`: Potential raw column exists but requires explicit human/domain approval before aliasing.

### Complete Feature Mapping Registry

| Config Feature | Raw Column | Status | Reason |
|---|---|---|---|
| **Mandatory Controls** | | | |
| `event_id` | `event_id` | `DIRECT` | Exact match; grouping key only. |
| `mission_id` | `mission_id` | `DIRECT` | Exact match; excluded from feature vectors. |
| `risk` | `risk` | `DIRECT` | Exact match; target only. |
| `time_to_tca` | `time_to_tca` | `DIRECT` | Exact match; warning time feature. |
| `c_object_type` | `c_object_type` | `DIRECT` | Exact match; categorical object class. |
| `max_risk_estimate` | `max_risk_estimate` | `DIRECT` | Exact match; ESA physics baseline. |
| `max_risk_scaling` | `max_risk_scaling` | `DIRECT` | Exact match; ESA physics baseline scaling. |
| **Encounter Geometry** | | | |
| `miss_distance` | `miss_distance` | `DIRECT` | Exact match; miss distance at TCA. |
| `relative_speed` | `relative_speed` | `DIRECT` | Exact match; relative velocity magnitude. |
| `relative_position_r` | `relative_position_r` | `DIRECT` | Exact match; RTN relative position (radial). |
| `relative_position_t` | `relative_position_t` | `DIRECT` | Exact match; RTN relative position (along-track). |
| `relative_position_n` | `relative_position_n` | `DIRECT` | Exact match; RTN relative position (normal). |
| `relative_velocity_r` | `relative_velocity_r` | `DIRECT` | Exact match; RTN relative velocity (radial). |
| `relative_velocity_t` | `relative_velocity_t` | `DIRECT` | Exact match; RTN relative velocity (along-track). |
| `relative_velocity_n` | `relative_velocity_n` | `DIRECT` | Exact match; RTN relative velocity (normal). |
| `mahalanobis_distance` | `mahalanobis_distance` | `DIRECT` | Exact match; covariance-scaled distance. |
| **Encounter Location** | | | |
| `geocentric_latitude` | `geocentric_latitude` | `DIRECT` | Exact match; encounter latitude. |
| `azimuth` | `azimuth` | `DIRECT` | Exact match; encounter azimuth. |
| `elevation` | `elevation` | `DIRECT` | Exact match; encounter elevation. |
| **Target Orbital State** | | | |
| `t_a` / `t_sma` | `t_j2k_sma` | `NEEDS_HUMAN_REVIEW` | Raw column `t_j2k_sma` is J2000 semi-major axis (km). Requires sign-off. |
| `t_e` / `t_eccentricity` | `t_j2k_ecc` | `NEEDS_HUMAN_REVIEW` | Raw column `t_j2k_ecc` is J2000 eccentricity. Requires sign-off. |
| `t_i` / `t_inclination` | `t_j2k_inc` | `NEEDS_HUMAN_REVIEW` | Raw column `t_j2k_inc` is J2000 inclination (deg). Requires sign-off. |
| `t_perigee` | `t_h_per` | `NEEDS_HUMAN_REVIEW` | Raw column `t_h_per` is perigee altitude (km), not orbit radius. Requires sign-off. |
| `t_apogee` | `t_h_apo` | `NEEDS_HUMAN_REVIEW` | Raw column `t_h_apo` is apogee altitude (km), not orbit radius. Requires sign-off. |
| `t_argp` | *None* | `NOT_AVAILABLE` | Argument of perigee is not in raw ESA dataset. |
| `t_raan` | *None* | `NOT_AVAILABLE` | RAAN is not in raw ESA dataset. |
| `t_mean_anomaly` | *None* | `NOT_AVAILABLE` | Mean anomaly is not in raw ESA dataset. |
| **Chaser Orbital State** | | | |
| `c_a` / `c_sma` | `c_j2k_sma` | `NEEDS_HUMAN_REVIEW` | Raw column `c_j2k_sma` is J2000 semi-major axis (km). Requires sign-off. |
| `c_e` / `c_eccentricity` | `c_j2k_ecc` | `NEEDS_HUMAN_REVIEW` | Raw column `c_j2k_ecc` is J2000 eccentricity. Requires sign-off. |
| `c_i` / `c_inclination` | `c_j2k_inc` | `NEEDS_HUMAN_REVIEW` | Raw column `c_j2k_inc` is J2000 inclination (deg). Requires sign-off. |
| `c_perigee` | `c_h_per` | `NEEDS_HUMAN_REVIEW` | Raw column `c_h_per` is perigee altitude (km). Requires sign-off. |
| `c_apogee` | `c_h_apo` | `NEEDS_HUMAN_REVIEW` | Raw column `c_h_apo` is apogee altitude (km). Requires sign-off. |
| `c_argp` | *None* | `NOT_AVAILABLE` | Chaser argument of perigee is not in raw CSV. |
| `c_raan` | *None* | `NOT_AVAILABLE` | Chaser RAAN is not in raw CSV. |
| `c_mean_anomaly` | *None* | `NOT_AVAILABLE` | Chaser mean anomaly is not in raw CSV. |
| **Uncertainty & Covariance** | | | |
| `t_sigma_r` | `t_sigma_r` | `DIRECT` | Exact match; target radial sigma. |
| `t_sigma_t` | `t_sigma_t` | `DIRECT` | Exact match; target along-track sigma. |
| `t_sigma_n` | `t_sigma_n` | `DIRECT` | Exact match; target cross-track sigma. |
| `c_sigma_r` | `c_sigma_r` | `DIRECT` | Exact match; chaser radial sigma. |
| `c_sigma_t` | `c_sigma_t` | `DIRECT` | Exact match; chaser along-track sigma. |
| `c_sigma_n` | `c_sigma_n` | `DIRECT` | Exact match; chaser cross-track sigma. |
| `miss_distance_sigma` | *None* | `NOT_AVAILABLE` | Not provided directly in raw CSV. |
| `covariance_determinant` | `t_pos...det`, `c_pos...det` | `NEEDS_HUMAN_REVIEW` | Raw has separate target & chaser position covariance determinants. |
| `covariance_correlation` | *None* | `NOT_AVAILABLE` | Raw has pairwise cross-terms (`t_ct_r`, etc.), not a single correlation scalar. |
| **Observation Quality** | | | |
| `t_obs_available` | `t_obs_available` | `DIRECT` | Exact match; target available observations. |
| `t_obs_used` | `t_obs_used` | `DIRECT` | Exact match; target used observations. |
| `c_obs_available` | `c_obs_available` | `DIRECT` | Exact match; chaser available observations. |
| `c_obs_used` | `c_obs_used` | `DIRECT` | Exact match; chaser used observations. |
| `t_residuals_accepted` | `t_residuals_accepted` | `DIRECT` | Exact match; percentage accepted residuals. |
| `c_residuals_accepted` | `c_residuals_accepted` | `DIRECT` | Exact match; percentage accepted residuals. |
| `t_weighted_rms` | `t_weighted_rms` | `DIRECT` | Exact match; weighted RMS. |
| `c_weighted_rms` | `c_weighted_rms` | `DIRECT` | Exact match; weighted RMS. |
| `t_od_span` / `c_od_span` | `*_actual_od_span` | `NEEDS_HUMAN_REVIEW` | Raw has both `actual_od_span` and `recommended_od_span`. |
| `t_rms` / `c_rms` | `*_weighted_rms` | `NEEDS_HUMAN_REVIEW` | Requires sign-off to map `rms` to `weighted_rms`. |
| `t_residual` / `c_residual`| `*_residuals_accepted`| `NEEDS_HUMAN_REVIEW` | Raw has residual acceptance percentage. Requires sign-off. |
| **Contextual Indices** | | | |
| `F10` / `f10` | `F10` | `DIRECT` | Exact match (case-insensitive); 10.7cm solar flux. |
| `F3M` / `f3m` | `F3M` | `DIRECT` | Exact match (case-insensitive); 81-day centered solar flux. |
| `AP` / `ap` | `AP` | `DIRECT` | Exact match (case-insensitive); geomagnetic index. |
| `SSN` / `ssn` | `SSN` | `DIRECT` | Exact match (case-insensitive); sunspot number. |

---

## 9. Potential Data Issues

1. **Floor Censoring at `-30.0`**: More than 50% of the final-CDM event risk targets are clamped at `-30.0`. Regression objectives (e.g. Huber loss) and classification thresholds (e.g. `risk >= -6.0`) must properly reflect this floor.
2. **Missing Chaser RCS (`32.49%`)**: `c_rcs_estimate` is absent for 52,841 CDMs. Train-only median imputation + missingness indicator is required.
3. **Missing Velocity Covariances (`5.68%`)**: 9,241 chaser rows and 9,230 target rows lack velocity-sigma components (`t_crdot_r`, `c_sigma_rdot`, etc.).
4. **Post-TCA CDMs (`391` rows)**: Rows with $time\_to\_tca < 0$ exist in the dataset; horizon cutoffs ($time\_to\_tca \ge H$) will naturally filter them.
5. **Zero 7-day Horizon Eligibility**: No CDM in the dataset has $time\_to\_tca \ge 7.0$ days.

---

## 10. Unknowns / Ambiguities

1. **Orbital Feature Aliasing**: Domain confirmation needed whether `t_j2k_sma` (km) and `t_h_per` (km altitude) should be adopted as the canonical orbital state features.
2. **Covariance Determinant Selection**: Confirmation needed whether `t_position_covariance_det` and `c_position_covariance_det` should both be retained as separate features.
3. **OD Span Selection**: Confirmation needed whether `t_actual_od_span` or `t_recommended_od_span` (or both) should be used.

---

## 11. Recommendation for Next Step

1. Obtain user approval for the proposed feature mappings flagged as `NEEDS_HUMAN_REVIEW`.
2. Update `configs/features.yaml` and `src/orvexa/schema.py` with the approved aliases.
3. Proceed to event-level dataset construction with strict horizon filtering ($H \in \{2, 3, 5\}$ days) and event-aware chronological splitting.
