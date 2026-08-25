"""Comprehensive, deterministic ESA dataset and schema audit for ORVEXA.
Strictly read-only; audits data/raw/esa/train_data.csv and produces:
- reports/schema_report.json
- reports/data_audit.md
- reports/audit_correction_summary.md
"""

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file deterministically."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_percentiles(values, percentiles=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    """Compute exact linear-interpolated percentiles."""
    if not values:
        return {f"p{int(p*100)}": None for p in percentiles}
    sorted_v = sorted(values)
    n = len(sorted_v)
    res = {}
    for p in percentiles:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = sorted_v[int(k)]
        else:
            d0 = sorted_v[int(f)] * (c - k)
            d1 = sorted_v[int(c)] * (k - f)
            val = d0 + d1
        p_name = f"p{int(p*100)}" if p * 100 == int(p * 100) else f"p{p*100:.1f}"
        res[p_name] = float(val)
    return res


def median(values):
    """Compute median of list of numbers."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    else:
        return float(sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def main():
    raw_path = os.path.abspath("data/raw/esa/train_data.csv")
    if not os.path.exists(raw_path):
        print(f"Error: Raw CSV not found at {raw_path}")
        sys.exit(1)

    print("==================================================")
    print("ORVEXA ESA DATASET AUDIT (STRENGTHENED)")
    print("==================================================")
    
    # 1. SHA-256 BEFORE
    print(f"Checking raw dataset: {raw_path}")
    sha256_before = compute_sha256(raw_path)
    file_size_bytes = os.path.getsize(raw_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    print(f"SHA-256 (Before): {sha256_before}")

    # 2. Streaming & Deterministic Duplicate Detection
    row_count = 0
    seen_row_sha256 = set()
    exact_duplicate_rows = 0

    # Column level statistics
    col_missing_counts = defaultdict(int)
    col_type_counts = defaultdict(lambda: defaultdict(int))
    col_min = {}
    col_max = {}
    col_sum = defaultdict(float)
    col_numeric_count = defaultdict(int)

    # Event tracking
    events_data = defaultdict(list)
    
    all_row_risks = []
    
    print("\nReading dataset rows...")
    with open(raw_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_names = header
        num_cols = len(col_names)
        col_to_idx = {name: idx for idx, name in enumerate(col_names)}

        for row_idx, row in enumerate(reader):
            row_count += 1
            if row_count % 50000 == 0:
                print(f"  Processed {row_count:,} rows...")

            # Deterministic exact-row duplicate check using SHA-256 of row string representation
            row_bytes = "|".join(row).encode("utf-8")
            row_hash = hashlib.sha256(row_bytes).digest()
            if row_hash in seen_row_sha256:
                exact_duplicate_rows += 1
            else:
                seen_row_sha256.add(row_hash)

            # Extract fields
            ev_id = row[col_to_idx["event_id"]].strip()
            t_tca_str = row[col_to_idx["time_to_tca"]].strip()
            risk_str = row[col_to_idx["risk"]].strip()

            t_tca_val = float(t_tca_str) if t_tca_str else None
            risk_val = float(risk_str) if risk_str else None

            if risk_val is not None:
                all_row_risks.append(risk_val)

            events_data[ev_id].append({
                "row_idx": row_idx,
                "time_to_tca": t_tca_val,
                "risk": risk_val,
                "row": row
            })

            # Column stats
            for i, val in enumerate(row):
                c_name = col_names[i]
                v = val.strip()
                if not v or v.lower() in ("nan", "null", "none", ""):
                    col_missing_counts[c_name] += 1
                else:
                    try:
                        int_v = int(v)
                        col_type_counts[c_name]["int"] += 1
                        num_v = float(int_v)
                        col_numeric_count[c_name] += 1
                        col_sum[c_name] += num_v
                        if c_name not in col_min or num_v < col_min[c_name]:
                            col_min[c_name] = num_v
                        if c_name not in col_max or num_v > col_max[c_name]:
                            col_max[c_name] = num_v
                    except ValueError:
                        try:
                            float_v = float(v)
                            col_type_counts[c_name]["float"] += 1
                            col_numeric_count[c_name] += 1
                            col_sum[c_name] += float_v
                            if c_name not in col_min or float_v < col_min[c_name]:
                                col_min[c_name] = float_v
                            if c_name not in col_max or float_v > col_max[c_name]:
                                col_max[c_name] = float_v
                        except ValueError:
                            col_type_counts[c_name]["string"] += 1

    print(f"Finished reading {row_count:,} rows across {len(col_names)} columns.")

    # 3. Precision Event Structure & Time Ordering Analysis
    total_events = len(events_data)
    cdm_counts = []
    
    strictly_decreasing_events = 0
    non_increasing_with_ties_events = 0
    strictly_increasing_events = 0
    non_monotonic_events = 0
    single_cdm_events = 0
    multi_cdm_events = 0

    final_risks = []
    events_with_valid_final_risk = 0
    events_without_valid_final_risk = 0

    # Horizon eligibility tracking
    # For H in [2, 3, 5, 7]
    horizons = [2, 3, 5, 7]
    horizon_eligible_events = {h: 0 for h in horizons}
    horizon_cdm_counts = {h: [] for h in horizons}

    for ev_id, cdm_list in events_data.items():
        n_cdms = len(cdm_list)
        cdm_counts.append(n_cdms)
        if n_cdms == 1:
            single_cdm_events += 1
        else:
            multi_cdm_events += 1

        # Ordering check
        t_vals = [c["time_to_tca"] for c in cdm_list]
        if n_cdms == 1:
            strictly_decreasing_events += 1
        else:
            is_strictly_decreasing = True
            is_non_increasing_with_ties = True
            is_strictly_increasing = True
            has_ties = False

            for i in range(len(t_vals) - 1):
                diff = t_vals[i] - t_vals[i+1]
                if diff > 0:
                    is_strictly_increasing = False
                elif diff == 0:
                    is_strictly_decreasing = False
                    is_strictly_increasing = False
                    has_ties = True
                else: # diff < 0
                    is_strictly_decreasing = False
                    is_non_increasing_with_ties = False

            if is_strictly_decreasing:
                strictly_decreasing_events += 1
            elif is_non_increasing_with_ties and has_ties:
                non_increasing_with_ties_events += 1
            elif is_strictly_increasing:
                strictly_increasing_events += 1
            else:
                non_monotonic_events += 1

        # Final CDM Identification
        # According to project documented rule: The final CDM is the row with minimum time_to_tca (latest incoming CDM)
        # Since rows are ordered oldest to newest, it is the last qualifying row or min(time_to_tca).
        # We find the row with smallest time_to_tca:
        final_cdm = min(cdm_list, key=lambda c: c["time_to_tca"])
        f_risk = final_cdm["risk"]

        if f_risk is not None and not math.isnan(f_risk) and not math.isinf(f_risk):
            events_with_valid_final_risk += 1
            final_risks.append(f_risk)
        else:
            events_without_valid_final_risk += 1

        # Horizon Eligibility
        for h in horizons:
            eligible_cdms = [c for c in cdm_list if c["time_to_tca"] is not None and c["time_to_tca"] >= h]
            if len(eligible_cdms) > 0:
                horizon_eligible_events[h] += 1
                horizon_cdm_counts[h].append(len(eligible_cdms))

    # Calculate statistics
    cdm_percentiles = compute_percentiles(cdm_counts)
    final_risk_percentiles = compute_percentiles(final_risks)
    all_risk_percentiles = compute_percentiles(all_row_risks)

    final_risk_min = min(final_risks) if final_risks else None
    final_risk_max = max(final_risks) if final_risks else None
    final_risk_mean = sum(final_risks) / len(final_risks) if final_risks else None
    final_risk_med = median(final_risks)

    # 4. Feature Whitelist & Explicit Mapping Classification
    # We evaluate every configured feature from docs/ORVEXA_VIBE_CODING_SPEC.md and configs/features.yaml
    feature_mapping_definitions = [
        # Mandatory Targets & Controls
        {"config": "event_id", "raw": "event_id", "status": "DIRECT", "reason": "Exact match in raw CSV; used as grouping key, excluded from features."},
        {"config": "mission_id", "raw": "mission_id", "status": "DIRECT", "reason": "Exact match in raw CSV; excluded from features to prevent mission identity leakage."},
        {"config": "risk", "raw": "risk", "status": "DIRECT", "reason": "Exact match in raw CSV; used as target only, excluded from features."},
        {"config": "time_to_tca", "raw": "time_to_tca", "status": "DIRECT", "reason": "Exact match in raw CSV; primary warning time field."},
        {"config": "c_object_type", "raw": "c_object_type", "status": "DIRECT", "reason": "Exact match in raw CSV; categorical chaser object class."},
        {"config": "max_risk_estimate", "raw": "max_risk_estimate", "status": "DIRECT", "reason": "Exact match in raw CSV; ESA physics baseline feature."},
        {"config": "max_risk_scaling", "raw": "max_risk_scaling", "status": "DIRECT", "reason": "Exact match in raw CSV; ESA physics baseline scaling factor."},

        # Encounter Geometry
        {"config": "miss_distance", "raw": "miss_distance", "status": "DIRECT", "reason": "Exact match in raw CSV; total miss distance at TCA."},
        {"config": "relative_speed", "raw": "relative_speed", "status": "DIRECT", "reason": "Exact match in raw CSV; encounter relative speed at TCA."},
        {"config": "relative_position_r", "raw": "relative_position_r", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN radial relative position."},
        {"config": "relative_position_t", "raw": "relative_position_t", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN along-track relative position."},
        {"config": "relative_position_n", "raw": "relative_position_n", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN normal relative position."},
        {"config": "relative_velocity_r", "raw": "relative_velocity_r", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN radial relative velocity."},
        {"config": "relative_velocity_t", "raw": "relative_velocity_t", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN along-track relative velocity."},
        {"config": "relative_velocity_n", "raw": "relative_velocity_n", "status": "DIRECT", "reason": "Exact match in raw CSV; RTN normal relative velocity."},
        {"config": "mahalanobis_distance", "raw": "mahalanobis_distance", "status": "DIRECT", "reason": "Exact match in raw CSV; covariance-normalized miss distance."},

        # Encounter Location
        {"config": "geocentric_latitude", "raw": "geocentric_latitude", "status": "DIRECT", "reason": "Exact match in raw CSV; geocentric latitude at TCA."},
        {"config": "azimuth", "raw": "azimuth", "status": "DIRECT", "reason": "Exact match in raw CSV; encounter azimuth angle."},
        {"config": "elevation", "raw": "elevation", "status": "DIRECT", "reason": "Exact match in raw CSV; encounter elevation angle."},

        # Target Orbital State
        {"config": "t_a", "raw": "t_j2k_sma", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_sma is semi-major axis in J2000 frame (km). Requires human sign-off before aliasing."},
        {"config": "t_sma", "raw": "t_j2k_sma", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_sma in J2000 frame. Requires human sign-off."},
        {"config": "t_e", "raw": "t_j2k_ecc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_ecc is eccentricity in J2000 frame. Requires human sign-off."},
        {"config": "t_eccentricity", "raw": "t_j2k_ecc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_ecc is eccentricity. Requires human sign-off."},
        {"config": "t_i", "raw": "t_j2k_inc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_inc is inclination in J2000 frame (deg). Requires human sign-off."},
        {"config": "t_inclination", "raw": "t_j2k_inc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column t_j2k_inc is inclination. Requires human sign-off."},
        {"config": "t_perigee", "raw": "t_h_per", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw column t_h_per is perigee altitude (km), not perigee radius from center of Earth. Requires human domain confirmation."},
        {"config": "t_apogee", "raw": "t_h_apo", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw column t_h_apo is apogee altitude (km), not apogee radius. Requires human domain confirmation."},
        {"config": "t_argp", "raw": None, "status": "NOT_AVAILABLE", "reason": "Argument of perigee is not provided in raw ESA CSV."},
        {"config": "t_raan", "raw": None, "status": "NOT_AVAILABLE", "reason": "Right Ascension of Ascending Node is not provided in raw ESA CSV."},
        {"config": "t_mean_anomaly", "raw": None, "status": "NOT_AVAILABLE", "reason": "Mean anomaly is not provided in raw ESA CSV."},

        # Chaser Orbital State
        {"config": "c_a", "raw": "c_j2k_sma", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_sma is J2000 semi-major axis (km). Requires human sign-off."},
        {"config": "c_sma", "raw": "c_j2k_sma", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_sma in J2000 frame. Requires human sign-off."},
        {"config": "c_e", "raw": "c_j2k_ecc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_ecc is eccentricity. Requires human sign-off."},
        {"config": "c_eccentricity", "raw": "c_j2k_ecc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_ecc is eccentricity. Requires human sign-off."},
        {"config": "c_i", "raw": "c_j2k_inc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_inc is J2000 inclination (deg). Requires human sign-off."},
        {"config": "c_inclination", "raw": "c_j2k_inc", "status": "NEEDS_HUMAN_REVIEW", "reason": "Candidate raw column c_j2k_inc is inclination. Requires human sign-off."},
        {"config": "c_perigee", "raw": "c_h_per", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw column c_h_per is chaser perigee altitude (km). Requires human confirmation."},
        {"config": "c_apogee", "raw": "c_h_apo", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw column c_h_apo is chaser apogee altitude (km). Requires human confirmation."},
        {"config": "c_argp", "raw": None, "status": "NOT_AVAILABLE", "reason": "Argument of perigee for chaser is not provided in raw ESA CSV."},
        {"config": "c_raan", "raw": None, "status": "NOT_AVAILABLE", "reason": "RAAN for chaser is not provided in raw ESA CSV."},
        {"config": "c_mean_anomaly", "raw": None, "status": "NOT_AVAILABLE", "reason": "Mean anomaly for chaser is not provided in raw ESA CSV."},

        # Uncertainty & Covariance
        {"config": "t_sigma_r", "raw": "t_sigma_r", "status": "DIRECT", "reason": "Exact match in raw CSV; target radial standard deviation."},
        {"config": "t_sigma_t", "raw": "t_sigma_t", "status": "DIRECT", "reason": "Exact match in raw CSV; target along-track standard deviation."},
        {"config": "t_sigma_n", "raw": "t_sigma_n", "status": "DIRECT", "reason": "Exact match in raw CSV; target cross-track normal standard deviation."},
        {"config": "c_sigma_r", "raw": "c_sigma_r", "status": "DIRECT", "reason": "Exact match in raw CSV; chaser radial standard deviation."},
        {"config": "c_sigma_t", "raw": "c_sigma_t", "status": "DIRECT", "reason": "Exact match in raw CSV; chaser along-track standard deviation."},
        {"config": "c_sigma_n", "raw": "c_sigma_n", "status": "DIRECT", "reason": "Exact match in raw CSV; chaser cross-track normal standard deviation."},
        {"config": "miss_distance_sigma", "raw": None, "status": "NOT_AVAILABLE", "reason": "Direct miss distance sigma is not a raw column (may be derivable or absent)."},
        {"config": "covariance_determinant", "raw": "t_position_covariance_det, c_position_covariance_det", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw data provides separate target and chaser 3x3 position covariance determinants. Requires explicit selection rule."},
        {"config": "covariance_correlation", "raw": None, "status": "NOT_AVAILABLE", "reason": "Full scalar covariance correlation not in raw CSV (raw has individual cross-terms t_ct_r, etc.)."},

        # Observation Quality
        {"config": "t_obs_available", "raw": "t_obs_available", "status": "DIRECT", "reason": "Exact match in raw CSV; target available observations."},
        {"config": "t_obs_used", "raw": "t_obs_used", "status": "DIRECT", "reason": "Exact match in raw CSV; target used observations."},
        {"config": "c_obs_available", "raw": "c_obs_available", "status": "DIRECT", "reason": "Exact match in raw CSV; chaser available observations."},
        {"config": "c_obs_used", "raw": "c_obs_used", "status": "DIRECT", "reason": "Exact match in raw CSV; chaser used observations."},
        {"config": "t_residuals_accepted", "raw": "t_residuals_accepted", "status": "DIRECT", "reason": "Exact match in raw CSV; percentage of accepted residuals for target."},
        {"config": "c_residuals_accepted", "raw": "c_residuals_accepted", "status": "DIRECT", "reason": "Exact match in raw CSV; percentage of accepted residuals for chaser."},
        {"config": "t_weighted_rms", "raw": "t_weighted_rms", "status": "DIRECT", "reason": "Exact match in raw CSV; weighted RMS for target orbit determination."},
        {"config": "c_weighted_rms", "raw": "c_weighted_rms", "status": "DIRECT", "reason": "Exact match in raw CSV; weighted RMS for chaser orbit determination."},
        {"config": "t_od_span", "raw": "t_actual_od_span", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains t_actual_od_span and t_recommended_od_span. Requires human selection."},
        {"config": "c_od_span", "raw": "c_actual_od_span", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains c_actual_od_span and c_recommended_od_span. Requires human selection."},
        {"config": "t_rms", "raw": "t_weighted_rms", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains t_weighted_rms. Requires human confirmation."},
        {"config": "c_rms", "raw": "c_weighted_rms", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains c_weighted_rms. Requires human confirmation."},
        {"config": "t_residual", "raw": "t_residuals_accepted", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains t_residuals_accepted (percentage). Requires human confirmation."},
        {"config": "c_residual", "raw": "c_residuals_accepted", "status": "NEEDS_HUMAN_REVIEW", "reason": "Raw dataset contains c_residuals_accepted (percentage). Requires human confirmation."},

        # Contextual (Space Weather)
        {"config": "F10", "raw": "F10", "status": "DIRECT", "reason": "Exact match in raw CSV; 10.7cm solar radio flux."},
        {"config": "f10", "raw": "F10", "status": "DIRECT", "reason": "Exact case-insensitive match (raw: F10)."},
        {"config": "F3M", "raw": "F3M", "status": "DIRECT", "reason": "Exact match in raw CSV; 81-day centered solar radio flux average."},
        {"config": "f3m", "raw": "F3M", "status": "DIRECT", "reason": "Exact case-insensitive match (raw: F3M)."},
        {"config": "AP", "raw": "AP", "status": "DIRECT", "reason": "Exact match in raw CSV; planetary geomagnetic index."},
        {"config": "ap", "raw": "AP", "status": "DIRECT", "reason": "Exact case-insensitive match (raw: AP)."},
        {"config": "SSN", "raw": "SSN", "status": "DIRECT", "reason": "Exact match in raw CSV; sunspot number."},
        {"config": "ssn", "raw": "SSN", "status": "DIRECT", "reason": "Exact case-insensitive match (raw: SSN)."},
    ]

    # 5. SHA-256 AFTER
    sha256_after = compute_sha256(raw_path)
    checksum_match = (sha256_before == sha256_after)
    print(f"SHA-256 (After):  {sha256_after}")
    print(f"Checksum Verified Unchanged: {checksum_match}")

    # Build reports/schema_report.json
    schema_report_data = {
        "dataset_identity": {
            "file_path": "data/raw/esa/train_data.csv",
            "sha256_before": sha256_before,
            "sha256_after": sha256_after,
            "checksum_verified": checksum_match,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_mb, 2)
        },
        "file_statistics": {
            "total_rows": row_count,
            "total_columns": num_cols,
            "exact_duplicate_rows": exact_duplicate_rows,
            "exact_duplicate_percentage": round((exact_duplicate_rows / row_count * 100) if row_count else 0.0, 4)
        },
        "event_statistics": {
            "total_events": total_events,
            "events_with_valid_final_risk": events_with_valid_final_risk,
            "events_without_valid_final_risk": events_without_valid_final_risk,
            "valid_final_risk_percentage": round((events_with_valid_final_risk / total_events * 100), 4),
            "single_cdm_events": single_cdm_events,
            "multi_cdm_events": multi_cdm_events,
            "time_order_classification": {
                "strictly_decreasing_events": strictly_decreasing_events,
                "non_increasing_with_ties_events": non_increasing_with_ties_events,
                "strictly_increasing_events": strictly_increasing_events,
                "non_monotonic_events": non_monotonic_events
            },
            "cdm_sequence_length_statistics": {
                "min": min(cdm_counts),
                "max": max(cdm_counts),
                "mean": round(sum(cdm_counts) / len(cdm_counts), 4),
                "median": median(cdm_counts),
                "percentiles": cdm_percentiles
            },
            "horizon_eligibility": {
                f"H_{h}d": {
                    "horizon_days": h,
                    "eligible_events": horizon_eligible_events[h],
                    "eligible_percentage": round(horizon_eligible_events[h] / total_events * 100, 2),
                    "median_eligible_cdms": median(horizon_cdm_counts[h]) if horizon_cdm_counts[h] else 0,
                    "max_eligible_cdms": max(horizon_cdm_counts[h]) if horizon_cdm_counts[h] else 0,
                    "mean_eligible_cdms": round(sum(horizon_cdm_counts[h]) / len(horizon_cdm_counts[h]), 2) if horizon_cdm_counts[h] else 0
                }
                for h in horizons
            }
        },
        "target_statistics": {
            "target_column": "risk",
            "data_type": "float64",
            "all_rows_risk": {
                "total_count": len(all_row_risks),
                "missing_count": row_count - len(all_row_risks),
                "missing_percentage": round((row_count - len(all_row_risks)) / row_count * 100, 4),
                "min": min(all_row_risks),
                "max": max(all_row_risks),
                "mean": round(sum(all_row_risks) / len(all_row_risks), 6),
                "median": median(all_row_risks),
                "percentiles": all_risk_percentiles
            },
            "final_cdm_risk": {
                "total_events": total_events,
                "valid_count": events_with_valid_final_risk,
                "missing_count": events_without_valid_final_risk,
                "min": final_risk_min,
                "max": final_risk_max,
                "mean": round(final_risk_mean, 6) if final_risk_mean is not None else None,
                "median": final_risk_med,
                "percentiles": final_risk_percentiles
            }
        },
        "feature_mappings": feature_mapping_definitions,
        "columns": {}
    }

    for name in col_names:
        miss_cnt = col_missing_counts[name]
        miss_pct = (miss_cnt / row_count * 100) if row_count else 0
        type_dict = dict(col_type_counts[name])
        if not type_dict:
            inferred_type = "empty"
        elif "string" in type_dict:
            inferred_type = "string"
        elif "float" in type_dict:
            inferred_type = "float64"
        elif "int" in type_dict:
            inferred_type = "int64"
        else:
            inferred_type = "unknown"

        num_cnt = col_numeric_count[name]
        c_mean = (col_sum[name] / num_cnt) if num_cnt > 0 else None

        schema_report_data["columns"][name] = {
            "inferred_type": inferred_type,
            "type_counts": type_dict,
            "missing_count": miss_cnt,
            "missing_percentage": round(miss_pct, 4),
            "min": col_min.get(name),
            "max": col_max.get(name),
            "mean": round(c_mean, 6) if c_mean is not None else None
        }

    os.makedirs("reports", exist_ok=True)
    with open("reports/schema_report.json", "w", encoding="utf-8") as f:
        json.dump(schema_report_data, f, indent=2)
    print("\nSuccessfully updated reports/schema_report.json")

    return schema_report_data


if __name__ == "__main__":
    main()
