"""Event builder module for ORVEXA.

Groups CDMs by event_id, extracts final target risk, constructs
horizon prefixes (H in {2, 3, 5} days), and enforces strict leakage controls.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


# Strict Direct feature list (34 features) from reports/schema_report.json
DIRECT_FEATURE_COLUMNS = [
    "time_to_tca",
    "c_object_type",
    "max_risk_estimate",
    "max_risk_scaling",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "mahalanobis_distance",
    "geocentric_latitude",
    "azimuth",
    "elevation",
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "t_obs_available",
    "t_obs_used",
    "c_obs_available",
    "c_obs_used",
    "t_residuals_accepted",
    "c_residuals_accepted",
    "t_weighted_rms",
    "c_weighted_rms",
    "F10",
    "F3M",
    "AP",
    "SSN",
]

EXCLUDED_IDENTIFIERS = ["event_id", "mission_id"]
EXCLUDED_TARGETS = ["risk"]

NEEDS_HUMAN_REVIEW_FEATURES = [
    "t_a", "t_sma", "t_e", "t_eccentricity", "t_i", "t_inclination",
    "t_perigee", "t_apogee", "c_a", "c_sma", "c_e", "c_eccentricity",
    "c_i", "c_inclination", "c_perigee", "c_apogee",
    "covariance_determinant", "t_od_span", "c_od_span",
    "t_rms", "c_rms", "t_residual", "c_residual"
]

NOT_AVAILABLE_FEATURES = [
    "t_argp", "t_raan", "t_mean_anomaly",
    "c_argp", "c_raan", "c_mean_anomaly",
    "miss_distance_sigma", "covariance_correlation"
]


@dataclass
class EventHorizonRecord:
    """Represents a single conjunction event view at warning horizon H."""
    event_id: str
    horizon_days: float
    sequence_length: int
    earliest_time_to_tca: float
    anchor_time_to_tca: float
    final_risk: float
    cdms: List[Dict[str, Any]]


def compute_file_sha256(file_path: str) -> str:
    """Deterministically compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_event_prefixes_from_raw(
    raw_rows_by_event: Dict[str, List[Dict[str, str]]],
    horizon_days: float,
    feature_columns: List[str] = DIRECT_FEATURE_COLUMNS,
) -> Tuple[List[EventHorizonRecord], Dict[str, Any]]:
    """Construct leakage-free event views for a given warning horizon cutoff.

    Parameters
    ----------
    raw_rows_by_event : dict
        Mapping from event_id to list of raw CDM row dictionaries in original order.
    horizon_days : float
        Warning cutoff in days (e.g. 2.0, 3.0, 5.0).
    feature_columns : list of str
        List of whitelisted DIRECT feature column names.

    Returns
    -------
    tuple of (list of EventHorizonRecord, dict of metadata)
    """
    eligible_events: List[EventHorizonRecord] = []
    skipped_zero_cdms = 0
    total_cdms_retained = 0

    for ev_id, cdm_list in raw_rows_by_event.items():
        if not cdm_list:
            continue

        # 1. Identify final CDM (minimum time_to_tca) to extract target label
        final_cdm = min(cdm_list, key=lambda r: float(r["time_to_tca"]))
        final_risk = float(final_cdm["risk"])

        # 2. Filter qualifying CDMs for this horizon: time_to_tca >= horizon_days
        # Order is strictly preserved (decreasing time_to_tca: earliest warning -> anchor)
        qualifying_cdms = []
        for cdm in cdm_list:
            t_tca = float(cdm["time_to_tca"])
            if t_tca >= horizon_days:
                # Extract only DIRECT features
                feat_dict = {col: cdm.get(col, "") for col in feature_columns}
                qualifying_cdms.append(feat_dict)

        if not qualifying_cdms:
            skipped_zero_cdms += 1
            continue

        seq_len = len(qualifying_cdms)
        earliest_tca = float(qualifying_cdms[0]["time_to_tca"])
        anchor_tca = float(qualifying_cdms[-1]["time_to_tca"])
        total_cdms_retained += seq_len

        eligible_events.append(
            EventHorizonRecord(
                event_id=ev_id,
                horizon_days=horizon_days,
                sequence_length=seq_len,
                earliest_time_to_tca=earliest_tca,
                anchor_time_to_tca=anchor_tca,
                final_risk=final_risk,
                cdms=qualifying_cdms,
            )
        )

    stats = {
        "horizon_days": horizon_days,
        "total_events_in_source": len(raw_rows_by_event),
        "eligible_events_count": len(eligible_events),
        "skipped_zero_cdms_count": skipped_zero_cdms,
        "total_cdms_retained": total_cdms_retained,
    }

    return eligible_events, stats
