"""Engineered temporal-summary feature extraction (deltas, rates of change, statistical aggregates).

Extracts multi-step dynamics across the qualifying prefix sequence (time_to_tca >= H).
Guarantees zero leakage: no future CDMs (time_to_tca < H) or targets are ever used.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# Key numeric columns for temporal delta and rate extraction
CORE_TEMPORAL_NUMERIC_COLS = [
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
    "mahalanobis_distance",
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "t_obs_used",
    "c_obs_used",
    "t_weighted_rms",
    "c_weighted_rms",
    "max_risk_estimate",
    "F10",
    "AP",
]


def extract_temporal_summary_features(
    event_cdms: List[Dict[str, Any]],
    numeric_cols: Optional[List[str]] = None,
    horizon_cutoff: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Extract temporal summary features and trajectory dynamics from qualifying CDMs.
    
    Args:
        event_cdms: List of CDM dictionaries for an event, chronologically ordered (oldest to newest).
        numeric_cols: Subset of numeric feature columns to summarize (defaults to CORE_TEMPORAL_NUMERIC_COLS).
        horizon_cutoff: Warning horizon H in days. Only CDMs with time_to_tca >= horizon_cutoff are eligible.
        
    Returns:
        Dictionary of aggregated temporal features, or None if no qualifying CDMs exist.
    """
    if not event_cdms:
        return None

    cols = list(numeric_cols or CORE_TEMPORAL_NUMERIC_COLS)

    # 1. Filter qualifying CDMs (strictly time_to_tca >= horizon_cutoff)
    qualifying: List[Dict[str, Any]] = []
    for cdm in event_cdms:
        if horizon_cutoff is not None:
            try:
                tt_tca = float(cdm.get("time_to_tca", -1.0))
                if tt_tca < horizon_cutoff:
                    continue
            except (ValueError, TypeError):
                continue
        qualifying.append(cdm)

    if not qualifying:
        return None

    # Qualifying CDMs ordered oldest to newest
    first_cdm = qualifying[0]
    latest_cdm = qualifying[-1]
    n_cdms = len(qualifying)

    try:
        t_first = float(first_cdm.get("time_to_tca", 0.0))
        t_latest = float(latest_cdm.get("time_to_tca", 0.0))
        dt_days = max(t_first - t_latest, 0.0)
    except (ValueError, TypeError):
        dt_days = 0.0

    temporal_row: Dict[str, Any] = {
        "event_id": latest_cdm.get("event_id"),
        "time_to_tca": latest_cdm.get("time_to_tca"),
        "c_object_type": latest_cdm.get("c_object_type", "UNKNOWN"),
        "cdm_count": float(n_cdms),
        "sequence_time_span_days": dt_days,
    }

    # 2. Extract dynamics for each numeric feature
    dt_safe = max(dt_days, 0.01)

    for col in cols:
        vals: List[float] = []
        for cdm in qualifying:
            raw = cdm.get(col)
            if raw is not None:
                try:
                    fval = float(raw)
                    if not (math.isnan(fval) or math.isinf(fval)):
                        vals.append(fval)
                except (ValueError, TypeError):
                    pass

        if not vals:
            # All values missing
            temporal_row[f"{col}__latest"] = None
            temporal_row[f"{col}__first"] = None
            temporal_row[f"{col}__delta"] = 0.0
            temporal_row[f"{col}__rate"] = 0.0
            temporal_row[f"{col}__min"] = None
            temporal_row[f"{col}__max"] = None
            temporal_row[f"{col}__mean"] = None
        else:
            val_first = vals[0]
            val_latest = vals[-1]
            delta = val_latest - val_first
            rate = delta / dt_safe if n_cdms > 1 else 0.0

            temporal_row[f"{col}__latest"] = val_latest
            temporal_row[f"{col}__first"] = val_first
            temporal_row[f"{col}__delta"] = delta
            temporal_row[f"{col}__rate"] = rate
            temporal_row[f"{col}__min"] = min(vals)
            temporal_row[f"{col}__max"] = max(vals)
            temporal_row[f"{col}__mean"] = sum(vals) / len(vals)

    # 3. Covariance contraction ratios
    def _ratio(latest_k: str, first_k: str) -> float:
        try:
            l_val = float(latest_cdm.get(latest_k, 1.0))
            f_val = float(first_cdm.get(first_k, 1.0))
            if f_val > 1e-6:
                return l_val / f_val
            return 1.0
        except (ValueError, TypeError):
            return 1.0

    temporal_row["t_sigma_r_shrinkage_ratio"] = _ratio("t_sigma_r", "t_sigma_r")
    temporal_row["c_sigma_r_shrinkage_ratio"] = _ratio("c_sigma_r", "c_sigma_r")

    return temporal_row


def extract_temporal_dataset(
    events: Dict[str, List[Dict[str, Any]]],
    numeric_cols: Optional[List[str]] = None,
    horizon_cutoff: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
    """Extract temporal-summary features and final targets across all events.
    
    Args:
        events: Mapping of event_id -> list of CDMs.
        numeric_cols: Subset of numeric features to summarize.
        horizon_cutoff: Warning horizon H in days.
        
    Returns:
        Tuple of (feature_records, target_risks, event_ids).
    """
    feature_records: List[Dict[str, Any]] = []
    target_risks: List[float] = []
    event_ids: List[str] = []

    for ev_id, cdms in events.items():
        if not cdms:
            continue

        temp_row = extract_temporal_summary_features(
            cdms, numeric_cols=numeric_cols, horizon_cutoff=horizon_cutoff
        )
        if temp_row is None:
            continue

        final_cdm = cdms[-1]
        try:
            target_risk = float(final_cdm["risk"])
        except (KeyError, ValueError, TypeError):
            continue

        feature_records.append(temp_row)
        target_risks.append(target_risk)
        event_ids.append(ev_id)

    return feature_records, target_risks, event_ids
