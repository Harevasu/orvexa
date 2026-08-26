"""Snapshot feature extraction from the latest qualifying CDM before warning cutoff."""

import math
from typing import Any, Dict, List, Optional, Tuple


def extract_snapshot_features(
    event_cdms: List[Dict[str, Any]],
    feature_cols: List[str],
    horizon_cutoff: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Extract snapshot features from the single most recent qualifying CDM satisfying time_to_tca >= horizon_cutoff.
    
    Args:
        event_cdms: List of CDM dictionaries for a single event, sorted chronologically (oldest to newest).
        feature_cols: List of permissible feature column names to extract.
        horizon_cutoff: Warning horizon H in days. Only CDMs with time_to_tca >= horizon_cutoff are eligible.
        
    Returns:
        Dictionary of snapshot features, or None if no CDM satisfies the horizon cutoff.
    """
    if not event_cdms:
        return None

    # Filter qualifying CDMs
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

    # The most recent qualifying CDM is the one with minimum time_to_tca
    # Since event_cdms is sorted oldest-to-newest, the last qualifying item is the latest
    latest_cdm = min(
        qualifying,
        key=lambda x: float(x.get("time_to_tca", 999.0))
    )

    snapshot_row: Dict[str, Any] = {}
    for col in feature_cols:
        snapshot_row[col] = latest_cdm.get(col)

    # Attach event metadata
    snapshot_row["event_id"] = latest_cdm.get("event_id")
    snapshot_row["time_to_tca"] = latest_cdm.get("time_to_tca")

    return snapshot_row


def extract_snapshot_dataset(
    events: Dict[str, List[Dict[str, Any]]],
    feature_cols: List[str],
    horizon_cutoff: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[float], List[str]]:
    """Extract snapshot feature rows, final target risk, and event IDs across all events.
    
    Args:
        events: Mapping of event_id -> list of CDMs.
        feature_cols: Permissible feature column names.
        horizon_cutoff: Horizon H in days.
        
    Returns:
        Tuple of (feature_records, target_risks, event_ids).
    """
    feature_records: List[Dict[str, Any]] = []
    target_risks: List[float] = []
    event_ids: List[str] = []

    for ev_id, cdms in events.items():
        if not cdms:
            continue

        snap = extract_snapshot_features(cdms, feature_cols, horizon_cutoff=horizon_cutoff)
        if snap is None:
            continue

        # Final target is always the risk of the final CDM of the event
        final_cdm = cdms[-1]
        try:
            target_risk = float(final_cdm["risk"])
        except (KeyError, ValueError, TypeError):
            continue

        feature_records.append(snap)
        target_risks.append(target_risk)
        event_ids.append(ev_id)

    return feature_records, target_risks, event_ids
