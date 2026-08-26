"""Data quality audit, missingness profiling, duplicate check, and event statistics."""

import math
from typing import Any, Dict, List, Set, Tuple


def run_data_quality_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute comprehensive data quality audit on raw or processed CDM records.
    
    Args:
        records: List of CDM dictionaries.
        
    Returns:
        Dictionary of summary statistics, missingness counts, event sequence lengths, and target checks.
    """
    n_rows = len(records)
    if n_rows == 0:
        return {"total_rows": 0, "total_events": 0, "status": "empty"}

    # 1. Column-wise missingness
    all_keys: Set[str] = set()
    for rec in records:
        all_keys.update(rec.keys())

    missingness_counts: Dict[str, int] = {k: 0 for k in all_keys}
    events_map: Dict[str, int] = {}
    high_risk_count = 0
    risk_values: List[float] = []

    for rec in records:
        ev_id = str(rec.get("event_id", "UNKNOWN"))
        events_map[ev_id] = events_map.get(ev_id, 0) + 1

        for k in all_keys:
            val = rec.get(k)
            if val is None or val == "" or val == "NaN" or val == "None":
                missingness_counts[k] += 1

        raw_risk = rec.get("risk")
        if raw_risk is not None:
            try:
                r_val = float(raw_risk)
                if not (math.isnan(r_val) or math.isinf(r_val)):
                    risk_values.append(r_val)
                    if r_val >= -5.0:
                        high_risk_count += 1
            except (ValueError, TypeError):
                pass

    # 2. Sequence length distribution
    seq_lengths = sorted(events_map.values())
    n_events = len(events_map)
    min_len = seq_lengths[0] if seq_lengths else 0
    max_len = seq_lengths[-1] if seq_lengths else 0
    median_len = seq_lengths[n_events // 2] if seq_lengths else 0
    mean_len = sum(seq_lengths) / max(n_events, 1)

    # 3. Missingness rates
    missing_rates = {k: round(v / n_rows, 5) for k, v in missingness_counts.items()}

    return {
        "total_rows": n_rows,
        "total_unique_events": n_events,
        "sequence_lengths": {
            "min": min_len,
            "max": max_len,
            "median": median_len,
            "mean": round(mean_len, 2),
        },
        "missingness_rates": missing_rates,
        "target_risk_summary": {
            "total_risk_records": len(risk_values),
            "high_risk_count_ge_minus_5": high_risk_count,
            "min_risk": min(risk_values) if risk_values else None,
            "max_risk": max(risk_values) if risk_values else None,
        },
    }
