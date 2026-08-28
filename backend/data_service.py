"""Data service for managing non-sealed demo conjunction events.

Strict Quarantine Rule:
Only Phase 5 Validation (1,529 events) and Calibration (1,008 events) partitions
are loaded into memory. Quarantined Internal Test (1,677 events) and Historical
Master Test (1,974 events) are strictly rejected and never loaded for demo inference.
"""

from collections import defaultdict
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from backend.schemas import CDMRecord, EventDetailResponse, EventSummary


class ConjunctionDataService:
    """Manages allowed demo conjunction events from non-sealed Phase 5 splits."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent
        self.split_manifest_path = (
            self.workspace_root / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"
        )
        self.raw_data_path = self.workspace_root / "data" / "raw" / "esa" / "train_data.csv"

        self.allowed_event_ids: Set[str] = set()
        self.val_event_ids: Set[str] = set()
        self.cal_event_ids: Set[str] = set()
        self.quarantined_test_ids: Set[str] = set()

        # In-memory storage of allowed events: event_id -> List[Dict[str, Any]] sorted oldest to newest
        self._events_by_id: Dict[str, List[Dict[str, Any]]] = {}
        self._summaries_by_id: Dict[str, EventSummary] = {}
        self._is_loaded: bool = False

    def load_data(self) -> None:
        """Load manifest and allowed non-sealed event records into memory."""
        if self._is_loaded:
            return

        if not self.split_manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {self.split_manifest_path}")

        with open(self.split_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.val_event_ids = set(str(x) for x in manifest_data.get("val_event_ids", []))
        self.cal_event_ids = set(str(x) for x in manifest_data.get("cal_event_ids", []))
        self.quarantined_test_ids = set(str(x) for x in manifest_data.get("test_event_ids", []))

        # Allowed set is STRICTLY validation and calibration
        self.allowed_event_ids = self.val_event_ids.union(self.cal_event_ids)

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Missing raw ESA data: {self.raw_data_path}")

        # Read raw CSV
        df = pd.read_csv(self.raw_data_path)
        # Convert integer event IDs to string for matching
        allowed_int_or_str = set()
        for x in self.allowed_event_ids:
            try:
                allowed_int_or_str.add(int(x))
            except ValueError:
                pass
            allowed_int_or_str.add(x)

        # Filter strictly to allowed non-sealed events
        df_allowed = df[df["event_id"].isin(allowed_int_or_str)].copy()
        df_allowed["event_id_str"] = df_allowed["event_id"].astype(str)

        # Sort chronologically by event_id, then time_to_tca descending (oldest to newest approach)
        df_allowed = df_allowed.sort_values(["event_id_str", "time_to_tca"], ascending=[True, False])

        records = df_allowed.to_dict(orient="records")
        events_dict: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            ev_id = r["event_id_str"]
            events_dict[ev_id].append(r)

        self._events_by_id = dict(events_dict)

        # Build summaries
        for ev_id, cdms in self._events_by_id.items():
            split_name = "validation" if ev_id in self.val_event_ids else "calibration"
            ttcas = [float(c.get("time_to_tca", 0.0)) for c in cdms if c.get("time_to_tca") is not None]
            earliest = max(ttcas) if ttcas else 0.0
            latest = min(ttcas) if ttcas else 0.0
            final_risk = float(cdms[-1].get("risk", -30.0))

            obj_types = [str(c.get("c_object_type", "UNKNOWN")).strip().upper() for c in cdms]
            prim_obj = obj_types[-1] if obj_types else "UNKNOWN"
            if not prim_obj or prim_obj in ("NAN", "NONE", "NULL"):
                prim_obj = "UNKNOWN"

            miss_dists = [
                float(c["miss_distance"])
                for c in cdms
                if "miss_distance" in c and pd.notna(c["miss_distance"])
            ]
            min_miss = min(miss_dists) if miss_dists else None

            mahal = [
                float(c["mahalanobis_distance"])
                for c in cdms
                if "mahalanobis_distance" in c and pd.notna(c["mahalanobis_distance"])
            ]
            max_mahal = max(mahal) if mahal else None

            self._summaries_by_id[ev_id] = EventSummary(
                event_id=ev_id,
                split=split_name,
                total_cdms=len(cdms),
                earliest_time_to_tca=round(earliest, 3),
                latest_time_to_tca=round(latest, 3),
                target_final_risk=round(final_risk, 4),
                primary_object_type=prim_obj,
                min_miss_distance=round(min_miss, 2) if min_miss is not None else None,
                max_mahalanobis=round(max_mahal, 2) if max_mahal is not None else None,
                qualifies_h2=earliest >= 2.0,
                qualifies_h3=earliest >= 3.0,
                qualifies_h5=earliest >= 5.0,
                qualifies_h6=earliest >= 6.0,
            )

        self._is_loaded = True

    def get_event_cdms(self, event_id: str) -> List[Dict[str, Any]]:
        """Retrieve raw CDMs for an allowed event."""
        if not self._is_loaded:
            self.load_data()

        ev_id = str(event_id)
        if ev_id in self.quarantined_test_ids:
            raise PermissionError(
                f"Quarantine violation: Event '{ev_id}' belongs to the quarantined Phase 5 Internal Test set."
            )

        if ev_id not in self._events_by_id:
            raise KeyError(f"Event ID '{ev_id}' not found in allowed non-sealed demo dataset.")

        return self._events_by_id[ev_id]

    def get_event_summary(self, event_id: str) -> EventSummary:
        """Get summary for a single event."""
        if not self._is_loaded:
            self.load_data()

        ev_id = str(event_id)
        if ev_id in self.quarantined_test_ids:
            raise PermissionError(
                f"Quarantine violation: Event '{ev_id}' belongs to the quarantined Phase 5 Internal Test set."
            )

        if ev_id not in self._summaries_by_id:
            raise KeyError(f"Event ID '{ev_id}' not found in allowed non-sealed demo dataset.")

        return self._summaries_by_id[ev_id]

    def list_events(
        self,
        search: Optional[str] = None,
        split: Optional[str] = None,
        min_risk: Optional[float] = None,
        horizon_qual: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[EventSummary], int]:
        """List and filter allowed demo events with pagination."""
        if not self._is_loaded:
            self.load_data()

        results = list(self._summaries_by_id.values())

        if search:
            s = search.strip().lower()
            results = [r for r in results if s in r.event_id.lower() or s in r.primary_object_type.lower()]

        if split:
            sp = split.strip().lower()
            results = [r for r in results if r.split.lower() == sp]

        if min_risk is not None:
            results = [r for r in results if r.target_final_risk >= min_risk]

        if horizon_qual:
            hq = horizon_qual.upper()
            if hq == "H2":
                results = [r for r in results if r.qualifies_h2]
            elif hq == "H3":
                results = [r for r in results if r.qualifies_h3]
            elif hq == "H5":
                results = [r for r in results if r.qualifies_h5]
            elif hq == "H6":
                results = [r for r in results if r.qualifies_h6]

        total_count = len(results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_results = results[start_idx:end_idx]

        return paged_results, total_count

    def get_event_detail(self, event_id: str) -> EventDetailResponse:
        """Get full CDM timeline and metadata for Event Detail page."""
        cdms_raw = self.get_event_cdms(event_id)
        summary = self.get_event_summary(event_id)

        cdm_records: List[CDMRecord] = []
        for c in cdms_raw:
            cdm_records.append(
                CDMRecord(
                    time_to_tca=float(c.get("time_to_tca", 0.0)),
                    risk=float(c["risk"]) if pd.notna(c.get("risk")) else None,
                    c_object_type=str(c.get("c_object_type", "UNKNOWN")),
                    miss_distance=float(c["miss_distance"]) if pd.notna(c.get("miss_distance")) else None,
                    relative_speed=float(c["relative_speed"]) if pd.notna(c.get("relative_speed")) else None,
                    relative_position_r=float(c["relative_position_r"]) if pd.notna(c.get("relative_position_r")) else None,
                    relative_position_t=float(c["relative_position_t"]) if pd.notna(c.get("relative_position_t")) else None,
                    relative_position_n=float(c["relative_position_n"]) if pd.notna(c.get("relative_position_n")) else None,
                    relative_velocity_r=float(c["relative_velocity_r"]) if pd.notna(c.get("relative_velocity_r")) else None,
                    relative_velocity_t=float(c["relative_velocity_t"]) if pd.notna(c.get("relative_velocity_t")) else None,
                    relative_velocity_n=float(c["relative_velocity_n"]) if pd.notna(c.get("relative_velocity_n")) else None,
                    mahalanobis_distance=float(c["mahalanobis_distance"]) if pd.notna(c.get("mahalanobis_distance")) else None,
                    t_sigma_r=float(c["t_sigma_r"]) if pd.notna(c.get("t_sigma_r")) else None,
                    t_sigma_t=float(c["t_sigma_t"]) if pd.notna(c.get("t_sigma_t")) else None,
                    t_sigma_n=float(c["t_sigma_n"]) if pd.notna(c.get("t_sigma_n")) else None,
                    c_sigma_r=float(c["c_sigma_r"]) if pd.notna(c.get("c_sigma_r")) else None,
                    c_sigma_t=float(c["c_sigma_t"]) if pd.notna(c.get("c_sigma_t")) else None,
                    c_sigma_n=float(c["c_sigma_n"]) if pd.notna(c.get("c_sigma_n")) else None,
                    t_obs_used=float(c["t_obs_used"]) if pd.notna(c.get("t_obs_used")) else None,
                    c_obs_used=float(c["c_obs_used"]) if pd.notna(c.get("c_obs_used")) else None,
                    F10=float(c["F10"]) if pd.notna(c.get("F10")) else None,
                    AP=float(c["AP"]) if pd.notna(c.get("AP")) else None,
                    SSN=float(c["SSN"]) if pd.notna(c.get("SSN")) else None,
                )
            )

        return EventDetailResponse(
            event_id=event_id,
            split=summary.split,
            total_cdms=summary.total_cdms,
            target_final_risk=summary.target_final_risk,
            primary_object_type=summary.primary_object_type,
            cdms=cdm_records,
        )


# Global singleton instance
data_service = ConjunctionDataService()
