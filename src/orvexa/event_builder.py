"""Event builder: groups CDMs by event_id, extracts final target, and constructs horizon prefixes."""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class EventRecord:
    """Represents a single conjunction event prefix view at horizon H."""
    event_id: str
    horizon_days: float
    target_risk: float
    anchor_time_to_tca: float
    sequence_length: int
    prefix_duration_days: float
    row_indices: List[int]


def build_event_views(raw_df: Any, horizon_days: float) -> Any:
    """Construct leakage-free event views for a given warning horizon cutoff."""
    raise NotImplementedError("Event building not yet implemented.")
