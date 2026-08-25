"""Event-aware chronological dataset splitting (70% train / 15% validation / 15% test)."""

from dataclasses import dataclass
from typing import Any, List, Set


@dataclass
class SplitManifest:
    """Stores exact event partitions and boundary timestamps."""
    train_event_ids: List[str]
    val_event_ids: List[str]
    test_event_ids: List[str]
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def make_chronological_splits(events_df: Any, train_ratio: float = 0.70, val_ratio: float = 0.15) -> SplitManifest:
    """Create disjoint event-level partitions ordered chronologically."""
    raise NotImplementedError("Chronological splitting not yet implemented.")


def assert_split_disjointness(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> bool:
    """Assert zero event ID overlap across train, validation, and test partitions."""
    return train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
