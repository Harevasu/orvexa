"""Event-aware chronological dataset splitting (70% train / 15% validation / 15% test).

Enforces strict zero-leakage constraints:
- All CDMs belonging to a single event_id are placed entirely within one split partition.
- Chronological ordering is preserved: earliest events for training, subsequent for validation, final for test.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class SplitManifest:
    """Stores exact event partitions, counts, and boundary indices."""

    train_event_ids: List[str]
    val_event_ids: List[str]
    test_event_ids: List[str]
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train_set = set(self.train_event_ids)
        val_set = set(self.val_event_ids)
        test_set = set(self.test_event_ids)

        if not train_set.isdisjoint(val_set):
            overlap = train_set.intersection(val_set)
            raise ValueError(f"Train and Validation split overlap detected: {len(overlap)} shared event IDs.")
        if not train_set.isdisjoint(test_set):
            overlap = train_set.intersection(test_set)
            raise ValueError(f"Train and Test split overlap detected: {len(overlap)} shared event IDs.")
        if not val_set.isdisjoint(test_set):
            overlap = val_set.intersection(test_set)
            raise ValueError(f"Validation and Test split overlap detected: {len(overlap)} shared event IDs.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "train_event_ids": self.train_event_ids,
            "val_event_ids": self.val_event_ids,
            "test_event_ids": self.test_event_ids,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "counts": {
                "train": len(self.train_event_ids),
                "val": len(self.val_event_ids),
                "test": len(self.test_event_ids),
                "total": len(self.train_event_ids) + len(self.val_event_ids) + len(self.test_event_ids),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitManifest":
        """Load manifest from dictionary."""
        return cls(
            train_event_ids=data["train_event_ids"],
            val_event_ids=data["val_event_ids"],
            test_event_ids=data["test_event_ids"],
            train_ratio=data.get("train_ratio", 0.70),
            val_ratio=data.get("val_ratio", 0.15),
            test_ratio=data.get("test_ratio", 0.15),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[Path, str]) -> None:
        """Save split manifest to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "SplitManifest":
        """Load split manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def make_chronological_splits(
    event_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitManifest:
    """Partition ordered unique event IDs into disjoint chronological train/val/test splits.
    
    Args:
        event_ids: Chronologically ordered list of unique event IDs.
        train_ratio: Fraction of earliest events for training (default: 0.70).
        val_ratio: Fraction of intermediate events for validation (default: 0.15).
        
    Returns:
        SplitManifest containing disjoint partition lists.
    """
    if not event_ids:
        raise ValueError("Cannot partition an empty list of event IDs.")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered_unique: List[str] = []
    for ev in event_ids:
        if ev not in seen:
            seen.add(ev)
            ordered_unique.append(ev)

    n_total = len(ordered_unique)
    n_train = int(math.floor(n_total * train_ratio))
    n_val = int(math.floor(n_total * val_ratio))
    
    train_ids = ordered_unique[:n_train]
    val_ids = ordered_unique[n_train : n_train + n_val]
    test_ids = ordered_unique[n_train + n_val :]

    test_ratio = 1.0 - train_ratio - val_ratio

    return SplitManifest(
        train_event_ids=train_ids,
        val_event_ids=val_ids,
        test_event_ids=test_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        metadata={"total_unique_events": n_total},
    )


def assert_split_disjointness(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> bool:
    """Assert zero event ID overlap across train, validation, and test partitions."""
    return train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)


@dataclass
class Phase5SplitManifest:
    """Stores exact 4-way Phase 5 event partitions (Train, Val, Calibration, InternalTest)."""

    train_event_ids: List[str]
    val_event_ids: List[str]
    cal_event_ids: List[str]
    test_event_ids: List[str]
    quarantined_test_event_ids: List[str] = field(default_factory=list)
    train_ratio: float = 0.60
    val_ratio: float = 0.15
    cal_ratio: float = 0.10
    test_ratio: float = 0.15
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tr_set = set(self.train_event_ids)
        va_set = set(self.val_event_ids)
        ca_set = set(self.cal_event_ids)
        te_set = set(self.test_event_ids)
        quarantine_set = set(self.quarantined_test_event_ids)

        # Internal 4-way pairwise disjointness
        pairs = [
            ("Train", "Validation", tr_set, va_set),
            ("Train", "Calibration", tr_set, ca_set),
            ("Train", "InternalTest", tr_set, te_set),
            ("Validation", "Calibration", va_set, ca_set),
            ("Validation", "InternalTest", va_set, te_set),
            ("Calibration", "InternalTest", ca_set, te_set),
        ]
        for name1, name2, s1, s2 in pairs:
            if not s1.isdisjoint(s2):
                overlap = s1.intersection(s2)
                raise ValueError(f"Phase 5 {name1} and {name2} overlap detected: {len(overlap)} shared event IDs.")

        # Historical quarantined test disjointness
        if quarantine_set:
            for pname, pset in [("Train", tr_set), ("Validation", va_set), ("Calibration", ca_set), ("InternalTest", te_set)]:
                if not pset.isdisjoint(quarantine_set):
                    overlap = pset.intersection(quarantine_set)
                    raise ValueError(f"CRITICAL LEAKAGE: Phase 5 {pname} overlaps with historical quarantined test partition ({len(overlap)} shared event IDs).")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Phase 5 manifest to dictionary."""
        return {
            "train_event_ids": self.train_event_ids,
            "val_event_ids": self.val_event_ids,
            "cal_event_ids": self.cal_event_ids,
            "test_event_ids": self.test_event_ids,
            "quarantined_test_event_ids": self.quarantined_test_event_ids,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "cal_ratio": self.cal_ratio,
            "test_ratio": self.test_ratio,
            "counts": {
                "train": len(self.train_event_ids),
                "validation": len(self.val_event_ids),
                "calibration": len(self.cal_event_ids),
                "internal_test": len(self.test_event_ids),
                "total_eligible": len(self.train_event_ids) + len(self.val_event_ids) + len(self.cal_event_ids) + len(self.test_event_ids),
                "quarantined_test": len(self.quarantined_test_event_ids),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase5SplitManifest":
        """Load Phase 5 manifest from dictionary."""
        return cls(
            train_event_ids=data["train_event_ids"],
            val_event_ids=data["val_event_ids"],
            cal_event_ids=data["cal_event_ids"],
            test_event_ids=data["test_event_ids"],
            quarantined_test_event_ids=data.get("quarantined_test_event_ids", []),
            train_ratio=data.get("train_ratio", 0.60),
            val_ratio=data.get("val_ratio", 0.15),
            cal_ratio=data.get("cal_ratio", 0.10),
            test_ratio=data.get("test_ratio", 0.15),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[Path, str]) -> None:
        """Save Phase 5 split manifest to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "Phase5SplitManifest":
        """Load Phase 5 split manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def make_phase5_splits(
    eligible_event_ids: List[str],
    quarantined_test_ids: Optional[List[str]] = None,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    cal_ratio: float = 0.10,
) -> Phase5SplitManifest:
    """Partition ordered eligible event IDs into 4 disjoint chronological Phase 5 partitions.
    
    Args:
        eligible_event_ids: Chronologically ordered list of unique eligible event IDs.
        quarantined_test_ids: Historical locked test event IDs that must never appear in any partition.
        train_ratio: Fraction for training (default: 0.60).
        val_ratio: Fraction for validation / model selection (default: 0.15).
        cal_ratio: Fraction for post-hoc uncertainty calibration (default: 0.10).
        
    Returns:
        Phase5SplitManifest with verified disjoint partitions and quarantine assertions.
    """
    if not eligible_event_ids:
        raise ValueError("Cannot partition an empty list of eligible event IDs.")

    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered_unique: List[str] = []
    for ev in eligible_event_ids:
        if ev not in seen:
            seen.add(ev)
            ordered_unique.append(ev)

    n_total = len(ordered_unique)
    n_tr = int(math.floor(n_total * train_ratio))
    n_va = int(math.floor(n_total * val_ratio))
    n_ca = int(math.floor(n_total * cal_ratio))
    
    train_ids = ordered_unique[:n_tr]
    val_ids = ordered_unique[n_tr : n_tr + n_va]
    cal_ids = ordered_unique[n_tr + n_va : n_tr + n_va + n_ca]
    test_ids = ordered_unique[n_tr + n_va + n_ca :]

    test_ratio = 1.0 - train_ratio - val_ratio - cal_ratio

    return Phase5SplitManifest(
        train_event_ids=train_ids,
        val_event_ids=val_ids,
        cal_event_ids=cal_ids,
        test_event_ids=test_ids,
        quarantined_test_event_ids=quarantined_test_ids or [],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        cal_ratio=cal_ratio,
        test_ratio=test_ratio,
        metadata={"total_eligible_events": n_total},
    )

