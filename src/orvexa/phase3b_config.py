"""ORVEXA Phase 3B Experiment Configuration and Matrix Definitions.

Defines the controlled isolated intervention matrix:
- M0: Frozen Phase 2B TCN baseline (34 channels, linear scaling, zeroed c_object_type, no dt).
- M1: Categorical correction ONLY (37 channels, one-hot c_object_type, linear scaling, no dt).
- M2: Covariance transformation ONLY (34 channels, log10 sigma/mahalanobis scaling, zeroed c_object_type, no dt).
- M3: Delta-t ONLY (35 channels, explicit inter-arrival time dt, linear scaling, zeroed c_object_type).
- M4: M1 + M2 (Categorical + Covariance log10, 37 channels, no dt) [Future Combination].
- M5: M1 + M2 + M3 (Categorical + Covariance log10 + dt, 38 channels) [Future Combination].

Enforces strict zero-leakage, training-only fitting, and deterministic channel orders.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from orvexa.event_builder import DIRECT_FEATURE_COLUMNS


# Canonical raw categories for c_object_type
RAW_OBJECT_TYPE_CATEGORIES = [
    "DEBRIS",
    "PAYLOAD",
    "ROCKET BODY",
    "UNKNOWN",
]

# One-hot encoded column names for M1/M4/M5
OBJECT_TYPE_ONE_HOT_CHANNELS = [
    "c_object_type__DEBRIS",
    "c_object_type__PAYLOAD",
    "c_object_type__ROCKET_BODY",
    "c_object_type__UNKNOWN",
]

# Features subject to monotonic log10 transformation in M2/M4/M5
COVARIANCE_LOG10_FEATURE_COLUMNS = [
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "mahalanobis_distance",
]


class ExperimentID(str, Enum):
    """Phase 3B experiment identifiers."""
    M0 = "M0"  # Baseline Phase 2B (Frozen reference)
    M1 = "M1"  # Categorical correction ONLY
    M2 = "M2"  # Covariance log10 transform ONLY
    M3 = "M3"  # Explicit Delta-t ONLY
    M4 = "M4"  # Combination: M1 + M2 (Categorical + Covariance)
    M5 = "M5"  # Full Combination: M1 + M2 + M3 (Categorical + Covariance + Delta-t)


@dataclass
class Phase3BExperimentConfig:
    """Configuration for a specific Phase 3B controlled experiment."""
    experiment_id: ExperimentID
    description: str
    include_categorical_one_hot: bool = False
    include_covariance_log10: bool = False
    include_delta_t: bool = False
    max_seq_len: int = 23
    kernel_size: int = 3
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    channels: List[int] = field(default_factory=lambda: [64, 64, 128])
    dropout: float = 0.15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    epochs: int = 50
    patience: int = 15
    huber_delta: float = 1.0

    @property
    def feature_columns(self) -> List[str]:
        """Compute exact ordered list of input feature channels."""
        channels: List[str] = []

        # Channel 0: time_to_tca
        channels.append("time_to_tca")

        # Optional Channel 1: delta_t (if M3 or M5)
        if self.include_delta_t:
            channels.append("delta_t")

        # Categorical Channel(s):
        if self.include_categorical_one_hot:
            channels.extend(OBJECT_TYPE_ONE_HOT_CHANNELS)
        else:
            # Broken/legacy channel matching M0 behavior
            channels.append("c_object_type")

        # Remaining numeric direct features (index 2..33 from DIRECT_FEATURE_COLUMNS)
        remaining_direct = [
            c for c in DIRECT_FEATURE_COLUMNS
            if c not in ("time_to_tca", "c_object_type")
        ]
        channels.extend(remaining_direct)

        return channels

    @property
    def n_channels(self) -> int:
        """Total input channel count."""
        return len(self.feature_columns)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "experiment_id": self.experiment_id.value,
            "description": self.description,
            "include_categorical_one_hot": self.include_categorical_one_hot,
            "include_covariance_log10": self.include_covariance_log10,
            "include_delta_t": self.include_delta_t,
            "n_channels": self.n_channels,
            "feature_columns": self.feature_columns,
            "max_seq_len": self.max_seq_len,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "channels": self.channels,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "huber_delta": self.huber_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase3BExperimentConfig":
        """Reconstruct configuration from dictionary."""
        return cls(
            experiment_id=ExperimentID(data["experiment_id"]),
            description=data["description"],
            include_categorical_one_hot=data.get("include_categorical_one_hot", False),
            include_covariance_log10=data.get("include_covariance_log10", False),
            include_delta_t=data.get("include_delta_t", False),
            max_seq_len=data.get("max_seq_len", 23),
            kernel_size=data.get("kernel_size", 3),
            dilations=data.get("dilations", [1, 2, 4]),
            channels=data.get("channels", [64, 64, 128]),
            dropout=data.get("dropout", 0.15),
            learning_rate=data.get("learning_rate", 0.001),
            weight_decay=data.get("weight_decay", 0.0001),
            batch_size=data.get("batch_size", 64),
            epochs=data.get("epochs", 50),
            patience=data.get("patience", 15),
            huber_delta=data.get("huber_delta", 1.0),
        )


def get_experiment_config(exp_id: Union[ExperimentID, str]) -> Phase3BExperimentConfig:
    """Factory for standard Phase 3B controlled experiment configurations."""
    if isinstance(exp_id, str):
        exp_id = ExperimentID(exp_id.upper())

    if exp_id == ExperimentID.M0:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M0,
            description="M0: Frozen Phase 2B Baseline (34 channels, linear z-score, zeroed c_object_type, no dt)",
            include_categorical_one_hot=False,
            include_covariance_log10=False,
            include_delta_t=False,
        )
    elif exp_id == ExperimentID.M1:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M1,
            description="M1: Categorical Correction ONLY (37 channels, one-hot c_object_type, linear z-score, no dt)",
            include_categorical_one_hot=True,
            include_covariance_log10=False,
            include_delta_t=False,
        )
    elif exp_id == ExperimentID.M2:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M2,
            description="M2: Covariance Log10 Transform ONLY (34 channels, log10 sigmas/mahalanobis, zeroed c_object_type, no dt)",
            include_categorical_one_hot=False,
            include_covariance_log10=True,
            include_delta_t=False,
        )
    elif exp_id == ExperimentID.M3:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M3,
            description="M3: Explicit Delta-t ONLY (35 channels, inter-arrival time dt, linear z-score, zeroed c_object_type)",
            include_categorical_one_hot=False,
            include_covariance_log10=False,
            include_delta_t=True,
        )
    elif exp_id == ExperimentID.M4:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M4,
            description="M4: Combination M1 + M2 (Categorical Correction + Covariance Log10, 37 channels, no dt)",
            include_categorical_one_hot=True,
            include_covariance_log10=True,
            include_delta_t=False,
        )
    elif exp_id == ExperimentID.M5:
        return Phase3BExperimentConfig(
            experiment_id=ExperimentID.M5,
            description="M5: Full Combination M1 + M2 + M3 (Categorical + Covariance Log10 + Delta-t, 38 channels)",
            include_categorical_one_hot=True,
            include_covariance_log10=True,
            include_delta_t=True,
        )
    else:
        raise ValueError(f"Unknown Phase 3B experiment ID: {exp_id}")
