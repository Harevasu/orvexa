"""Schema registry, column verification, and feature alias definitions."""

from typing import Any, Dict, List, Optional, Set, Tuple

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
)

MANDATORY_RAW_COLUMNS = ["event_id", "time_to_tca", "risk"]


class SchemaRegistry:
    """Registry for validating column schemas, mandatory fields, and whitelisted features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.mandatory_columns: List[str] = list(
            self.config.get("mandatory_columns", MANDATORY_RAW_COLUMNS)
        )
        self.whitelisted_features: List[str] = list(
            self.config.get("whitelisted_features", DIRECT_FEATURE_COLUMNS)
        )
        self.excluded_identifiers: List[str] = list(
            self.config.get("excluded_identifiers", EXCLUDED_IDENTIFIERS)
        )
        self.excluded_targets: List[str] = list(
            self.config.get("excluded_targets", EXCLUDED_TARGETS)
        )

    def validate_columns(self, observed_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate presence of mandatory columns in observed dataset.
        
        Args:
            observed_columns: List of observed column names in raw dataset.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        obs_set = set(observed_columns)
        errors = []
        for col in self.mandatory_columns:
            if col not in obs_set:
                errors.append(f"Missing mandatory column: '{col}'")

        return len(errors) == 0, errors

    def get_permissible_feature_columns(self, observed_columns: List[str]) -> List[str]:
        """Filter observed columns to return only valid whitelisted features."""
        obs_set = set(observed_columns)
        return [c for c in self.whitelisted_features if c in obs_set]
