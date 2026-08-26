"""Tests for feature registry whitelisting, exclusions, and order persistence."""

import unittest

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
)
from orvexa.features_snapshot import extract_snapshot_features
from orvexa.features_temporal import CORE_TEMPORAL_NUMERIC_COLS, extract_temporal_summary_features


class TestFeatureRegistry(unittest.TestCase):
    """Test suite ensuring strict feature whitelist compliance and exclusion rules."""

    def test_whitelist_feature_counts_and_exclusions(self):
        self.assertEqual(len(DIRECT_FEATURE_COLUMNS), 34)
        for col in EXCLUDED_IDENTIFIERS:
            self.assertNotIn(col, DIRECT_FEATURE_COLUMNS, f"Identifier {col} must not be in direct features!")
        for col in EXCLUDED_TARGETS:
            self.assertNotIn(col, DIRECT_FEATURE_COLUMNS, f"Target {col} must not be in direct features!")
        for col in NEEDS_HUMAN_REVIEW_FEATURES:
            self.assertNotIn(col, DIRECT_FEATURE_COLUMNS, f"Unreviewed feature {col} must not be in direct features!")

    def test_snapshot_feature_extraction_respects_whitelist(self):
        mock_cdms = [
            {
                "event_id": "ev_01",
                "mission_id": "1",
                "risk": "-18.0",
                "time_to_tca": "3.5",
                "miss_distance": "1500.0",
                "relative_speed": "11000.0",
                "unknown_private_col": "should_be_ignored",
            }
        ]
        snap = extract_snapshot_features(
            mock_cdms,
            feature_cols=["miss_distance", "relative_speed"],
            horizon_cutoff=2.0,
        )
        self.assertIsNotNone(snap)
        self.assertIn("miss_distance", snap)
        self.assertIn("relative_speed", snap)
        self.assertNotIn("unknown_private_col", snap)

    def test_temporal_summary_feature_whitelist(self):
        mock_cdms = [
            {
                "event_id": "ev_01",
                "time_to_tca": "4.0",
                "miss_distance": "2000.0",
                "relative_speed": "10000.0",
                "t_sigma_r": "1.0",
                "c_sigma_r": "2.0",
            },
            {
                "event_id": "ev_01",
                "time_to_tca": "2.5",
                "miss_distance": "1500.0",
                "relative_speed": "10050.0",
                "t_sigma_r": "0.8",
                "c_sigma_r": "1.6",
            },
        ]
        temp_features = extract_temporal_summary_features(
            mock_cdms,
            numeric_cols=["miss_distance", "relative_speed"],
            horizon_cutoff=2.0,
        )
        self.assertIsNotNone(temp_features)
        self.assertEqual(temp_features["cdm_count"], 2.0)
        self.assertEqual(temp_features["miss_distance__latest"], 1500.0)
        self.assertEqual(temp_features["miss_distance__first"], 2000.0)
        self.assertEqual(temp_features["miss_distance__delta"], -500.0)
        self.assertAlmostEqual(temp_features["sequence_time_span_days"], 1.5)


if __name__ == "__main__":
    unittest.main()
