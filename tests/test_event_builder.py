"""Unit tests for event builder logic and leakage prevention."""

import os
import sys
import unittest
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
    build_event_prefixes_from_raw,
)


class TestEventBuilder(unittest.TestCase):
    """Test suite for event grouping, final target extraction, and prefix construction."""

    def setUp(self):
        """Create synthetic test events with decreasing time_to_tca."""
        self.mock_events: Dict[str, List[Dict[str, str]]] = {
            "ev_1": [
                {
                    "event_id": "ev_1",
                    "time_to_tca": "5.5",
                    "mission_id": "1",
                    "risk": "-28.0",
                    "c_object_type": "DEBRIS",
                    "miss_distance": "12000.0",
                    "relative_speed": "10500.0",
                    "relative_position_r": "1.0",
                    "relative_position_t": "2.0",
                    "relative_position_n": "3.0",
                    "relative_velocity_r": "4.0",
                    "relative_velocity_t": "5.0",
                    "relative_velocity_n": "6.0",
                    "mahalanobis_distance": "50.0",
                    "geocentric_latitude": "10.0",
                    "azimuth": "120.0",
                    "elevation": "-15.0",
                    "t_sigma_r": "1.1",
                    "t_sigma_t": "2.2",
                    "t_sigma_n": "3.3",
                    "c_sigma_r": "4.4",
                    "c_sigma_t": "5.5",
                    "c_sigma_n": "6.6",
                    "t_obs_available": "100",
                    "t_obs_used": "95",
                    "c_obs_available": "20",
                    "c_obs_used": "18",
                    "t_residuals_accepted": "98.0",
                    "c_residuals_accepted": "97.0",
                    "t_weighted_rms": "0.8",
                    "c_weighted_rms": "1.1",
                    "max_risk_estimate": "-25.0",
                    "max_risk_scaling": "0.0",
                    "F10": "90.0",
                    "F3M": "92.0",
                    "AP": "10.0",
                    "SSN": "40.0",
                },
                {
                    "event_id": "ev_1",
                    "time_to_tca": "2.5",
                    "mission_id": "1",
                    "risk": "-20.0",
                    "c_object_type": "DEBRIS",
                    "miss_distance": "8000.0",
                    "relative_speed": "10500.0",
                    "relative_position_r": "1.0",
                    "relative_position_t": "2.0",
                    "relative_position_n": "3.0",
                    "relative_velocity_r": "4.0",
                    "relative_velocity_t": "5.0",
                    "relative_velocity_n": "6.0",
                    "mahalanobis_distance": "30.0",
                    "geocentric_latitude": "10.0",
                    "azimuth": "120.0",
                    "elevation": "-15.0",
                    "t_sigma_r": "0.9",
                    "t_sigma_t": "1.8",
                    "t_sigma_n": "2.5",
                    "c_sigma_r": "3.0",
                    "c_sigma_t": "4.0",
                    "c_sigma_n": "5.0",
                    "t_obs_available": "150",
                    "t_obs_used": "145",
                    "c_obs_available": "30",
                    "c_obs_used": "28",
                    "t_residuals_accepted": "99.0",
                    "c_residuals_accepted": "98.0",
                    "t_weighted_rms": "0.7",
                    "c_weighted_rms": "0.9",
                    "max_risk_estimate": "-18.0",
                    "max_risk_scaling": "0.0",
                    "F10": "90.0",
                    "F3M": "92.0",
                    "AP": "10.0",
                    "SSN": "40.0",
                },
                {
                    "event_id": "ev_1",
                    "time_to_tca": "0.5",
                    "mission_id": "1",
                    "risk": "-5.2",  # Final true target
                    "c_object_type": "DEBRIS",
                    "miss_distance": "500.0",
                    "relative_speed": "10500.0",
                    "relative_position_r": "1.0",
                    "relative_position_t": "2.0",
                    "relative_position_n": "3.0",
                    "relative_velocity_r": "4.0",
                    "relative_velocity_t": "5.0",
                    "relative_velocity_n": "6.0",
                    "mahalanobis_distance": "5.0",
                    "geocentric_latitude": "10.0",
                    "azimuth": "120.0",
                    "elevation": "-15.0",
                    "t_sigma_r": "0.5",
                    "t_sigma_t": "1.0",
                    "t_sigma_n": "1.2",
                    "c_sigma_r": "1.5",
                    "c_sigma_t": "2.0",
                    "c_sigma_n": "2.5",
                    "t_obs_available": "200",
                    "t_obs_used": "198",
                    "c_obs_available": "45",
                    "c_obs_used": "44",
                    "t_residuals_accepted": "99.5",
                    "c_residuals_accepted": "99.0",
                    "t_weighted_rms": "0.6",
                    "c_weighted_rms": "0.8",
                    "max_risk_estimate": "-5.0",
                    "max_risk_scaling": "0.0",
                    "F10": "90.0",
                    "F3M": "92.0",
                    "AP": "10.0",
                    "SSN": "40.0",
                },
            ],
            "ev_2_late_only": [
                {
                    "event_id": "ev_2_late_only",
                    "time_to_tca": "1.2",
                    "mission_id": "2",
                    "risk": "-15.0",
                    "c_object_type": "PAYLOAD",
                    "miss_distance": "25000.0",
                    "relative_speed": "8000.0",
                    "relative_position_r": "0.0",
                    "relative_position_t": "0.0",
                    "relative_position_n": "0.0",
                    "relative_velocity_r": "0.0",
                    "relative_velocity_t": "0.0",
                    "relative_velocity_n": "0.0",
                    "mahalanobis_distance": "100.0",
                    "geocentric_latitude": "0.0",
                    "azimuth": "0.0",
                    "elevation": "0.0",
                    "t_sigma_r": "1.0",
                    "t_sigma_t": "1.0",
                    "t_sigma_n": "1.0",
                    "c_sigma_r": "1.0",
                    "c_sigma_t": "1.0",
                    "c_sigma_n": "1.0",
                    "t_obs_available": "50",
                    "t_obs_used": "50",
                    "c_obs_available": "10",
                    "c_obs_used": "10",
                    "t_residuals_accepted": "95.0",
                    "c_residuals_accepted": "95.0",
                    "t_weighted_rms": "1.0",
                    "c_weighted_rms": "1.0",
                    "max_risk_estimate": "-15.0",
                    "max_risk_scaling": "0.0",
                    "F10": "80.0",
                    "F3M": "80.0",
                    "AP": "5.0",
                    "SSN": "20.0",
                }
            ],
        }

    def test_final_target_extraction(self):
        """Verify that final_risk is taken from the minimum time_to_tca CDM."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=2.0)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.event_id, "ev_1")
        # Final risk is -5.2 from row with time_to_tca 0.5
        self.assertEqual(rec.final_risk, -5.2)

    def test_horizon_filtering_excludes_post_cutoff_cdms(self):
        """Verify that CDMs with time_to_tca < H are strictly excluded."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=2.0)
        rec = records[0]
        # Only CDMs with time_to_tca >= 2.0 (5.5 and 2.5) should be retained
        self.assertEqual(rec.sequence_length, 2)
        retained_tcas = [float(c["time_to_tca"]) for c in rec.cdms]
        self.assertEqual(retained_tcas, [5.5, 2.5])
        self.assertTrue(all(t >= 2.0 for t in retained_tcas))

    def test_ineligible_event_exclusion(self):
        """Verify that events with no CDM >= H are excluded."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=2.0)
        event_ids = [r.event_id for r in records]
        self.assertNotIn("ev_2_late_only", event_ids)
        self.assertEqual(stats["skipped_zero_cdms_count"], 1)

    def test_feature_whitelist_enforcement(self):
        """Verify that input feature dictionaries only contain DIRECT features."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=2.0)
        rec = records[0]
        for cdm in rec.cdms:
            # Check presence of direct features
            self.assertEqual(set(cdm.keys()), set(DIRECT_FEATURE_COLUMNS))
            # Check absence of identifiers and target
            for excl in EXCLUDED_IDENTIFIERS + EXCLUDED_TARGETS:
                self.assertNotIn(excl, cdm)
            # Check absence of unapproved features
            for rev in NEEDS_HUMAN_REVIEW_FEATURES:
                self.assertNotIn(rev, cdm)
            for na in NOT_AVAILABLE_FEATURES:
                self.assertNotIn(na, cdm)


if __name__ == "__main__":
    unittest.main()
