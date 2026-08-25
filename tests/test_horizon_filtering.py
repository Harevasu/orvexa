"""Unit tests for horizon cutoff filtering (time_to_tca >= H)."""

import os
import sys
import unittest
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.event_builder import build_event_prefixes_from_raw


class TestHorizonFiltering(unittest.TestCase):
    """Test suite for strict horizon cutoff enforcement across warning horizons."""

    def setUp(self):
        """Build test event with CDMs across a span of days."""
        self.mock_events: Dict[str, List[Dict[str, str]]] = {
            "ev_multi": [
                {"event_id": "ev_multi", "time_to_tca": "6.2", "risk": "-25.0"},
                {"event_id": "ev_multi", "time_to_tca": "4.8", "risk": "-20.0"},
                {"event_id": "ev_multi", "time_to_tca": "3.1", "risk": "-18.0"},
                {"event_id": "ev_multi", "time_to_tca": "2.1", "risk": "-10.0"},
                {"event_id": "ev_multi", "time_to_tca": "1.0", "risk": "-5.0"},
                {"event_id": "ev_multi", "time_to_tca": "0.1", "risk": "-4.0"},
            ],
            "ev_short": [
                {"event_id": "ev_short", "time_to_tca": "2.4", "risk": "-30.0"},
                {"event_id": "ev_short", "time_to_tca": "0.5", "risk": "-28.0"},
            ],
        }

    def test_horizon_5_filtering(self):
        """Verify H=5 filtering retains only rows >= 5.0."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=5.0, feature_columns=["time_to_tca"])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.event_id, "ev_multi")
        self.assertEqual(rec.sequence_length, 1)
        self.assertEqual(float(rec.cdms[0]["time_to_tca"]), 6.2)
        # Final risk is -4.0 from row with time_to_tca 0.1
        self.assertEqual(rec.final_risk, -4.0)

    def test_horizon_3_filtering(self):
        """Verify H=3 filtering retains only rows >= 3.0."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=3.0, feature_columns=["time_to_tca"])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec.sequence_length, 3)
        tcas = [float(c["time_to_tca"]) for c in rec.cdms]
        self.assertEqual(tcas, [6.2, 4.8, 3.1])

    def test_horizon_2_filtering(self):
        """Verify H=2 filtering retains rows >= 2.0 for both events."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=2.0, feature_columns=["time_to_tca"])
        self.assertEqual(len(records), 2)
        rec_multi = next(r for r in records if r.event_id == "ev_multi")
        rec_short = next(r for r in records if r.event_id == "ev_short")
        self.assertEqual(rec_multi.sequence_length, 4)
        self.assertEqual(rec_short.sequence_length, 1)

    def test_horizon_7_filtering_produces_zero(self):
        """Verify H=7 filtering produces 0 eligible events when max(time_to_tca) < 7.0."""
        records, stats = build_event_prefixes_from_raw(self.mock_events, horizon_days=7.0, feature_columns=["time_to_tca"])
        self.assertEqual(len(records), 0)
        self.assertEqual(stats["eligible_events_count"], 0)


if __name__ == "__main__":
    unittest.main()
