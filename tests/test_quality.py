"""Tests for data quality, missingness auditing, and duplicate checks."""

import unittest

from orvexa.quality import run_data_quality_audit


class TestQuality(unittest.TestCase):
    """Test suite for data quality auditing routines."""

    def test_run_data_quality_audit(self):
        records = [
            {"event_id": "ev1", "risk": "-4.5", "miss_distance": "100.0"},
            {"event_id": "ev1", "risk": "-4.5", "miss_distance": "90.0"},
            {"event_id": "ev2", "risk": "-20.0", "miss_distance": None},
        ]
        audit = run_data_quality_audit(records)
        self.assertEqual(audit["total_rows"], 3)
        self.assertEqual(audit["total_unique_events"], 2)
        self.assertEqual(audit["sequence_lengths"]["max"], 2)
        self.assertEqual(audit["sequence_lengths"]["min"], 1)
        self.assertEqual(audit["target_risk_summary"]["high_risk_count_ge_minus_5"], 2)
        self.assertEqual(audit["missingness_rates"]["miss_distance"], round(1 / 3, 5))


if __name__ == "__main__":
    unittest.main()
