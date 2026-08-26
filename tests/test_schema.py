"""Tests for schema validation and column requirements."""

import unittest

from orvexa.schema import SchemaRegistry


class TestSchema(unittest.TestCase):
    """Test suite for SchemaRegistry column verification."""

    def test_mandatory_columns_validation(self):
        reg = SchemaRegistry()
        valid_cols = ["event_id", "time_to_tca", "risk", "miss_distance"]
        is_valid, errors = reg.validate_columns(valid_cols)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        missing_cols = ["event_id", "miss_distance"]  # missing time_to_tca and risk
        is_valid_bad, errors_bad = reg.validate_columns(missing_cols)
        self.assertFalse(is_valid_bad)
        self.assertEqual(len(errors_bad), 2)

    def test_permissible_feature_columns_filtering(self):
        reg = SchemaRegistry()
        observed = ["event_id", "risk", "miss_distance", "relative_speed", "random_unregistered_col"]
        permissible = reg.get_permissible_feature_columns(observed)
        self.assertIn("miss_distance", permissible)
        self.assertIn("relative_speed", permissible)
        self.assertNotIn("event_id", permissible)
        self.assertNotIn("risk", permissible)
        self.assertNotIn("random_unregistered_col", permissible)


if __name__ == "__main__":
    unittest.main()
