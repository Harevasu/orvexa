"""Tests for train-only preprocessing transformers and leakage prevention."""

import os
import shutil
import tempfile
import unittest

from orvexa.preprocessing import TrainFittedPreprocessor


class TestTrainFittedPreprocessor(unittest.TestCase):
    """Test suite for TrainFittedPreprocessor."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_records = [
            {"miss_distance": "100.0", "c_object_type": "DEBRIS", "t_obs_used": "10"},
            {"miss_distance": "200.0", "c_object_type": "PAYLOAD", "t_obs_used": "20"},
            {"miss_distance": "300.0", "c_object_type": "DEBRIS", "t_obs_used": "30"},
            {"miss_distance": None, "c_object_type": "UNKNOWN", "t_obs_used": "40"},  # Missing in train
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_fitting_statistics(self):
        prep = TrainFittedPreprocessor(
            numeric_features=["miss_distance", "t_obs_used"],
            categorical_features=["c_object_type"],
            add_missing_indicators=True,
            scale_numeric=False,
        )
        prep.fit(self.train_records)

        self.assertTrue(prep.is_fitted_)
        # Median of [100, 200, 300] is 200.0
        self.assertEqual(prep.stats_["miss_distance"].median, 200.0)
        self.assertEqual(prep.stats_["miss_distance"].missing_count, 1)

    def test_transform_with_missingness_indicators(self):
        prep = TrainFittedPreprocessor(
            numeric_features=["miss_distance"],
            categorical_features=["c_object_type"],
            add_missing_indicators=True,
            scale_numeric=False,
        )
        prep.fit(self.train_records)

        test_records = [
            {"miss_distance": "150.0", "c_object_type": "DEBRIS"},
            {"miss_distance": None, "c_object_type": "NEW_CAT"},  # Unseen category + missing val
        ]
        matrix = prep.transform(test_records)
        self.assertEqual(len(matrix), 2)

        # Row 1: miss_distance=150, missing_ind=0
        self.assertEqual(matrix[0][0], 150.0)
        self.assertEqual(matrix[0][1], 0.0)

        # Row 2: miss_distance=200 (imputed with train median), missing_ind=1
        self.assertEqual(matrix[1][0], 200.0)
        self.assertEqual(matrix[1][1], 1.0)

    def test_serialization_and_reload(self):
        prep = TrainFittedPreprocessor(
            numeric_features=["miss_distance"],
            categorical_features=["c_object_type"],
        )
        prep.fit(self.train_records)
        save_path = os.path.join(self.temp_dir, "preprocessor.json")
        prep.save(save_path)

        loaded = TrainFittedPreprocessor.load(save_path)
        self.assertTrue(loaded.is_fitted_)
        self.assertEqual(prep.stats_["miss_distance"].median, loaded.stats_["miss_distance"].median)

        # Transform output must be identical
        self.assertEqual(prep.transform(self.train_records), loaded.transform(self.train_records))


if __name__ == "__main__":
    unittest.main()
