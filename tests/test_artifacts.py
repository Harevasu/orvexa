"""Tests for artifact storage, paths, and persistence."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.artifact_store import ArtifactStore


class TestArtifactStore(unittest.TestCase):
    """Test suite for ArtifactStore path resolution and JSON serialization."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = ArtifactStore(root_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_path_generators_and_directory_creation(self):
        model_dir = self.store.get_model_dir("xgboost_risk", 2.0)
        self.assertTrue(model_dir.exists())
        self.assertTrue(str(model_dir).endswith("xgboost_risk_h2.0"))

        prep_path = self.store.get_preprocessor_path(3.0)
        self.assertTrue(prep_path.parent.exists())
        self.assertTrue(str(prep_path).endswith("preprocessor_h3.0.json"))

        metrics_path = self.store.get_metrics_path("tcn_causal", 5.0, split="val")
        self.assertTrue(metrics_path.parent.exists())
        self.assertTrue(str(metrics_path).endswith("tcn_causal_h5.0_val.json"))

    def test_save_and_load_json(self):
        data = {"metric": "recall_at_5pct", "value": 0.85, "horizon": 2.0}
        path = self.store.get_metrics_path("linear_ridge", 2.0)
        self.store.save_json(data, path)
        self.assertTrue(path.exists())

        loaded = self.store.load_json(path)
        self.assertEqual(loaded["metric"], "recall_at_5pct")
        self.assertEqual(loaded["value"], 0.85)


if __name__ == "__main__":
    unittest.main()
