"""Tests for data leakage prevention and partition disjointness."""

import os
import shutil
import tempfile
import unittest

from orvexa.splitting import (
    SplitManifest,
    assert_split_disjointness,
    make_chronological_splits,
)


class TestSplitLeakage(unittest.TestCase):
    """Test suite ensuring strict zero-leakage event partitioning."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.event_ids = [f"event_{i:04d}" for i in range(100)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chronological_split_ratios_and_disjointness(self):
        manifest = make_chronological_splits(self.event_ids, train_ratio=0.70, val_ratio=0.15)
        self.assertEqual(len(manifest.train_event_ids), 70)
        self.assertEqual(len(manifest.val_event_ids), 15)
        self.assertEqual(len(manifest.test_event_ids), 15)

        train_s = set(manifest.train_event_ids)
        val_s = set(manifest.val_event_ids)
        test_s = set(manifest.test_event_ids)

        self.assertTrue(assert_split_disjointness(train_s, val_s, test_s))
        self.assertEqual(len(train_s) + len(val_s) + len(test_s), 100)

    def test_chronological_ordering_preserved(self):
        manifest = make_chronological_splits(self.event_ids, train_ratio=0.70, val_ratio=0.15)
        # Train events are earliest
        self.assertEqual(manifest.train_event_ids[0], "event_0000")
        self.assertEqual(manifest.train_event_ids[-1], "event_0069")
        # Val events follow train
        self.assertEqual(manifest.val_event_ids[0], "event_0070")
        self.assertEqual(manifest.val_event_ids[-1], "event_0084")
        # Test events are latest
        self.assertEqual(manifest.test_event_ids[0], "event_0085")
        self.assertEqual(manifest.test_event_ids[-1], "event_0099")

    def test_manifest_serialization_and_overlap_rejection(self):
        manifest = make_chronological_splits(self.event_ids)
        path = os.path.join(self.temp_dir, "splits.json")
        manifest.save(path)

        loaded = SplitManifest.load(path)
        self.assertEqual(manifest.train_event_ids, loaded.train_event_ids)
        self.assertEqual(manifest.val_event_ids, loaded.val_event_ids)
        self.assertEqual(manifest.test_event_ids, loaded.test_event_ids)

        # Overlap must raise ValueError
        with self.assertRaises(ValueError):
            SplitManifest(
                train_event_ids=["ev1", "ev2"],
                val_event_ids=["ev2", "ev3"],  # ev2 overlaps
                test_event_ids=["ev4"],
            )


if __name__ == "__main__":
    unittest.main()
