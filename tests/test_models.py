"""Tests for model interfaces, fitting contracts, physics baselines, and risk prediction output format."""

import math
import os
import shutil
import tempfile
import unittest

from orvexa.models_linear import LinearRiskModel
from orvexa.models_physics import (
    PhysicsMaxRiskModel,
    compute_foster_2d_pc,
    compute_frisbee_max_pc,
    compute_mahalanobis_distance,
    project_to_encounter_bplane,
)
from orvexa.models_xgb import XGBoostRiskModel


class TestPhysicsModels(unittest.TestCase):
    """Test suite for physics baselines and collision probability calculations."""

    def test_physics_max_risk_model_predictions(self):
        records = [
            {"max_risk_estimate": "-18.5", "max_risk_scaling": "0.5"},
            {"max_risk_estimate": "-25.0"},
            {"max_risk_estimate": None},  # Missing
        ]
        model = PhysicsMaxRiskModel(scaling_col="max_risk_scaling")
        preds = model.predict_risk(records)
        self.assertEqual(len(preds), 3)
        self.assertAlmostEqual(preds[0], -18.0)
        self.assertAlmostEqual(preds[1], -25.0)
        self.assertAlmostEqual(preds[2], -30.0)

    def test_physics_max_risk_probability(self):
        records = [
            {"max_risk_estimate": "-4.0"},
            {"max_risk_estimate": "-10.0"},
        ]
        model = PhysicsMaxRiskModel()
        probs = model.predict_probability(records, threshold_log10=-5.0)
        self.assertEqual(len(probs), 2)
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[1], 0.0)

    def test_foster_2d_pc_properties(self):
        # Case 1: Miss distance zero (head-on collision geometry)
        miss_2d = [0.0, 0.0]
        cov_2d = [[1.0, 0.0], [0.0, 1.0]]
        hbr = 0.1
        pc_center = compute_foster_2d_pc(miss_2d, cov_2d, hard_body_radius=hbr)
        self.assertGreater(pc_center, 0.0)
        self.assertLessEqual(pc_center, 1.0)

        # Case 2: Very large miss distance (probability should drop toward zero)
        miss_far = [50.0, 50.0]
        pc_far = compute_foster_2d_pc(miss_far, cov_2d, hard_body_radius=hbr)
        self.assertLess(pc_far, 1e-15)

        # Case 3: Larger HBR increases collision probability
        pc_larger = compute_foster_2d_pc(miss_2d, cov_2d, hard_body_radius=0.2)
        self.assertGreater(pc_larger, pc_center)

    def test_frisbee_max_pc_bounds(self):
        # When miss distance <= HBR, MaxPc must be 1.0
        self.assertEqual(compute_frisbee_max_pc(0.005, hard_body_radius=0.01), 1.0)
        self.assertEqual(compute_frisbee_max_pc(0.01, hard_body_radius=0.01), 1.0)

        # When miss distance > HBR, MaxPc drops inversely with d^2
        max_pc_100m = compute_frisbee_max_pc(0.1, hard_body_radius=0.01)  # d = 100m
        max_pc_200m = compute_frisbee_max_pc(0.2, hard_body_radius=0.01)  # d = 200m
        self.assertGreater(max_pc_100m, max_pc_200m)
        self.assertAlmostEqual(max_pc_100m / max_pc_200m, 4.0, places=3)

    def test_bplane_projection_orthogonality(self):
        rel_pos = [1.0, 2.0, 3.0]
        rel_vel = [0.0, 10.0, 0.0]  # pure y-velocity
        cov_3x3 = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
        miss_2d, cov_2d = project_to_encounter_bplane(rel_pos, rel_vel, cov_3x3)
        self.assertEqual(len(miss_2d), 2)
        self.assertEqual(len(cov_2d), 2)
        # Covariance must remain symmetric positive-definite
        self.assertAlmostEqual(cov_2d[0][1], cov_2d[1][0])
        self.assertGreater(cov_2d[0][0], 0.0)
        self.assertGreater(cov_2d[1][1], 0.0)

    def test_mahalanobis_distance(self):
        rel_pos = [2.0, 0.0, 0.0]
        cov_3x3 = [
            [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        dm = compute_mahalanobis_distance(rel_pos, cov_3x3)
        # sqrt( (2/2)^2 ) = 1.0
        self.assertAlmostEqual(dm, 1.0, places=4)


class TestLinearAndTreeModels(unittest.TestCase):
    """Test suite for LinearRiskModel and XGBoostRiskModel fitting and determinism."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Synthetic linear relationship: y = -20.0 + 2.0*x1 - 3.0*x2
        self.X_train = [
            [1.0, 0.5],
            [2.0, 1.0],
            [0.0, 2.0],
            [-1.0, -0.5],
            [3.0, 0.0],
        ]
        self.y_train = [
            -20.0 + 2.0 * x[0] - 3.0 * x[1] for x in self.X_train
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_linear_risk_model_fit_and_predict(self):
        model = LinearRiskModel(alpha=0.01)
        model.fit(self.X_train, self.y_train)
        preds = model.predict_risk(self.X_train)
        self.assertEqual(len(preds), len(self.y_train))
        for p, y in zip(preds, self.y_train):
            self.assertAlmostEqual(p, y, places=1)

    def test_linear_risk_model_serialization(self):
        model = LinearRiskModel(alpha=0.5)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join(self.temp_dir, "linear_model.json")
        model.save(save_path)

        loaded = LinearRiskModel.load(save_path)
        self.assertTrue(loaded.is_fitted_)
        self.assertEqual(model.predict_risk(self.X_train), loaded.predict_risk(self.X_train))

    def test_xgboost_risk_model_fit_and_predict(self):
        model = XGBoostRiskModel(n_estimators=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict_risk(self.X_train)
        self.assertEqual(len(preds), len(self.y_train))

    def test_xgboost_risk_model_serialization(self):
        model = XGBoostRiskModel(n_estimators=5, random_state=42)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join(self.temp_dir, "xgb_model.json")
        model.save(save_path)

        loaded = XGBoostRiskModel.load(save_path)
        self.assertTrue(loaded.is_fitted_)
        self.assertEqual(loaded.n_estimators, 5)


if __name__ == "__main__":
    unittest.main()
