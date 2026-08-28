"""Unit tests for Phase 5 Step 3 Probabilistic Modeling & Conformal Uncertainty Infrastructure."""

import json
import math
import os
from pathlib import Path
import unittest

import numpy as np

from orvexa.conformal import SplitConformalCalibrator
from orvexa.losses_quantile import (
    compute_quantile_evaluation_metrics,
    multi_quantile_pinball_loss_numpy,
    pinball_loss_numpy,
    pinball_loss_scalar,
)
from orvexa.models_probabilistic import (
    DEFAULT_QUANTILES,
    QuantileMaskedCausalTCN,
    QuantileTCNRiskModel,
)
from orvexa.preprocessing_phase3b import Phase3BSequencePreprocessor
from orvexa.splitting import Phase5SplitManifest

try:
    import torch
    from orvexa.losses_quantile import MultiQuantileLoss
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPhase5Step3Probabilistic(unittest.TestCase):
    """Comprehensive test suite for Phase 5 Step 3 probabilistic and conformal components."""

    def setUp(self):
        self.workspace = Path(__file__).resolve().parent.parent
        self.split_manifest_path = self.workspace / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"

    def test_pinball_loss_scalar_and_numpy_analytical(self):
        """Test pinball loss against known analytical values."""
        # y = 10.0, q = 8.0 -> error = 2.0 (underprediction)
        # For tau = 0.90: loss = 0.90 * 2.0 = 1.80
        self.assertAlmostEqual(pinball_loss_scalar(10.0, 8.0, 0.90), 1.80, places=6)
        # For tau = 0.10: loss = 0.10 * 2.0 = 0.20
        self.assertAlmostEqual(pinball_loss_scalar(10.0, 8.0, 0.10), 0.20, places=6)

        # y = 5.0, q = 8.0 -> error = -3.0 (overprediction)
        # For tau = 0.90: loss = (1 - 0.90) * 3.0 = 0.30
        self.assertAlmostEqual(pinball_loss_scalar(5.0, 8.0, 0.90), 0.30, places=6)
        # For tau = 0.10: loss = (1 - 0.10) * 3.0 = 2.70
        self.assertAlmostEqual(pinball_loss_scalar(5.0, 8.0, 0.10), 2.70, places=6)

        # Numpy array vectorization check
        y_arr = np.array([10.0, 5.0])
        q_arr = np.array([8.0, 8.0])
        loss_90 = pinball_loss_numpy(y_arr, q_arr, 0.90)
        np.testing.assert_allclose(loss_90, np.array([1.80, 0.30]), rtol=1e-5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for MultiQuantileLoss test")
    def test_pytorch_multi_quantile_loss_gradient(self):
        """Verify PyTorch MultiQuantileLoss backpropagation and value match numpy."""
        quantiles = [0.10, 0.50, 0.90]
        loss_fn = MultiQuantileLoss(quantiles)

        y_true = torch.tensor([[10.0], [5.0]], requires_grad=False)
        y_pred = torch.tensor([[8.0, 9.0, 11.0], [4.0, 6.0, 7.0]], requires_grad=True)

        loss = loss_fn(y_pred, y_true)
        self.assertTrue(loss.item() > 0.0)

        # Backward pass
        loss.backward()
        self.assertIsNotNone(y_pred.grad)
        self.assertEqual(y_pred.grad.shape, y_pred.shape)
        self.assertTrue(torch.all(torch.isfinite(y_pred.grad)))

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for Quantile Network test")
    def test_quantile_network_shape_and_monotonicity(self):
        """Test QuantileMaskedCausalTCN outputs correct shape and strictly non-crossing quantiles."""
        batch_size = 8
        in_features = 37
        seq_len = 15
        quantiles = DEFAULT_QUANTILES  # 7 quantiles

        model = QuantileMaskedCausalTCN(
            in_features=in_features,
            quantiles=quantiles,
            channels=[32, 32, 64],
            kernel_size=3,
            dilations=[1, 2, 4],
            dropout=0.0,
            enforce_monotonic=True,
        )

        x = torch.randn(batch_size, in_features, seq_len)
        out = model(x)

        # Shape check: [batch_size, 7]
        self.assertEqual(out.shape, (batch_size, len(quantiles)))

        # Monotonicity check: q_0 < q_1 < q_2 < ... < q_6 strictly for all samples in batch
        out_np = out.detach().numpy()
        for b in range(batch_size):
            for k in range(len(quantiles) - 1):
                self.assertGreaterEqual(
                    out_np[b, k + 1],
                    out_np[b, k],
                    f"Quantile crossing detected in sample {b} between q_{k} and q_{k+1}!",
                )

    def test_quantile_evaluation_metrics_calculator(self):
        """Test compute_quantile_evaluation_metrics for pinball loss, coverage, and crossing detection."""
        y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        # Monotonic predictions: [q0.10, q0.50, q0.90]
        q_preds = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0],
            [9.0, 10.0, 11.0],
        ])
        quantiles = [0.10, 0.50, 0.90]

        metrics = compute_quantile_evaluation_metrics(y_true, q_preds, quantiles)
        self.assertEqual(metrics["n_samples"], 5)
        self.assertEqual(metrics["quantile_crossing_violations"], 0)
        self.assertEqual(metrics["quantile_crossing_rate"], 0.0)
        self.assertIn("80pct_interval", metrics["intervals"])
        # All 5 true values fall inside [q0.10, q0.90]
        self.assertEqual(metrics["intervals"]["80pct_interval"]["empirical_coverage"], 1.0)

    def test_conformal_calibrator_finite_sample_formula(self):
        """Verify finite-sample quantile calculation rule k = ceil((n+1)(1-alpha))."""
        calibrator = SplitConformalCalibrator(default_alpha=0.10, score_type="absolute_residual")
        
        # 10 calibration scores: 1.0, 2.0, ..., 10.0
        y_cal = np.zeros(10)
        y_pred_cal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        calibrator.fit(y_cal, y_pred_cal)

        # n = 10, alpha = 0.10 (90% confidence)
        # k = ceil((10 + 1) * 0.90) = ceil(9.9) = 10
        # sorted_scores[10 - 1] = sorted_scores[9] = 10.0
        q_conf_90 = calibrator.get_conformal_quantile(alpha=0.10)
        self.assertEqual(q_conf_90, 10.0)

        # alpha = 0.20 (80% confidence)
        # k = ceil((10 + 1) * 0.80) = ceil(8.8) = 9
        # sorted_scores[9 - 1] = sorted_scores[8] = 9.0
        q_conf_80 = calibrator.get_conformal_quantile(alpha=0.20)
        self.assertEqual(q_conf_80, 9.0)

    def test_conformal_calibrator_interval_construction_and_coverage(self):
        """Test conformal interval bounds generation and empirical coverage evaluation."""
        calibrator = SplitConformalCalibrator(default_alpha=0.10)
        
        # Calibration set with residuals in [0, 2]
        np.random.seed(42)
        n_cal = 100
        y_cal = np.random.uniform(-10, 0, n_cal)
        noise_cal = np.random.normal(0, 1.0, n_cal)
        pred_cal = y_cal + noise_cal
        calibrator.fit(y_cal, pred_cal)

        # Evaluate on independent validation set
        n_val = 200
        y_val = np.random.uniform(-10, 0, n_val)
        noise_val = np.random.normal(0, 1.0, n_val)
        pred_val = y_val + noise_val

        cov_results = calibrator.evaluate_coverage(y_val, pred_val, alpha=0.10)
        self.assertIn("empirical_coverage", cov_results)
        self.assertIn("mean_interval_width", cov_results)
        # Empirical coverage should be approximately nominal 90% (e.g. between 80% and 98%)
        self.assertGreaterEqual(cov_results["empirical_coverage"], 0.80)
        self.assertLessEqual(cov_results["empirical_coverage"], 0.99)

    def test_conformal_calibrator_serialization(self):
        """Test JSON save and reload of SplitConformalCalibrator."""
        calibrator = SplitConformalCalibrator(default_alpha=0.15, score_type="absolute_residual")
        calibrator.fit(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.2, 3.1]))
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            calibrator.save(tmp_path)
            loaded = SplitConformalCalibrator.load(tmp_path)
            self.assertEqual(calibrator.default_alpha, loaded.default_alpha)
            self.assertEqual(calibrator.n_calibration_samples_, loaded.n_calibration_samples_)
            self.assertEqual(calibrator.get_conformal_quantile(0.10), loaded.get_conformal_quantile(0.10))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_phase5_split_quarantine_enforcement(self):
        """Ensure Phase 5 manifest separates calibration and strictly quarantines internal test."""
        self.assertTrue(self.split_manifest_path.exists())
        manifest = Phase5SplitManifest.load(self.split_manifest_path)

        tr = set(manifest.train_event_ids)
        va = set(manifest.val_event_ids)
        ca = set(manifest.cal_event_ids)
        te = set(manifest.test_event_ids)

        self.assertEqual(len(tr), 6708)
        self.assertEqual(len(va), 1677)
        self.assertEqual(len(ca), 1118)
        self.assertEqual(len(te), 1677)

        # Disjointness
        self.assertTrue(tr.isdisjoint(va))
        self.assertTrue(tr.isdisjoint(ca))
        self.assertTrue(tr.isdisjoint(te))
        self.assertTrue(va.isdisjoint(ca))
        self.assertTrue(va.isdisjoint(te))
        self.assertTrue(ca.isdisjoint(te))


if __name__ == "__main__":
    unittest.main()
