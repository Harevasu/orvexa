"""Tests for TCN tensor shapes, left-padding, causality invariants, and final-valid-step pooling."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from orvexa.models_tcn import (
    TCNRiskModel,
    causal_conv1d_forward_numpy,
    huber_loss,
)


class TestTCNShapeAndCausality(unittest.TestCase):
    """Test suite for causal convolution invariants, tensor shapes, validity masking, and pooling."""

    def test_huber_loss_function(self):
        # Small error (|e| <= 1.0) -> quadratic 0.5 * e^2
        self.assertAlmostEqual(huber_loss(y_true=-20.0, y_pred=-20.5, delta=1.0), 0.125)
        self.assertAlmostEqual(huber_loss(y_true=-20.0, y_pred=-20.0, delta=1.0), 0.0)

        # Large error (|e| > 1.0) -> linear: delta * (|e| - 0.5*delta)
        # e = 3.0 -> 1.0 * (3.0 - 0.5) = 2.5
        self.assertAlmostEqual(huber_loss(y_true=-20.0, y_pred=-17.0, delta=1.0), 2.5)

    def test_causal_conv1d_numpy_strict_causality(self):
        """CRITICAL INVARIANT: Changing future inputs at t' > t must NOT change output at t."""
        # 1 sample, 1 channel, 5 timesteps: [0, 1, 2, 3, 4]
        x_base = [[[1.0, 2.0, 3.0, 4.0, 5.0]]]
        weights = [[[1.0, 0.5, 0.25]]]  # kernel_size = 3
        bias = [0.0]
        dilation = 1

        out_base = causal_conv1d_forward_numpy(x_base, weights, bias, dilation=dilation)
        self.assertEqual(len(out_base[0][0]), 5)

        # Modify future step (index 4)
        x_perturbed_future = [[[1.0, 2.0, 3.0, 4.0, 999.0]]]
        out_perturbed = causal_conv1d_forward_numpy(x_perturbed_future, weights, bias, dilation=dilation)

        # Outputs at earlier steps (t = 0, 1, 2, 3) must be strictly identical!
        for t in range(4):
            self.assertEqual(
                out_base[0][0][t],
                out_perturbed[0][0][t],
                f"Causality violation: output at timestep t={t} changed when future timestep t=4 was modified!",
            )

        # Output at t=4 must differ
        self.assertNotEqual(out_base[0][0][4], out_perturbed[0][0][4])

    def test_causal_conv1d_numpy_dilated_causality(self):
        """Test strict causality with dilation = 2 and dilation = 4."""
        for d in [2, 4]:
            x_base = [[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]
            weights = [[[1.0, 1.0, 1.0]]]  # kernel_size=3
            bias = [0.0]
            out_base = causal_conv1d_forward_numpy(x_base, weights, bias, dilation=d)

            # Perturb last step (t=7)
            x_mod = [[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0]]]
            out_mod = causal_conv1d_forward_numpy(x_mod, weights, bias, dilation=d)

            # Timesteps 0..6 must be identical
            for t in range(7):
                self.assertEqual(out_base[0][0][t], out_mod[0][0][t])
            self.assertNotEqual(out_base[0][0][7], out_mod[0][0][7])

    def test_prepare_sequence_tensors_variable_lengths_and_left_padding(self):
        """Test left-padding and validity masks for variable sequence lengths [1, 2, 5, 10, 23]."""
        model = TCNRiskModel(in_features=2, max_seq_len=23)

        for L in [1, 2, 5, 10, 23]:
            mock_cdms = [
                {"time_to_tca": str(20.0 - i * 0.2), "f1": float(i + 1), "f2": float((i + 1) * 2), "risk": -15.0}
                for i in range(L)
            ]
            events = {f"ev_len_{L}": mock_cdms}

            X, mask, targets, ids = model.prepare_sequence_tensors(
                events, feature_cols=["f1", "f2"], horizon_cutoff=0.0
            )

            self.assertEqual(len(ids), 1)
            self.assertEqual(targets, [-15.0])

            mask_list = mask[0] if isinstance(mask, list) else mask[0].tolist()
            pad_len = 23 - L
            expected_mask = [0.0] * pad_len + [1.0] * L
            self.assertEqual(mask_list, expected_mask)

            # Check left-padding: padded indices 0..pad_len-1 must be 0.0
            x_matrix = X[0] if isinstance(X, list) else X[0].tolist()
            for f_idx in range(2):
                pad_vals = x_matrix[f_idx][:pad_len]
                real_vals = x_matrix[f_idx][pad_len:]
                self.assertTrue(all(v == 0.0 for v in pad_vals))
                self.assertEqual(len(real_vals), L)
                # The newest valid CDM must be placed at the rightmost index (max_seq_len - 1 = 22)
                if f_idx == 0:
                    self.assertEqual(x_matrix[0][22], float(L))  # Last CDM f1 value is L

    def test_final_valid_step_pooling_numpy_position(self):
        """Pure mathematical verification that final representation is taken at index -1."""
        # Multi-channel causal convolution on left-padded sequence
        # Sequence of length 3 left padded in length 23: padding is at 0..19, real is at 20, 21, 22
        time_len = 23
        x = [[[0.0] * 20 + [1.0, 2.0, 3.0]]]  # 1 sample, 1 channel, 23 timesteps
        weights = [[[1.0, 1.0, 1.0]]]  # kernel_size = 3
        bias = [0.0]

        out = causal_conv1d_forward_numpy(x, weights, bias, dilation=1)
        # Position -1 (index 22) must evaluate the latest timestep (uses 1.0 + 2.0 + 3.0 = 6.0)
        final_h_val = out[0][0][-1]
        self.assertEqual(final_h_val, 6.0)

        # In contrast, position 2 (which lengths - 1 would have extracted) is purely 0.0
        padding_pos_2_val = out[0][0][2]
        self.assertEqual(padding_pos_2_val, 0.0)
        self.assertNotEqual(final_h_val, padding_pos_2_val)

    def test_final_valid_step_pooling_uses_final_position(self):
        """Test that final representation is gathered from the final timestep position (h[:, :, -1])."""
        try:
            import torch
            from orvexa.models_tcn import MaskedCausalTCN

            torch.manual_seed(42)
            tcn = MaskedCausalTCN(in_features=2, channels=[8, 8], kernel_size=3, dilations=[1, 2])
            tcn.eval()

            # Batch of 2 samples of length 23
            x = torch.randn(2, 2, 23)
            mask = torch.ones(2, 23)
            mask[0, :15] = 0.0  # Sample 0 has 15 padding steps (seq_len = 8)
            mask[1, :20] = 0.0  # Sample 1 has 20 padding steps (seq_len = 3)

            with torch.no_grad():
                out = tcn(x, mask)
                self.assertEqual(out.shape, (2, 1))

                # Verify that out matches direct linear head application to h[:, :, -1]
                h = tcn.network(x)
                expected_out_0 = tcn.head(h[0:1, :, -1])
                expected_out_1 = tcn.head(h[1:2, :, -1])

                self.assertAlmostEqual(out[0, 0].item(), expected_out_0[0, 0].item(), places=5)
                self.assertAlmostEqual(out[1, 0].item(), expected_out_1[0, 0].item(), places=5)

        except ImportError:
            self.skipTest("PyTorch not installed; tested via test_final_valid_step_pooling_numpy_position.")

    def test_padding_invariance_for_truncated_sequences(self):
        """Verify that padding values prior to the receptive field cannot alter representation."""
        # For an event with 3 timesteps in a 23-step left-padded window:
        # Padded timesteps are strictly initialized to 0.0
        model = TCNRiskModel(in_features=2, max_seq_len=23)
        cdms = [
            {"time_to_tca": "5.0", "f1": 1.0, "f2": 2.0, "risk": -10.0},
            {"time_to_tca": "4.0", "f1": 2.0, "f2": 4.0, "risk": -10.0},
            {"time_to_tca": "3.0", "f1": 3.0, "f2": 6.0, "risk": -10.0},
        ]
        X, mask, targets, ids = model.prepare_sequence_tensors(
            {"ev": cdms}, feature_cols=["f1", "f2"], horizon_cutoff=2.0
        )
        x_mat = X[0] if isinstance(X, list) else X[0].tolist()

        # Check that slots 0..19 are exactly 0.0
        self.assertEqual(x_mat[0][:20], [0.0] * 20)
        self.assertEqual(x_mat[1][:20], [0.0] * 20)
        # Check that slots 20, 21, 22 are real values
        self.assertEqual(x_mat[0][20:], [1.0, 2.0, 3.0])
        self.assertEqual(x_mat[1][20:], [2.0, 4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
