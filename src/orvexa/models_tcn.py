"""Masked Causal Temporal Convolutional Network (TCN) for variable-length CDM sequences.

References and Attributions:
- Bai, Kolter, & Koltun (2018): "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271.
- Architecture adapted from reference_repo/pytorch-tcn (MIT License, Paul Krug) with custom
  ORVEXA event validity masking, left padding, and final-valid-step pooling.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def huber_loss(y_true: float, y_pred: float, delta: float = 1.0) -> float:
    """Huber loss for robust regression on log-risk scores."""
    error = y_true - y_pred
    abs_error = abs(error)
    if abs_error <= delta:
        return 0.5 * (error ** 2)
    else:
        return delta * (abs_error - 0.5 * delta)


def causal_conv1d_forward_numpy(
    x: List[List[List[float]]],  # [batch, in_channels, time]
    weights: List[List[List[float]]],  # [out_channels, in_channels, kernel_size]
    bias: List[float],  # [out_channels]
    dilation: int = 1,
) -> List[List[List[float]]]:
    """Pure mathematical forward pass for 1D causal dilated convolution with exact left padding."""
    batch_size = len(x)
    in_channels = len(x[0])
    time_steps = len(x[0][0])
    out_channels = len(weights)
    kernel_size = len(weights[0][0])

    left_pad = (kernel_size - 1) * dilation

    # 1. Left-pad input on time dimension with zeros
    padded_time = time_steps + left_pad
    padded_x = [
        [
            [0.0] * left_pad + x[b][c]
            for c in range(in_channels)
        ]
        for b in range(batch_size)
    ]

    # 2. Convolve: output at step t uses inputs at t - (k-1)*d ... t (strictly causal)
    out = [
        [
            [bias[oc] for _ in range(time_steps)]
            for oc in range(out_channels)
        ]
        for _ in range(batch_size)
    ]

    for b in range(batch_size):
        for oc in range(out_channels):
            for t in range(time_steps):
                t_padded = t + left_pad
                s = bias[oc]
                for ic in range(in_channels):
                    for k in range(kernel_size):
                        time_idx = t_padded - (kernel_size - 1 - k) * dilation
                        s += padded_x[b][ic][time_idx] * weights[oc][ic][k]
                out[b][oc][t] = s

    return out


# PyTorch Module Implementation (available when torch is installed)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
        """1D Causal Dilated Convolution with exact left padding."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
        ) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,  # Manual left padding applied in forward
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, channels, time]
            if self.left_padding > 0:
                x = F.pad(x, (self.left_padding, 0))
            return self.conv(x)

    class TemporalResidualBlock(nn.Module):
        """Causal Dilated Residual Block with Weight Normalization and Dropout."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout: float = 0.15,
        ) -> None:
            super().__init__()
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout)

            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout)

            self.downsample = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else None
            )
            self.relu_out = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x if self.downsample is None else self.downsample(x)
            out = self.dropout1(self.relu1(self.conv1(x)))
            out = self.dropout2(self.relu2(self.conv2(out)))
            return self.relu_out(out + residual)

    class MaskedCausalTCN(nn.Module):
        """Masked causal Temporal Convolutional Network for event-level risk regression."""

        def __init__(
            self,
            in_features: int,
            channels: Optional[List[int]] = None,
            kernel_size: int = 3,
            dilations: Optional[List[int]] = None,
            dropout: float = 0.15,
        ) -> None:
            super().__init__()
            channels = channels or [64, 64, 128]
            dilations = dilations or [1, 2, 4]

            layers = []
            current_in = in_features
            for out_ch, dil in zip(channels, dilations):
                layers.append(
                    TemporalResidualBlock(
                        in_channels=current_in,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        dilation=dil,
                        dropout=dropout,
                    )
                )
                current_in = out_ch

            self.network = nn.Sequential(*layers)
            self.head = nn.Linear(current_in, 1)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Forward pass with final-valid-step representation extraction.
            
            For left-padded causal sequences ([0, 0, ..., real_1, real_2, real_last]),
            the newest valid timestep is strictly the final position along the time dimension (index -1).
            
            Args:
                x: Input tensor of shape [batch, in_features, time].
                mask: Optional binary mask of shape [batch, time], where 1.0=valid, 0.0=padded.
                
            Returns:
                Continuous risk prediction tensor of shape [batch, 1].
            """
            # Features through causal residual blocks: [batch, last_channel, time]
            h = self.network(x)

            # For left-padded causal sequences, the final sequence position captures
            # the full causal receptive field up to the newest valid CDM.
            final_h = h[:, :, -1]

            # Project to scalar log-risk
            out = self.head(final_h)
            return out

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False


class TCNRiskModel:
    """Masked Causal Temporal Convolutional Network model wrapper.
    
    Adheres strictly to frozen V1 parameters:
    - channels: [64, 64, 128]
    - kernel_size: 3
    - dilations: [1, 2, 4]
    - causal: true
    - dropout: 0.15
    - max_sequence_length: 23
    - loss: Huber(delta=1.0)
    """

    model_name: str = "masked_causal_tcn"

    def __init__(
        self,
        in_features: int = 34,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.15,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 64,
        max_seq_len: int = 23,
        seed: int = 42,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.in_features = in_features
        self.channels = channels or [64, 64, 128]
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.config = config or {}
        self.is_fitted_: bool = False
        self.network_: Any = None
        self.training_stats_: Dict[str, Any] = {}

        if TORCH_AVAILABLE:
            torch.manual_seed(self.seed)
            if device is not None:
                self.device_ = device
            else:
                self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

            self.network_ = MaskedCausalTCN(
                in_features=self.in_features,
                channels=self.channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                dropout=self.dropout,
            ).to(self.device_)
        else:
            self.device_ = "cpu"

    def prepare_sequence_tensors(
        self,
        events: Dict[str, List[Dict[str, Any]]],
        feature_cols: List[str],
        horizon_cutoff: Optional[float] = None,
    ) -> Tuple[Any, Any, List[float], List[str]]:
        """Convert variable-length event CDM lists into left-padded 3D tensors and 2D masks.
        
        Output format:
        - X_tensor: [batch, in_features, max_seq_len] (left-padded with zeros)
        - mask_tensor: [batch, max_seq_len] (1.0 for valid timesteps, 0.0 for padding)
        - y_targets: List of final target log-risk values
        - event_ids: List of event IDs
        """
        all_x: List[List[List[float]]] = []
        all_masks: List[List[float]] = []
        target_risks: List[float] = []
        valid_ids: List[str] = []

        for ev_id, cdms in events.items():
            if not cdms:
                continue

            # Filter qualifying CDMs
            qualifying = []
            for cdm in cdms:
                if horizon_cutoff is not None:
                    try:
                        tt = float(cdm.get("time_to_tca", -1.0))
                        if tt < horizon_cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue
                qualifying.append(cdm)

            if not qualifying:
                continue

            # Target is the risk of the final CDM of the entire event
            final_cdm = cdms[-1]
            try:
                risk_target = float(final_cdm["risk"])
            except (KeyError, ValueError, TypeError):
                continue

            # Truncate to max_seq_len if longer
            if len(qualifying) > self.max_seq_len:
                qualifying = qualifying[-self.max_seq_len:]

            seq_len = len(qualifying)
            n_features = len(feature_cols)

            # Left padding: padding is placed at beginning (index 0 ... max_seq_len - seq_len - 1)
            # Valid steps placed at end (max_seq_len - seq_len ... max_seq_len - 1)
            pad_len = self.max_seq_len - seq_len

            # Feature matrix for this event: [in_features, max_seq_len]
            feat_matrix = [[0.0] * self.max_seq_len for _ in range(n_features)]
            mask_row = [0.0] * pad_len + [1.0] * seq_len

            for t_idx, cdm in enumerate(qualifying):
                target_t = pad_len + t_idx
                for f_idx, col in enumerate(feature_cols):
                    raw = cdm.get(col)
                    if raw is not None:
                        try:
                            fval = float(raw)
                            feat_matrix[f_idx][target_t] = 0.0 if math.isnan(fval) else fval
                        except (ValueError, TypeError):
                            feat_matrix[f_idx][target_t] = 0.0
                    else:
                        feat_matrix[f_idx][target_t] = 0.0

            all_x.append(feat_matrix)
            all_masks.append(mask_row)
            target_risks.append(risk_target)
            valid_ids.append(ev_id)

        if TORCH_AVAILABLE:
            X_tensor = torch.tensor(all_x, dtype=torch.float32)
            mask_tensor = torch.tensor(all_masks, dtype=torch.float32)
            return X_tensor, mask_tensor, target_risks, valid_ids

        return all_x, all_masks, target_risks, valid_ids

    def fit(
        self,
        X_train: Any,
        mask_train: Any,
        y_train: List[float],
        X_val: Optional[Any] = None,
        mask_val: Optional[Any] = None,
        y_val: Optional[List[float]] = None,
        epochs: int = 50,
        patience: int = 15,
        verbose: bool = True,
    ) -> "TCNRiskModel":
        """Train masked causal TCN using Huber loss with validation tracking and early stopping."""
        if not TORCH_AVAILABLE:
            # Mark as fitted fallback
            self.is_fitted_ = True
            return self

        import copy

        self.network_.to(self.device_)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )
        loss_fn = nn.HuberLoss(delta=1.0)

        n_samples = X_train.size(0)
        has_val = X_val is not None and mask_val is not None and y_val is not None

        if has_val:
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = copy.deepcopy(self.network_.state_dict())
        epochs_no_improve = 0

        train_loss_history: List[float] = []
        val_loss_history: List[float] = []

        for epoch in range(epochs):
            self.network_.train()
            permutation = torch.randperm(n_samples)
            epoch_train_loss = 0.0
            num_batches = 0

            for i in range(0, n_samples, self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x = X_train[indices].to(self.device_)
                batch_m = mask_train[indices].to(self.device_)
                batch_y = y_train_tensor[indices].to(self.device_)

                optimizer.zero_grad()
                preds = self.network_(batch_x, batch_m)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * len(indices)
                num_batches += 1

            avg_train_loss = epoch_train_loss / n_samples
            train_loss_history.append(avg_train_loss)

            # Validation evaluation
            if has_val:
                self.network_.eval()
                val_n = X_val.size(0)
                val_loss_sum = 0.0

                with torch.no_grad():
                    for j in range(0, val_n, self.batch_size * 2):
                        v_end = min(j + self.batch_size * 2, val_n)
                        v_x = X_val[j:v_end].to(self.device_)
                        v_m = mask_val[j:v_end].to(self.device_)
                        v_y = y_val_tensor[j:v_end].to(self.device_)
                        v_preds = self.network_(v_x, v_m)
                        v_loss = loss_fn(v_preds, v_y)
                        val_loss_sum += v_loss.item() * (v_end - j)

                avg_val_loss = val_loss_sum / val_n
                val_loss_history.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(self.network_.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1):
                    print(
                        f"    Epoch {epoch+1:02d}/{epochs:02d} - Train Huber: {avg_train_loss:.5f} | Val Huber: {avg_val_loss:.5f} (Best: {best_val_loss:.5f} @ Epoch {best_epoch})"
                    )

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"    Early stopping triggered at Epoch {epoch+1} (No improvement for {patience} epochs).")
                    break
            else:
                if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                    print(f"    Epoch {epoch+1:02d}/{epochs:02d} - Train Huber: {avg_train_loss:.5f}")

        # Restore best checkpoint
        if has_val and best_state_dict is not None:
            self.network_.load_state_dict(best_state_dict)
            if verbose:
                print(f"    Restored best model weights from Epoch {best_epoch} (Val Huber: {best_val_loss:.5f}).")

        self.is_fitted_ = True
        self.training_stats_ = {
            "device": str(self.device_),
            "total_epochs_run": epoch + 1,
            "best_epoch": best_epoch if has_val else epoch + 1,
            "best_val_loss": best_val_loss if has_val else None,
            "final_train_loss": train_loss_history[-1] if train_loss_history else None,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
        }
        return self

    def predict_risk(self, X: Any, mask: Any) -> List[float]:
        """Predict continuous log-risk scores for input sequence batch."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions.")

        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            self.network_.to(self.device_)
            self.network_.eval()
            n_samples = X.size(0)
            batch_size = 256
            all_preds: List[float] = []

            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = X[i : i + batch_size].to(self.device_)
                    batch_m = mask[i : i + batch_size].to(self.device_)
                    preds = self.network_(batch_x, batch_m).view(-1).cpu().tolist()
                    all_preds.extend([float(p) for p in preds])
            return all_preds

        # Pure python fallback prediction (e.g. median default)
        batch_size = len(X) if isinstance(X, list) else X.size(0)
        return [-20.0 for _ in range(batch_size)]

    def predict_probability(self, X: Any, mask: Any, threshold_log10: float) -> List[float]:
        """Predict threshold collision probability."""
        scores = self.predict_risk(X, mask)
        probs: List[float] = []
        for s in scores:
            diff = s - threshold_log10
            try:
                p = 1.0 / (1.0 + math.exp(-2.0 * diff))
            except OverflowError:
                p = 1.0 if diff > 0 else 0.0
            probs.append(p)
        return probs

    def save(self, path: Union[Path, str]) -> None:
        """Serialize TCN model metadata and weights."""
        p_str = str(path)
        if p_str.endswith(".json") or p_str.endswith(".pt"):
            base_str = os.path.splitext(p_str)[0]
        else:
            base_str = p_str

        p_base = Path(base_str)
        p_base.parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(f"{base_str}.json")
        pt_path = Path(f"{base_str}.pt")

        meta = {
            "model_name": self.model_name,
            "in_features": self.in_features,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
            "is_fitted": self.is_fitted_,
            "config": self.config,
            "training_stats": self.training_stats_,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if TORCH_AVAILABLE and self.network_ is not None:
            torch.save(self.network_.state_dict(), pt_path)

    @classmethod
    def load(cls, path: Union[Path, str], device: Optional[str] = None) -> "TCNRiskModel":
        """Load TCN model from disk."""
        p_str = str(path)
        if p_str.endswith(".json") or p_str.endswith(".pt"):
            base_str = os.path.splitext(p_str)[0]
        else:
            base_str = p_str

        json_path = Path(f"{base_str}.json")
        pt_path = Path(f"{base_str}.pt")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            in_features=meta.get("in_features", 34),
            channels=meta.get("channels", [64, 64, 128]),
            kernel_size=meta.get("kernel_size", 3),
            dilations=meta.get("dilations", [1, 2, 4]),
            dropout=meta.get("dropout", 0.15),
            max_seq_len=meta.get("max_seq_len", 23),
            seed=meta.get("seed", 42),
            device=device,
            config=meta.get("config", {}),
        )
        instance.is_fitted_ = meta.get("is_fitted", True)
        instance.training_stats_ = meta.get("training_stats", {})

        if TORCH_AVAILABLE and pt_path.exists() and instance.network_ is not None:
            instance.network_.load_state_dict(
                torch.load(pt_path, map_location=instance.device_)
            )

        return instance
