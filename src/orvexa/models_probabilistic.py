"""Probabilistic Temporal Convolutional Network (TCN) for Conjunction Risk Prediction.

Implements multi-quantile regression with guaranteed monotonic non-crossing parameterization
and full compatibility with the M4 37-channel input representation.

References:
- Koenker, R., & Bassett, G. (1978). "Regression Quantiles", Econometrica.
- Bai, Kolter, & Koltun (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271.
- Cannon, A. J. (2018). "Non-crossing neural network quantile regression", Stochastic Environmental Research and Risk Assessment.
"""

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from orvexa.losses_quantile import (
    compute_quantile_evaluation_metrics,
    multi_quantile_pinball_loss_numpy,
)
from orvexa.models_tcn import (
    CausalConv1d,
    TemporalResidualBlock,
    causal_conv1d_forward_numpy,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


if TORCH_AVAILABLE:

    class QuantileMaskedCausalTCN(nn.Module):
        """Causal TCN network with guaranteed monotonic multi-quantile output head.
        
        Uses cumulative softplus step increments to strictly prevent quantile crossing:
            q_0 = Linear_0(h)
            q_k = q_{k-1} + Softplus(Linear_k(h)) + epsilon
        """

        def __init__(
            self,
            in_features: int = 37,
            quantiles: Optional[List[float]] = None,
            channels: Optional[List[int]] = None,
            kernel_size: int = 3,
            dilations: Optional[List[int]] = None,
            dropout: float = 0.15,
            enforce_monotonic: bool = True,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.quantiles = sorted(quantiles or DEFAULT_QUANTILES)
            self.n_quantiles = len(self.quantiles)
            self.enforce_monotonic = enforce_monotonic

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
            self.final_channel = current_in

            if self.enforce_monotonic:
                # Base quantile (lowest tau, e.g. 0.05)
                self.base_head = nn.Linear(current_in, 1)
                # Incremental positive step heads for subsequent quantiles
                self.step_heads = nn.ModuleList(
                    [nn.Linear(current_in, 1) for _ in range(self.n_quantiles - 1)]
                )
            else:
                # Unconstrained multi-head projection
                self.unconstrained_head = nn.Linear(current_in, self.n_quantiles)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Forward pass returning [batch_size, n_quantiles] tensor.
            
            Args:
                x: Input tensor [batch, in_features, time].
                mask: Optional validity mask [batch, time].
                
            Returns:
                Multi-quantile prediction tensor [batch, n_quantiles].
            """
            h = self.network(x)
            # Final sequence timestep captures full causal receptive field
            final_h = h[:, :, -1]

            if self.enforce_monotonic:
                q0 = self.base_head(final_h)  # [batch, 1]
                q_list = [q0]
                current_q = q0
                for step_head in self.step_heads:
                    delta = F.softplus(step_head(final_h)) + 1e-6
                    current_q = current_q + delta
                    q_list.append(current_q)
                out = torch.cat(q_list, dim=-1)  # [batch, n_quantiles]
            else:
                out = self.unconstrained_head(final_h)

            return out


class QuantileTCNRiskModel:
    """Multi-quantile Causal Temporal Convolutional Network model wrapper for Phase 5.
    
    Supports:
    - Multi-quantile point and distribution forecasting (Pinball loss)
    - Monotonic non-crossing output constraints
    - Left-padded causal sequences
    - PyTorch training with early stopping and learning rate scheduling
    """

    model_name: str = "quantile_causal_tcn"

    def __init__(
        self,
        in_features: int = 37,
        quantiles: Optional[List[float]] = None,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.15,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 64,
        max_seq_len: int = 23,
        enforce_monotonic: bool = True,
        seed: int = 42,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.in_features = in_features
        self.quantiles = sorted(quantiles or DEFAULT_QUANTILES)
        self.n_quantiles = len(self.quantiles)
        self.channels = channels or [64, 64, 128]
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.enforce_monotonic = enforce_monotonic
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
        else:
            self.device_ = "cpu"

    def fit(
        self,
        X_train: Any,
        mask_train: Optional[Any],
        y_train: Union[List[float], np.ndarray, Any],
        X_val: Optional[Any] = None,
        mask_val: Optional[Any] = None,
        y_val: Optional[Union[List[float], np.ndarray, Any]] = None,
        epochs: int = 50,
        patience: int = 15,
        verbose: bool = False,
    ) -> "QuantileTCNRiskModel":
        """Fit Quantile TCN on Phase 5 training sequence tensors.
        
        Args:
            X_train: Torch tensor or numpy array of shape [N_train, in_features, time].
            mask_train: Validity mask tensor [N_train, time].
            y_train: Target log-risk values of shape [N_train].
            X_val: Validation sequence tensor [N_val, in_features, time].
            mask_val: Validation mask tensor [N_val, time].
            y_val: Validation target log-risks [N_val].
            epochs: Maximum training epochs.
            patience: Early stopping patience based on validation mean pinball loss.
            verbose: Logging verbosity.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to fit QuantileTCNRiskModel.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Convert inputs to torch tensors
        if not isinstance(X_train, torch.Tensor):
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
        else:
            X_train_t = X_train.float()

        if not isinstance(y_train, torch.Tensor):
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        else:
            y_train_t = y_train.float().view(-1, 1)

        has_val = X_val is not None and y_val is not None
        if has_val:
            if not isinstance(X_val, torch.Tensor):
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
            else:
                X_val_t = X_val.float()
            if not isinstance(y_val, torch.Tensor):
                y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            else:
                y_val_t = y_val.float().view(-1, 1)

        # Initialize network
        self.network_ = QuantileMaskedCausalTCN(
            in_features=self.in_features,
            quantiles=self.quantiles,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout,
            enforce_monotonic=self.enforce_monotonic,
        ).to(self.device_)

        from orvexa.losses_quantile import MultiQuantileLoss
        criterion = MultiQuantileLoss(self.quantiles).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )

        n_train = len(X_train_t)
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            self.network_.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, self.batch_size):
                indices = perm[i : i + self.batch_size]
                xb = X_train_t[indices].to(self.device_)
                yb = y_train_t[indices].to(self.device_)

                optimizer.zero_grad()
                preds = self.network_(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(indices)
                n_batches += len(indices)

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            if has_val:
                self.network_.eval()
                with torch.no_grad():
                    xv = X_val_t.to(self.device_)
                    yv = y_val_t.to(self.device_)
                    val_preds = self.network_(xv)
                    val_loss = criterion(val_preds, yv).item()
                    val_losses.append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.network_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == epochs):
                    print(
                        f"Epoch {epoch:3d}/{epochs} | Train Pinball: {train_loss:.5f} | Val Pinball: {val_loss:.5f} | Best: {best_val_loss:.5f} (ep {best_epoch})"
                    )

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break
            else:
                best_epoch = epoch
                best_val_loss = train_loss
                best_state = copy.deepcopy(self.network_.state_dict())

        # Load best weights
        if best_state is not None:
            self.network_.load_state_dict(best_state)

        self.is_fitted_ = True
        self.training_stats_ = {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "total_epochs_trained": len(train_losses),
            "final_train_loss": round(train_losses[-1], 6),
            "train_loss_history": [round(x, 6) for x in train_losses],
            "val_loss_history": [round(x, 6) for x in val_losses] if has_val else [],
        }

        return self

    def predict_quantiles(self, X: Any, mask: Optional[Any] = None) -> np.ndarray:
        """Predict all quantiles for input sequence tensor.
        
        Args:
            X: Input sequence tensor [N, in_features, time] or numpy array.
            mask: Optional validity mask.
            
        Returns:
            Numpy array of shape [N, n_quantiles].
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted.")

        if not isinstance(X, torch.Tensor):
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X.float()

        self.network_.eval()
        with torch.no_grad():
            preds = self.network_(X_t.to(self.device_))
            return preds.cpu().numpy()

    def predict_risk(self, X: Any, mask: Optional[Any] = None) -> List[float]:
        """Return median quantile (tau=0.50) as continuous scalar risk point estimate."""
        q_preds = self.predict_quantiles(X, mask=mask)
        if 0.50 in self.quantiles:
            idx_med = self.quantiles.index(0.50)
        else:
            idx_med = len(self.quantiles) // 2
        return [float(x) for x in q_preds[:, idx_med]]

    def save(self, path: Union[Path, str]) -> None:
        """Save model configuration and weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "model_name": self.model_name,
            "in_features": self.in_features,
            "quantiles": self.quantiles,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "enforce_monotonic": self.enforce_monotonic,
            "seed": self.seed,
            "training_stats": self.training_stats_,
            "config": self.config,
        }

        # Save JSON config
        json_path = p.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Save PyTorch weights
        if TORCH_AVAILABLE and self.network_ is not None:
            pt_path = p.with_suffix(".pt")
            torch.save(self.network_.state_dict(), pt_path)

    @classmethod
    def load(cls, path: Union[Path, str], device: Optional[str] = None) -> "QuantileTCNRiskModel":
        """Load model configuration and weights from disk."""
        p = Path(path)
        json_path = p.with_suffix(".json")
        pt_path = p.with_suffix(".pt")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(
            in_features=data.get("in_features", 37),
            quantiles=data.get("quantiles", DEFAULT_QUANTILES),
            channels=data.get("channels", [64, 64, 128]),
            kernel_size=data.get("kernel_size", 3),
            dilations=data.get("dilations", [1, 2, 4]),
            dropout=data.get("dropout", 0.15),
            learning_rate=data.get("learning_rate", 0.001),
            weight_decay=data.get("weight_decay", 0.0001),
            batch_size=data.get("batch_size", 64),
            max_seq_len=data.get("max_seq_len", 23),
            enforce_monotonic=data.get("enforce_monotonic", True),
            seed=data.get("seed", 42),
            device=device,
            config=data.get("config", {}),
        )
        model.training_stats_ = data.get("training_stats", {})

        if TORCH_AVAILABLE and pt_path.exists():
            model.network_ = QuantileMaskedCausalTCN(
                in_features=model.in_features,
                quantiles=model.quantiles,
                channels=model.channels,
                kernel_size=model.kernel_size,
                dilations=model.dilations,
                dropout=model.dropout,
                enforce_monotonic=model.enforce_monotonic,
            ).to(model.device_)
            state_dict = torch.load(pt_path, map_location=model.device_)
            model.network_.load_state_dict(state_dict)
            model.network_.eval()
            model.is_fitted_ = True

        return model
