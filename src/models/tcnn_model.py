"""
Temporal CNN — 1D convolution over time windows for sequence classification.

Architecture: [Conv1d → BatchNorm → ReLU → Dropout] × N → AdaptiveAvgPool → FC → Output

Input shape: (batch, seq_length, n_features) — the model permutes to (batch, features, seq_len)
for Conv1d which expects channels-first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel
from src.utils import get_device, get_logger

logger = get_logger(__name__)


class _TemporalCNNNetwork(nn.Module):
    """1D temporal CNN."""

    def __init__(
        self,
        input_channels: int,
        channels: list[int] = (64, 128, 64),
        kernel_sizes: list[int] = (3, 3, 3),
        dropout: float = 0.3,
        fc_dim: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_channels

        for out_ch, ks in zip(channels, kernel_sizes):
            padding = ks // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        return self.fc(x).squeeze(-1)


class TemporalCNNModel(BaseModel):
    """Temporal CNN wrapper for sequence classification."""

    name = "tcnn"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.device = get_device()
        self.seq_length = self.params.get("seq_length", 20)

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 10,
    ) -> "TemporalCNNModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        n_features = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[-1]

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_train, y_train)
        if X_seq is None:
            logger.warning("Not enough data for sequence creation — falling back to flat input.")
            return self

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        channels = self.params.get("channels", [64, 128, 64])
        kernel_sizes = self.params.get("kernel_sizes", [3, 3, 3])
        dropout = self.params.get("dropout", 0.3)
        fc_dim = self.params.get("fc_dim", 64)

        self.model = _TemporalCNNNetwork(
            input_channels=n_features,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            fc_dim=fc_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
        )

        # Validation
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_vseq, y_vseq = self._create_sequences(X_val, y_val)
            if X_vseq is not None:
                X_vt = torch.tensor(X_vseq, dtype=torch.float32)
                y_vt = torch.tensor(y_vseq, dtype=torch.float32)
            else:
                has_val = False

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_vt.to(self.device))
                    val_loss = criterion(val_out, y_vt.to(self.device)).item()
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                    self.best_iteration = epoch
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.info("TCNN early stop at epoch %d", epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_seq, _ = self._create_sequences(X, None)
        if X_seq is None:
            return np.full(len(X), 0.5)
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        # Pad beginning with 0.5 for the sequence window
        pad = np.full(self.seq_length - 1, 0.5)
        return np.concatenate([pad, probs])

    def _create_sequences(
        self, X, y=None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Create sliding window sequences from flat feature matrix."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values

        n = len(X)
        seq_len = self.seq_length
        if n < seq_len:
            return None, None

        sequences = []
        for i in range(n - seq_len + 1):
            sequences.append(X[i : i + seq_len])

        X_seq = np.array(sequences)

        if y is not None:
            y_seq = y[seq_len - 1 :]
            return X_seq, y_seq

        return X_seq, None
