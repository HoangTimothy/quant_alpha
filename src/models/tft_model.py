"""
Temporal Fusion Transformer (simplified) for time-series classification.

Key components:
  1. Variable Selection Network — learns which features matter per time step
  2. LSTM Encoder/Decoder (simplified from full TFT)
  3. Multi-Head Attention for temporal patterns
  4. Gated Residual Network (GRN) blocks

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting", 2021.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel
from src.utils import get_device, get_logger

logger = get_logger(__name__)


class _GatedLinearUnit(nn.Module):
    """GLU activation: element-wise gating."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(x)) * self.fc2(x)


class _GatedResidualNetwork(nn.Module):
    """GRN block used throughout TFT."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = _GatedLinearUnit(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.elu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.glu(x)
        return self.layer_norm(x + residual)


class _VariableSelectionNetwork(nn.Module):
    """Learns feature importance weights per time step."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.grn = _GatedResidualNetwork(d_model, dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.weight_fc = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        projected = self.input_proj(x)
        grn_out = self.grn(projected)
        weights = self.softmax(self.weight_fc(grn_out))  # (batch, seq, input_dim)
        selected = x * weights
        return self.input_proj(selected), weights


class _TFTNetwork(nn.Module):
    """Simplified Temporal Fusion Transformer."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vsn = _VariableSelectionNetwork(input_dim, d_model, dropout)

        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, num_layers=num_encoder_layers,
            batch_first=True, dropout=dropout if num_encoder_layers > 1 else 0,
        )

        # Self-attention (interpretable multi-head attention)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Post-attention GRN
        self.post_attn_grn = _GatedResidualNetwork(d_model, dropout)

        # Output
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variable selection
        x, var_weights = self.vsn(x)

        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(x)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # Post-attention processing
        processed = self.post_attn_grn(attn_out)

        # Take last time step and predict
        last = processed[:, -1, :]
        return self.output_fc(last).squeeze(-1)


class TFTModel(BaseModel):
    """Temporal Fusion Transformer wrapper."""

    name = "tft"

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
    ) -> "TFTModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        n_features = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[-1]

        X_seq, y_seq = self._create_sequences(X_train, y_train)
        if X_seq is None:
            logger.warning("Not enough data for TFT sequences.")
            return self

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        self.model = _TFTNetwork(
            input_dim=n_features,
            d_model=self.params.get("d_model", 64),
            nhead=self.params.get("nhead", 4),
            num_encoder_layers=self.params.get("num_encoder_layers", 2),
            num_decoder_layers=self.params.get("num_decoder_layers", 1),
            dim_feedforward=self.params.get("dim_feedforward", 128),
            dropout=self.params.get("dropout", 0.2),
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / len(X_t)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_vt.to(self.device))
                    val_loss = criterion(val_out, y_vt.to(self.device)).item()
                scheduler.step(val_loss)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                    self.best_iteration = epoch
                else:
                    no_improve += 1
                if no_improve >= patience:
                    logger.info("TFT early stop at epoch %d", epoch)
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
        pad = np.full(self.seq_length - 1, 0.5)
        return np.concatenate([pad, probs])

    def _create_sequences(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        n = len(X)
        if n < self.seq_length:
            return None, None
        sequences = [X[i : i + self.seq_length] for i in range(n - self.seq_length + 1)]
        X_seq = np.array(sequences)
        y_seq = y[self.seq_length - 1 :] if y is not None else None
        return X_seq, y_seq
