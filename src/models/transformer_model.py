"""
Transformer Encoder for time-series classification.

Architecture:
  Input projection → Positional Encoding → TransformerEncoder × N → Mean Pool → FC → Output

Uses PyTorch's native TransformerEncoder with sinusoidal positional encoding.
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


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TransformerNetwork(nn.Module):
    """Transformer Encoder for time-series."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        # Mean pool across sequence dimension
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


class TransformerModel(BaseModel):
    """Transformer Encoder wrapper implementing BaseModel interface."""

    name = "transformer"

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
    ) -> "TransformerModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        n_features = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[-1]

        X_seq, y_seq = self._create_sequences(X_train, y_train)
        if X_seq is None:
            logger.warning("Not enough data for Transformer sequences.")
            return self

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        d_model = self.params.get("d_model", 64)
        nhead = self.params.get("nhead", 4)
        num_layers = self.params.get("num_layers", 2)
        dim_feedforward = self.params.get("dim_feedforward", 128)
        dropout = self.params.get("dropout", 0.2)

        self.model = _TransformerNetwork(
            input_dim=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            scheduler.step()

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
                    logger.info("Transformer early stop at epoch %d", epoch)
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
