"""
MLP (Multi-Layer Perceptron) — PyTorch implementation.

Architecture: Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
Supports binary classification and regression via configurable output activation.
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


class _MLPNetwork(nn.Module):
    """PyTorch MLP network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPModel(BaseModel):
    """MLP wrapper implementing the BaseModel interface."""

    name = "mlp"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.device = get_device()
        self.epochs = 100
        self.batch_size = 256
        self.lr = 0.001
        self.weight_decay = 0.0001
        self.patience = 10

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        patience: int | None = None,
    ) -> "MLPModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None

        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        lr = lr or self.lr
        weight_decay = weight_decay or self.weight_decay
        patience = patience or self.patience

        X_t = self._to_tensor(X_train)
        y_t = self._to_tensor(y_train, dtype=torch.float32)

        input_dim = X_t.shape[1]
        hidden_dims = self.params.get("hidden_dims", [256, 128, 64])
        dropout = self.params.get("dropout", 0.3)
        batch_norm = self.params.get("batch_norm", True)

        self.model = _MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # Validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = self._to_tensor(X_val)
            y_v = self._to_tensor(y_val, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_t)

            if has_val:
                val_loss = self._evaluate_loss(X_v, y_v, criterion)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                    self.best_iteration = epoch
                else:
                    no_improve += 1

                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d (best=%d)", epoch, self.best_iteration)
                    break

            if epoch % 20 == 0:
                val_msg = f", val_loss={val_loss:.5f}" if has_val else ""
                logger.info("Epoch %d/%d — train_loss=%.5f%s", epoch, epochs, train_loss, val_msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_t = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _evaluate_loss(
        self, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            loss = criterion(logits, y.to(self.device))
        return loss.item()

    def _to_tensor(
        self, data: np.ndarray | pd.DataFrame | pd.Series, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        return torch.tensor(data, dtype=dtype)
