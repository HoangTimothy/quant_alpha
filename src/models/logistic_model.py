"""Logistic Regression model wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class LogisticModel(BaseModel):
    name = "logistic"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "LogisticModel":
        X_scaled = self.scaler.fit_transform(X_train)
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None

        self.model = LogisticRegression(**self.params)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        coefs = np.abs(self.model.coef_[0])
        names = self.feature_names_ or [f"f{i}" for i in range(len(coefs))]
        return dict(zip(names, coefs))
