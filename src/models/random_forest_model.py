"""Random Forest model wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "RandomForestModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
