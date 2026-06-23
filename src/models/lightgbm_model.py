"""LightGBM model wrapper with early stopping support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .base_model import BaseModel
from src.utils import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "LightGBMModel":
        self.feature_names_ = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        early_stopping_rounds = self.params.get("early_stopping_rounds", 50)

        self.model = lgb.LGBMClassifier(**params)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=0),  # suppress verbose output
            ]

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.best_iteration = getattr(self.model, "best_iteration_", None)
        self.is_fitted = True

        if self.best_iteration is not None:
            logger.info("LightGBM best iteration: %d", self.best_iteration)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
