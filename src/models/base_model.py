"""
BaseModel — abstract interface for all ML/DL models in the pipeline.

Provides: fit / predict / predict_proba / save / load / checkpoint support.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all quant_alpha models."""

    name: str = "base"

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        self.best_iteration: int | None = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "BaseModel":
        """Train the model. Must set self.is_fitted = True."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point predictions."""
        ...

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return probability predictions (for classifiers).

        Default implementation returns predict() output —
        override in classification models.
        """
        return self.predict(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to disk."""
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "name": self.name,
                    "params": self.params,
                    "model": self.model,
                    "is_fitted": self.is_fitted,
                    "best_iteration": self.best_iteration,
                },
                f,
            )
        logger.info("Model saved → %s", path)

    def load(self, path: str | Path) -> "BaseModel":
        """Load a fitted model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.name = state["name"]
        self.params = state["params"]
        self.model = state["model"]
        self.is_fitted = state["is_fitted"]
        self.best_iteration = state.get("best_iteration")
        logger.info("Model loaded ← %s", path)
        return self

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances if available. Override in subclasses."""
        if hasattr(self.model, "feature_importances_"):
            return dict(
                zip(
                    getattr(self, "feature_names_", [f"f{i}" for i in range(len(self.model.feature_importances_))]),
                    self.model.feature_importances_,
                )
            )
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"
