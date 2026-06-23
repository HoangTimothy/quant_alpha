"""
Factor selection methods: Mutual Information, SHAP, PCA, IC ranking, RFE.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    mutual_info_classif,
    mutual_info_regression,
)

from src.utils import get_logger

logger = get_logger(__name__)


class FactorSelector:
    """Select the most predictive factors using various methods."""

    @staticmethod
    def mutual_information(
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 30,
        task: Literal["classification", "regression"] = "classification",
    ) -> list[str]:
        """Select top-k features by mutual information.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (no NaNs).
        y : pd.Series
            Target vector.
        k : int
            Number of features to select.
        task : str
            'classification' or 'regression'.

        Returns
        -------
        list[str]
            Sorted list of selected feature names (highest MI first).
        """
        X_clean = X.fillna(0)
        k = min(k, X_clean.shape[1])

        fn = mutual_info_classif if task == "classification" else mutual_info_regression
        mi_scores = fn(X_clean, y, random_state=42)

        mi_series = pd.Series(mi_scores, index=X_clean.columns).sort_values(ascending=False)
        selected = mi_series.head(k).index.tolist()
        logger.info("MI selection: top-%d features selected", k)
        return selected

    @staticmethod
    def shap_select(
        model,
        X: pd.DataFrame,
        k: int = 30,
    ) -> list[str]:
        """Select top-k features using SHAP TreeExplainer.

        Parameters
        ----------
        model : fitted tree-based model
            Must be compatible with shap.TreeExplainer (XGBoost, LightGBM, RF).
        X : pd.DataFrame
            Feature matrix for SHAP computation (subsample if large).
        k : int
            Number of features to select.

        Returns
        -------
        list[str]
            Top-k feature names by mean |SHAP value|.
        """
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — falling back to feature_importances_")
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=X.columns)
                return imp.nlargest(k).index.tolist()
            raise

        # Subsample for speed
        sample = X.sample(min(1000, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            # Multi-class: take mean across classes
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            shap_values = np.abs(shap_values)

        mean_shap = pd.Series(shap_values.mean(axis=0), index=X.columns)
        selected = mean_shap.nlargest(k).index.tolist()
        logger.info("SHAP selection: top-%d features selected", k)
        return selected

    @staticmethod
    def pca_select(
        X: pd.DataFrame,
        n_components: int | float = 0.95,
    ) -> tuple[np.ndarray, PCA]:
        """Reduce dimensionality using PCA.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        n_components : int | float
            If float, variance explained threshold. If int, exact number of components.

        Returns
        -------
        (transformed_data, fitted_pca) : tuple
        """
        from sklearn.preprocessing import StandardScaler

        X_clean = X.fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        logger.info(
            "PCA: %d components explain %.1f%% variance",
            pca.n_components_,
            pca.explained_variance_ratio_.sum() * 100,
        )
        return X_pca, pca

    @staticmethod
    def ic_ranking(
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 30,
    ) -> list[str]:
        """Select features by Information Coefficient (Spearman rank correlation with target).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target (typically future returns).
        top_k : int
            Number of features to select.

        Returns
        -------
        list[str]
            Top-k feature names by absolute IC.
        """
        ic_scores = {}
        for col in X.columns:
            mask = X[col].notna() & y.notna()
            if mask.sum() < 30:
                continue
            corr, _ = spearmanr(X.loc[mask, col], y.loc[mask])
            ic_scores[col] = abs(corr) if not np.isnan(corr) else 0.0

        ic_series = pd.Series(ic_scores).sort_values(ascending=False)
        selected = ic_series.head(top_k).index.tolist()
        logger.info(
            "IC ranking: top-%d features (best IC=%.4f)", top_k, ic_series.iloc[0] if len(ic_series) > 0 else 0
        )
        return selected

    @staticmethod
    def rfe_select(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 30,
    ) -> list[str]:
        """Recursive Feature Elimination.

        Parameters
        ----------
        model : estimator
            Must support `fit` and `feature_importances_` or `coef_`.
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target.
        k : int
            Number of features to select.

        Returns
        -------
        list[str]
            Selected feature names.
        """
        X_clean = X.fillna(0)
        k = min(k, X_clean.shape[1])

        rfe = RFE(estimator=model, n_features_to_select=k, step=5)
        rfe.fit(X_clean, y)

        selected = X_clean.columns[rfe.support_].tolist()
        logger.info("RFE selection: %d features selected", len(selected))
        return selected

    @classmethod
    def select(
        cls,
        method: str,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 30,
        model=None,
        task: str = "classification",
    ) -> list[str]:
        """Dispatch to the requested selection method.

        Parameters
        ----------
        method : str
            One of: 'mutual_info', 'shap', 'ic_ranking', 'rfe', 'pca'.
        """
        method = method.lower().replace("-", "_")

        if method == "mutual_info":
            return cls.mutual_information(X, y, k=top_k, task=task)
        elif method == "shap":
            if model is None:
                raise ValueError("SHAP selection requires a fitted model.")
            return cls.shap_select(model, X, k=top_k)
        elif method == "ic_ranking":
            return cls.ic_ranking(X, y, top_k=top_k)
        elif method == "rfe":
            if model is None:
                raise ValueError("RFE selection requires an estimator.")
            return cls.rfe_select(model, X, y, k=top_k)
        elif method == "pca":
            # PCA returns transformed data, not feature names
            _, pca = cls.pca_select(X, n_components=top_k)
            return [f"pca_{i}" for i in range(pca.n_components_)]
        else:
            raise ValueError(f"Unknown selection method: {method}")
