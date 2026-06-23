"""
Statistical tests for strategy evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.utils import get_logger

logger = get_logger(__name__)


class StatisticalTests:
    """Statistical significance tests for trading strategies."""

    @staticmethod
    def sharpe_bootstrap_ci(
        returns: pd.Series,
        n_bootstrap: int = 10_000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, float]:
        """Bootstrap confidence interval for the Sharpe ratio.

        Parameters
        ----------
        returns : pd.Series
            Strategy daily returns.
        n_bootstrap : int
            Number of bootstrap samples.
        ci : float
            Confidence level (e.g. 0.95 for 95%).

        Returns
        -------
        dict
            Contains 'sharpe', 'ci_lower', 'ci_upper', 'std'.
        """
        rng = np.random.RandomState(seed)
        returns = returns.dropna().values
        n = len(returns)

        if n < 10:
            return {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

        sharpe_samples = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=n, replace=True)
            s_mean = sample.mean() * 252
            s_std = sample.std() * np.sqrt(252)
            sharpe_samples[i] = s_mean / s_std if s_std > 0 else 0.0

        alpha = (1 - ci) / 2
        lower = np.percentile(sharpe_samples, alpha * 100)
        upper = np.percentile(sharpe_samples, (1 - alpha) * 100)
        point_sharpe = returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        return {
            "sharpe": float(point_sharpe),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "std": float(sharpe_samples.std()),
        }

    @staticmethod
    def mean_return_ttest(returns: pd.Series) -> dict[str, float]:
        """One-sample t-test: H0: mean daily return = 0.

        Returns
        -------
        dict
            Contains 't_stat', 'p_value', 'significant_5pct'.
        """
        returns = returns.dropna().values
        if len(returns) < 5:
            return {"t_stat": 0.0, "p_value": 1.0, "significant_5pct": False}

        t_stat, p_value = sp_stats.ttest_1samp(returns, 0)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant_5pct": bool(p_value < 0.05),
        }

    @staticmethod
    def paired_model_ttest(
        returns_a: pd.Series,
        returns_b: pd.Series,
    ) -> dict[str, float]:
        """Paired t-test: are two strategies' returns significantly different?

        Returns
        -------
        dict
            Contains 't_stat', 'p_value', 'significant_5pct', 'mean_diff'.
        """
        common = returns_a.index.intersection(returns_b.index)
        a = returns_a.loc[common].dropna()
        b = returns_b.loc[common].dropna()
        common2 = a.index.intersection(b.index)
        a, b = a.loc[common2], b.loc[common2]

        if len(a) < 5:
            return {"t_stat": 0.0, "p_value": 1.0, "significant_5pct": False, "mean_diff": 0.0}

        t_stat, p_value = sp_stats.ttest_rel(a, b)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant_5pct": bool(p_value < 0.05),
            "mean_diff": float((a - b).mean()),
        }

    @classmethod
    def full_statistical_report(
        cls,
        returns: pd.Series,
        n_bootstrap: int = 5000,
    ) -> dict:
        """Run all statistical tests on strategy returns."""
        return {
            "sharpe_bootstrap": cls.sharpe_bootstrap_ci(returns, n_bootstrap=n_bootstrap),
            "mean_return_ttest": cls.mean_return_ttest(returns),
        }
