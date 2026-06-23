"""
Portfolio construction: equal weight, volatility targeting, Kelly criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class PortfolioConstructor:
    """Construct portfolio weights from model signals."""

    @staticmethod
    def equal_weight(
        signals: pd.DataFrame,
        long_only: bool = False,
    ) -> pd.DataFrame:
        """Equal-weight allocation: 1/N among selected assets.

        Parameters
        ----------
        signals : pd.DataFrame
            (dates × tickers) signal matrix. Positive = long, negative = short.
        long_only : bool
            If True, only take long positions.

        Returns
        -------
        pd.DataFrame
            Portfolio weights summing to 1 (long-only) or net-neutral (long-short).
        """
        if long_only:
            selected = (signals > 0).astype(float)
        else:
            selected = signals.copy()
            selected[selected > 0] = 1.0
            selected[selected < 0] = -1.0

        # Normalise per row
        row_sums = selected.abs().sum(axis=1).replace(0, 1)
        weights = selected.div(row_sums, axis=0)
        return weights

    @staticmethod
    def vol_targeting(
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        target_vol: float = 0.15,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """Inverse-volatility weighted positions scaled to target annual vol.

        Each asset weight ∝ 1/σ_i, then the portfolio is scaled so that
        the estimated portfolio volatility equals target_vol.
        """
        returns = prices.pct_change()
        rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
        inv_vol = 1.0 / rolling_vol.replace(0, np.nan)
        inv_vol = inv_vol.fillna(0)

        # Apply signal direction
        direction = signals.copy()
        direction[direction > 0] = 1.0
        direction[direction < 0] = -1.0
        direction[direction == 0] = 0.0

        raw_weights = direction * inv_vol
        row_sums = raw_weights.abs().sum(axis=1).replace(0, 1)
        weights = raw_weights.div(row_sums, axis=0)

        # Scale to target vol
        port_vol = (weights.shift(1) * returns).sum(axis=1).rolling(lookback).std() * np.sqrt(252)
        vol_scalar = target_vol / port_vol.replace(0, np.nan).fillna(target_vol)
        vol_scalar = vol_scalar.clip(0, 2)

        weights = weights.mul(vol_scalar, axis=0)
        return weights

    @staticmethod
    def kelly_criterion(
        signals: pd.DataFrame,
        expected_returns: pd.DataFrame,
        covariance: pd.DataFrame | None = None,
        half_kelly: bool = True,
    ) -> pd.DataFrame:
        """Kelly criterion position sizing.

        For the simplified case: f_i = μ_i / σ_i²
        If half_kelly is True, use f_i / 2 (more conservative).

        Parameters
        ----------
        signals : pd.DataFrame
            Signal direction.
        expected_returns : pd.DataFrame
            Expected return per asset (same shape as signals).
        covariance : pd.DataFrame, optional
            Covariance matrix. If None, uses diagonal (variance only).
        half_kelly : bool
            Use half-Kelly fraction for safety.

        Returns
        -------
        pd.DataFrame
            Kelly-optimal weights.
        """
        # Simplified per-asset Kelly
        variance = expected_returns.rolling(60).var().replace(0, np.nan).fillna(1e-4)
        kelly_frac = expected_returns / variance

        if half_kelly:
            kelly_frac = kelly_frac / 2.0

        # Apply signal direction
        direction = signals.copy()
        direction[direction > 0] = 1.0
        direction[direction < 0] = -1.0
        direction[direction == 0] = 0.0

        weights = direction * kelly_frac.abs().clip(0, 1)

        # Normalize
        row_sums = weights.abs().sum(axis=1).replace(0, 1)
        weights = weights.div(row_sums, axis=0)

        return weights
