"""
Advanced alpha factors: realized volatility, rolling beta, downside deviation,
Shannon entropy, and Hurst exponent.

All computations run per-ticker via groupby to prevent cross-contamination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.utils import get_logger

logger = get_logger(__name__)


class AdvancedFactors:
    """Compute statistically advanced alpha factors."""

    def __init__(
        self,
        enable_realized_vol: bool = True,
        enable_rolling_beta: bool = True,
        enable_downside_deviation: bool = True,
        enable_entropy: bool = True,
        enable_hurst: bool = True,
        market_ticker: str = "SPY",
    ) -> None:
        self.flags = {
            "realized_vol": enable_realized_vol,
            "rolling_beta": enable_rolling_beta,
            "downside_deviation": enable_downside_deviation,
            "entropy": enable_entropy,
            "hurst": enable_hurst,
        }
        self.market_ticker = market_ticker

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enabled advanced factors.

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex (date, ticker) DataFrame that already has `return_1d` column
            (from FactorEngine).
        """
        result = df.copy()

        # Ensure we have daily returns
        if "return_1d" not in result.columns:
            result["return_1d"] = result.groupby(level="ticker")["adj_close"].pct_change()

        if self.flags["realized_vol"]:
            logger.info("Computing realized volatility…")
            result = result.groupby(level="ticker", group_keys=False).apply(
                self._realized_volatility
            )

        if self.flags["rolling_beta"]:
            logger.info("Computing rolling beta vs %s…", self.market_ticker)
            result = self._rolling_beta(result)

        if self.flags["downside_deviation"]:
            logger.info("Computing downside deviation…")
            result = result.groupby(level="ticker", group_keys=False).apply(
                self._downside_deviation
            )

        if self.flags["entropy"]:
            logger.info("Computing return entropy…")
            result = result.groupby(level="ticker", group_keys=False).apply(
                self._entropy
            )

        if self.flags["hurst"]:
            logger.info("Computing Hurst exponent…")
            result = result.groupby(level="ticker", group_keys=False).apply(
                self._hurst_exponent
            )

        return result

    # ── Realized Volatility ───────────────────────────────────────────
    @staticmethod
    def _realized_volatility(g: pd.DataFrame) -> pd.DataFrame:
        """Annualised realized volatility from squared log-returns."""
        log_ret = np.log1p(g["return_1d"].fillna(0))
        sq = log_ret ** 2
        for w in [5, 10, 20]:
            g[f"realized_vol_{w}d"] = np.sqrt(sq.rolling(w).sum() * (252 / w))
        return g

    # ── Rolling Beta vs Market ────────────────────────────────────────
    def _rolling_beta(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Rolling OLS beta of each stock against the market index."""
        tickers = df.index.get_level_values("ticker").unique()

        if self.market_ticker not in tickers:
            logger.warning(
                "Market ticker %s not in data — skipping rolling beta.",
                self.market_ticker,
            )
            return df

        # Extract market returns as a date-indexed Series
        market_ret = (
            df.xs(self.market_ticker, level="ticker")["return_1d"]
            .rename("market_return")
        )

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            if ticker == self.market_ticker:
                sub = df.xs(ticker, level="ticker").copy()
                sub["rolling_beta_60d"] = 1.0
                sub["ticker"] = ticker
                frames.append(sub.reset_index())
                continue

            sub = df.xs(ticker, level="ticker").copy()
            sub = sub.join(market_ret, on="date", how="left")

            cov = sub["return_1d"].rolling(window).cov(sub["market_return"])
            var = sub["market_return"].rolling(window).var()
            sub["rolling_beta_60d"] = cov / var.replace(0, np.nan)
            sub = sub.drop(columns=["market_return"], errors="ignore")
            sub["ticker"] = ticker
            frames.append(sub.reset_index())

        result = pd.concat(frames, ignore_index=True)
        result = result.set_index(["date", "ticker"]).sort_index()
        return result

    # ── Downside Deviation ────────────────────────────────────────────
    @staticmethod
    def _downside_deviation(g: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Rolling downside deviation (semi-deviation below zero)."""
        neg_ret = g["return_1d"].clip(upper=0)
        g["downside_dev_20d"] = np.sqrt((neg_ret ** 2).rolling(window).mean())
        return g

    # ── Shannon Entropy ───────────────────────────────────────────────
    @staticmethod
    def _entropy(g: pd.DataFrame, window: int = 20, n_bins: int = 10) -> pd.DataFrame:
        """Rolling Shannon entropy of return distribution."""
        returns = g["return_1d"].fillna(0)

        def _calc_entropy(x: np.ndarray) -> float:
            if len(x) < 5 or np.all(x == 0):
                return np.nan
            counts, _ = np.histogram(x, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

        g["entropy_20d"] = returns.rolling(window).apply(_calc_entropy, raw=True)
        return g

    # ── Hurst Exponent (R/S analysis) ─────────────────────────────────
    @staticmethod
    def _hurst_exponent(g: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """Rolling Hurst exponent via simplified R/S analysis.

        H < 0.5  → mean-reverting
        H = 0.5  → random walk
        H > 0.5  → trending
        """
        returns = g["return_1d"].fillna(0).values
        hurst_vals = np.full(len(returns), np.nan)

        for i in range(window, len(returns)):
            ts = returns[i - window : i]
            hurst_vals[i] = AdvancedFactors._rs_hurst(ts)

        g["hurst_100d"] = hurst_vals
        return g

    @staticmethod
    def _rs_hurst(ts: np.ndarray) -> float:
        """Compute Hurst exponent from a single time series window."""
        n = len(ts)
        if n < 20:
            return np.nan

        max_k = min(n // 2, 50)
        sizes = []
        rs_values = []

        for k in [max_k // 4, max_k // 2, max_k]:
            if k < 4:
                continue
            n_chunks = n // k
            if n_chunks < 1:
                continue

            rs_list = []
            for j in range(n_chunks):
                chunk = ts[j * k : (j + 1) * k]
                mean_val = chunk.mean()
                deviate = np.cumsum(chunk - mean_val)
                r = deviate.max() - deviate.min()
                s = chunk.std(ddof=1)
                if s > 1e-10:
                    rs_list.append(r / s)

            if rs_list:
                sizes.append(k)
                rs_values.append(np.mean(rs_list))

        if len(sizes) < 2:
            return np.nan

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        slope, _, _, _, _ = sp_stats.linregress(log_sizes, log_rs)
        return float(np.clip(slope, 0, 1))
