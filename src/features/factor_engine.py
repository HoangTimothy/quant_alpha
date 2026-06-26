"""
FactorEngine — compute 40+ basic alpha factors from OHLCV data.

All computations are performed per-ticker via groupby to avoid
cross-ticker contamination. Uses the `ta` library for standard
technical indicators and numpy/pandas for custom calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from src.utils import get_logger

logger = get_logger(__name__)


class FactorEngine:
    """Compute basic alpha factors from a MultiIndex (date, ticker) OHLCV DataFrame."""

    def __init__(
        self,
        enable_returns: bool = True,
        enable_trend: bool = True,
        enable_momentum: bool = True,
        enable_volatility: bool = True,
        enable_volume: bool = True,
        enable_price: bool = True,
        enable_time: bool = True,
        enable_lag: bool = True,
        enable_cross_sectional: bool = True,
    ) -> None:
        self.flags = {
            "returns": enable_returns,
            "trend": enable_trend,
            "momentum": enable_momentum,
            "volatility": enable_volatility,
            "volume": enable_volume,
            "price": enable_price,
            "time": enable_time,
            "lag": enable_lag,
            "cross_sectional": enable_cross_sectional,
        }

    # ── Main entry point ──────────────────────────────────────────────
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all enabled factor groups.

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex (date, ticker) with columns [open, high, low, close, volume, adj_close].

        Returns
        -------
        pd.DataFrame
            Original data + all computed factors.
        """
        result = df.copy()

        dispatch = {
            "returns": self._returns,
            "trend": self._trend,
            "momentum": self._momentum,
            "volatility": self._volatility,
            "volume": self._volume,
            "price": self._price,
            "time": self._time,
            "lag": self._lag,
        }

        for group, fn in dispatch.items():
            if self.flags.get(group, True):
                logger.info("Computing %s factors…", group)
                result = result.groupby(level="ticker", group_keys=False).apply(fn)

        # Cross-sectional (per-date) must run AFTER per-ticker factors
        if self.flags.get("cross_sectional", True):
            logger.info("Computing cross-sectional factors…")
            result = self._cross_sectional(result)

        logger.info("Total features: %d", result.shape[1] - len(df.columns))
        return result

    # ── Returns ───────────────────────────────────────────────────────
    @staticmethod
    def _returns(g: pd.DataFrame) -> pd.DataFrame:
        close = g["adj_close"]
        g["return_1d"] = close.pct_change(1)
        g["return_5d"] = close.pct_change(5)
        g["return_20d"] = close.pct_change(20)
        g["log_return_1d"] = np.log1p(g["return_1d"])
        return g

    # ── Trend ─────────────────────────────────────────────────────────
    @staticmethod
    def _trend(g: pd.DataFrame) -> pd.DataFrame:
        close = g["adj_close"]

        # Simple Moving Averages
        for w in [5, 10, 20, 50]:
            g[f"sma_{w}"] = close.rolling(w).mean()
            g[f"sma_ratio_{w}"] = close / g[f"sma_{w}"]

        # Exponential Moving Averages
        for span in [12, 26]:
            g[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

        # MACD
        macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        g["macd"] = macd_indicator.macd()
        g["macd_signal"] = macd_indicator.macd_signal()
        g["macd_hist"] = macd_indicator.macd_diff()

        return g

    # ── Momentum ──────────────────────────────────────────────────────
    @staticmethod
    def _momentum(g: pd.DataFrame) -> pd.DataFrame:
        close = g["adj_close"]

        # RSI
        g["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # Rate of Change
        g["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

        # Momentum (simple price change over window)
        g["momentum_10"] = close - close.shift(10)
        g["momentum_20"] = close - close.shift(20)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            g["high"], g["low"], close, window=14, smooth_window=3
        )
        g["stoch_k"] = stoch.stoch()
        g["stoch_d"] = stoch.stoch_signal()

        return g

    # ── Volatility ────────────────────────────────────────────────────
    @staticmethod
    def _volatility(g: pd.DataFrame) -> pd.DataFrame:
        close = g["adj_close"]
        returns = close.pct_change()

        # Rolling standard deviation of returns
        for w in [5, 10, 20]:
            g[f"volatility_{w}d"] = returns.rolling(w).std()

        # ATR
        atr = ta.volatility.AverageTrueRange(g["high"], g["low"], close, window=14)
        g["atr_14"] = atr.average_true_range()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        g["bb_upper"] = bb.bollinger_hband()
        g["bb_lower"] = bb.bollinger_lband()
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / close
        g["bb_pct"] = bb.bollinger_pband()

        return g

    # ── Volume ────────────────────────────────────────────────────────
    @staticmethod
    def _volume(g: pd.DataFrame) -> pd.DataFrame:
        close = g["adj_close"]
        volume = g["volume"]

        # On-Balance Volume
        g["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # VWAP (rolling intra-period approximation)
        typical = (g["high"] + g["low"] + close) / 3.0
        g["vwap_20"] = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Volume moving average ratio
        g["volume_sma_20_ratio"] = volume / volume.rolling(20).mean()

        return g

    # ── Price-based ───────────────────────────────────────────────────
    @staticmethod
    def _price(g: pd.DataFrame) -> pd.DataFrame:
        g["high_low_spread"] = (g["high"] - g["low"]) / g["adj_close"]
        g["close_open_ratio"] = g["close"] / g["open"]
        g["upper_shadow"] = (g["high"] - np.maximum(g["open"], g["close"])) / g["adj_close"]
        g["lower_shadow"] = (np.minimum(g["open"], g["close"]) - g["low"]) / g["adj_close"]
        return g

    # ── Time features ─────────────────────────────────────────────────
    @staticmethod
    def _time(g: pd.DataFrame) -> pd.DataFrame:
        dates = g.index.get_level_values("date")
        # Cyclical encoding for weekday (0-4) and month (1-12)
        g["weekday_sin"] = np.sin(2 * np.pi * dates.weekday / 5)
        g["weekday_cos"] = np.cos(2 * np.pi * dates.weekday / 5)
        g["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
        g["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
        return g

    # ── Lag features ──────────────────────────────────────────────────
    @staticmethod
    def _lag(g: pd.DataFrame) -> pd.DataFrame:
        ret = g["adj_close"].pct_change()
        for lag in [1, 5, 10, 20]:
            g[f"lag_return_{lag}"] = ret.shift(lag)
        return g

    # ── Cross-sectional (per-date) ────────────────────────────────────
    @staticmethod
    def _cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize numeric features within each date."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude raw OHLCV and date-based features (which have 0 variance across assets on any given date)
        exclude = {
            "open", "high", "low", "close", "volume", "adj_close",
            "weekday_sin", "weekday_cos", "month_sin", "month_cos"
        }
        factor_cols = [c for c in numeric_cols if c not in exclude]

        if not factor_cols:
            return df

        grouped = df[factor_cols].groupby(level="date")
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, np.nan)

        for col in factor_cols:
            # Handle NaN std gracefully by filling with 0.0 (meaning value equals mean)
            df[f"cs_zscore_{col}"] = ((df[col] - means[col]) / stds[col]).fillna(0.0)

        return df
