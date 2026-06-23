"""
Shared test fixtures — synthetic market data for all test modules.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """Generate synthetic OHLCV data with MultiIndex (date, ticker)."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    tickers = ["AAPL", "MSFT", "GOOGL"]

    frames = []
    for ticker in tickers:
        # Geometric Brownian Motion for realistic prices
        returns = np.random.normal(0.0005, 0.02, len(dates))
        close = 100 * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        open_ = close * (1 + np.random.normal(0, 0.005, len(dates)))
        volume = np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float)

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": close,
        }, index=dates)
        df["ticker"] = ticker
        df.index.name = "date"
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.set_index(["date", "ticker"]).sort_index()
    return combined


@pytest.fixture
def feature_matrix(synthetic_ohlcv):
    """Generate feature matrix from synthetic data."""
    np.random.seed(42)
    n = len(synthetic_ohlcv)
    features = pd.DataFrame({
        "sma_ratio_20": np.random.randn(n),
        "rsi_14": np.random.uniform(20, 80, n),
        "return_1d": np.random.randn(n) * 0.02,
        "volatility_20d": np.abs(np.random.randn(n)) * 0.02,
        "momentum_10": np.random.randn(n),
        "obv": np.random.randn(n) * 1000,
        "macd": np.random.randn(n) * 0.5,
        "bb_width": np.abs(np.random.randn(n)) * 0.05,
    }, index=synthetic_ohlcv.index)
    return features


@pytest.fixture
def binary_labels(feature_matrix):
    """Generate binary labels."""
    np.random.seed(42)
    return pd.Series(
        np.random.randint(0, 2, len(feature_matrix)),
        index=feature_matrix.index,
        name="label_binary",
    )


@pytest.fixture
def returns_series():
    """Generate synthetic daily returns."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = np.random.normal(0.0003, 0.015, len(dates))
    return pd.Series(returns, index=dates, name="returns")
