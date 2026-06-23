"""Tests for the MarketDataset data loader."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader.market_dataset import MarketDataset


class TestMarketDataset:
    """Tests for MarketDataset."""

    def test_init_defaults(self):
        ds = MarketDataset()
        assert len(ds.tickers) == 6
        assert ds.data is None

    def test_clean_missing(self, synthetic_ohlcv):
        ds = MarketDataset()
        ds.data = synthetic_ohlcv.copy()
        # Inject some NaNs
        ds.data.iloc[0, 0] = np.nan
        result = ds.clean_missing()
        assert result.notna().all().all()

    def test_split_time_series(self, synthetic_ohlcv):
        ds = MarketDataset()
        ds.data = synthetic_ohlcv.copy()

        dates = ds.data.index.get_level_values("date")
        mid1 = dates[len(dates) // 3]
        mid2 = dates[2 * len(dates) // 3]

        train, val, test = ds.split_time_series(
            train_end=str(mid1.date()),
            val_end=str(mid2.date()),
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # No overlap
        train_dates = train.index.get_level_values("date")
        val_dates = val.index.get_level_values("date")
        test_dates = test.index.get_level_values("date")

        assert train_dates.max() <= val_dates.min()
        assert val_dates.max() <= test_dates.min()

    def test_resample(self, synthetic_ohlcv):
        ds = MarketDataset()
        ds.data = synthetic_ohlcv.copy()
        original_len = len(ds.data)
        result = ds.resample("W")
        assert len(result) < original_len

    def test_multiindex_structure(self, synthetic_ohlcv):
        assert isinstance(synthetic_ohlcv.index, pd.MultiIndex)
        assert synthetic_ohlcv.index.names == ["date", "ticker"]
        assert "close" in synthetic_ohlcv.columns
        assert "volume" in synthetic_ohlcv.columns
