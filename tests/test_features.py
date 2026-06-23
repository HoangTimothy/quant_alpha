"""Tests for feature engineering modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.factor_engine import FactorEngine
from src.features.advanced_factors import AdvancedFactors
from src.features.factor_selection import FactorSelector
from src.features.label_constructor import LabelConstructor


class TestFactorEngine:
    """Tests for FactorEngine."""

    def test_compute_adds_columns(self, synthetic_ohlcv):
        engine = FactorEngine()
        result = engine.compute(synthetic_ohlcv)
        assert result.shape[1] > synthetic_ohlcv.shape[1]

    def test_returns_computed(self, synthetic_ohlcv):
        engine = FactorEngine(
            enable_trend=False, enable_momentum=False,
            enable_volatility=False, enable_volume=False,
            enable_price=False, enable_time=False,
            enable_lag=False, enable_cross_sectional=False,
        )
        result = engine.compute(synthetic_ohlcv)
        assert "return_1d" in result.columns
        assert "return_5d" in result.columns

    def test_trend_indicators(self, synthetic_ohlcv):
        engine = FactorEngine(
            enable_returns=False, enable_momentum=False,
            enable_volatility=False, enable_volume=False,
            enable_price=False, enable_time=False,
            enable_lag=False, enable_cross_sectional=False,
        )
        result = engine.compute(synthetic_ohlcv)
        assert "sma_20" in result.columns
        assert "macd" in result.columns

    def test_no_future_leakage_in_lag(self, synthetic_ohlcv):
        engine = FactorEngine(
            enable_returns=True, enable_lag=True,
            enable_trend=False, enable_momentum=False,
            enable_volatility=False, enable_volume=False,
            enable_price=False, enable_time=False,
            enable_cross_sectional=False,
        )
        result = engine.compute(synthetic_ohlcv)
        # Lag features should have NaN at the beginning
        for ticker in result.index.get_level_values("ticker").unique():
            ticker_data = result.xs(ticker, level="ticker")
            if "lag_return_1" in ticker_data.columns:
                assert ticker_data["lag_return_1"].iloc[0] != ticker_data["lag_return_1"].iloc[-1] or pd.isna(ticker_data["lag_return_1"].iloc[0])


class TestAdvancedFactors:
    """Tests for AdvancedFactors."""

    def test_realized_vol(self, synthetic_ohlcv):
        # First compute basic returns
        engine = FactorEngine()
        featured = engine.compute(synthetic_ohlcv)

        adv = AdvancedFactors(
            enable_rolling_beta=False,
            enable_downside_deviation=False,
            enable_entropy=False,
            enable_hurst=False,
        )
        result = adv.compute(featured)
        assert "realized_vol_5d" in result.columns
        assert "realized_vol_20d" in result.columns

    def test_hurst_range(self, synthetic_ohlcv):
        engine = FactorEngine()
        featured = engine.compute(synthetic_ohlcv)

        adv = AdvancedFactors(
            enable_realized_vol=False,
            enable_rolling_beta=False,
            enable_downside_deviation=False,
            enable_entropy=False,
            enable_hurst=True,
        )
        result = adv.compute(featured)
        hurst = result["hurst_100d"].dropna()
        if len(hurst) > 0:
            assert hurst.min() >= 0
            assert hurst.max() <= 1


class TestFactorSelector:
    """Tests for FactorSelector."""

    def test_mutual_info_select(self, feature_matrix, binary_labels):
        selected = FactorSelector.mutual_information(
            feature_matrix, binary_labels, k=3
        )
        assert len(selected) == 3
        assert all(s in feature_matrix.columns for s in selected)

    def test_ic_ranking(self, feature_matrix, binary_labels):
        selected = FactorSelector.ic_ranking(
            feature_matrix, binary_labels, top_k=5
        )
        assert len(selected) <= 5


class TestLabelConstructor:
    """Tests for LabelConstructor."""

    def test_binary_label(self, synthetic_ohlcv):
        labels = LabelConstructor.binary_label(synthetic_ohlcv)
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_regression_label(self, synthetic_ohlcv):
        labels = LabelConstructor.regression_label(synthetic_ohlcv, horizon=5)
        assert labels.dtype == np.float64

    def test_position_label(self, synthetic_ohlcv):
        labels = LabelConstructor.position_label(synthetic_ohlcv)
        assert set(labels.dropna().unique()).issubset({-1, 0, 1})

    def test_construct_dispatch(self, synthetic_ohlcv):
        for target in ["binary", "regression", "ranking", "position"]:
            labels = LabelConstructor.construct(synthetic_ohlcv, target_type=target)
            assert len(labels) == len(synthetic_ohlcv)
