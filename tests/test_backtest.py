"""Tests for backtesting engine."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting.backtest_engine import BacktestEngine


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def price_and_signals(self):
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=252)
        prices = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates))),
            index=dates,
        )
        # Simple momentum signal
        signals = pd.Series(
            np.random.uniform(0, 1, len(dates)),
            index=dates,
        )
        return prices, signals

    def test_run_long_only(self, price_and_signals):
        prices, signals = price_and_signals
        engine = BacktestEngine(commission_bps=10, slippage_bps=5)
        result = engine.run(signals, prices, mode="long_only")

        assert "portfolio_value" in result
        assert "returns" in result
        assert "stats" in result
        assert result["portfolio_value"].iloc[0] > 0

    def test_run_long_short(self, price_and_signals):
        prices, signals = price_and_signals
        engine = BacktestEngine(commission_bps=10, slippage_bps=5)
        result = engine.run(signals, prices, mode="long_short")
        assert len(result["returns"]) > 0

    def test_transaction_costs_reduce_returns(self, price_and_signals):
        prices, signals = price_and_signals

        engine_no_cost = BacktestEngine(commission_bps=0, slippage_bps=0)
        result_no_cost = engine_no_cost.run(signals, prices, mode="long_only")

        engine_with_cost = BacktestEngine(commission_bps=50, slippage_bps=50)
        result_with_cost = engine_with_cost.run(signals, prices, mode="long_only")

        # Portfolio with costs should have lower final value
        assert result_with_cost["stats"]["total_return"] <= result_no_cost["stats"]["total_return"]

    def test_stats_keys(self, price_and_signals):
        prices, signals = price_and_signals
        engine = BacktestEngine()
        result = engine.run(signals, prices)
        stats = result["stats"]

        expected_keys = ["total_return", "sharpe_ratio", "max_drawdown"]
        for key in expected_keys:
            assert key in stats, f"Missing stat: {key}"

    def test_vol_targeting(self, price_and_signals):
        prices, signals = price_and_signals
        engine = BacktestEngine()
        result = engine.run(
            signals, prices,
            position_sizing="vol_targeting",
            target_volatility=0.10,
        )
        assert len(result["returns"]) > 0
