"""Tests for evaluation metrics and statistical tests."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.statistical_tests import StatisticalTests


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_classification_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.7, 0.1])

        metrics = MetricsCalculator.classification_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.5])

        metrics = MetricsCalculator.regression_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_ranking_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.2, 1.8, 3.1, 4.2, 4.9])

        metrics = MetricsCalculator.ranking_metrics(y_true, y_pred)

        assert "ic" in metrics
        assert "rank_ic" in metrics
        assert -1 <= metrics["ic"] <= 1
        assert metrics["rank_ic"] > 0  # Should be positively correlated

    def test_trading_metrics(self, returns_series):
        metrics = MetricsCalculator.trading_metrics(returns_series)

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "cagr" in metrics
        assert "win_rate" in metrics
        assert metrics["max_drawdown"] <= 0

    def test_full_report(self, returns_series):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            report = MetricsCalculator.full_report(
                y_true, y_pred, y_prob,
                returns=returns_series,
                output_path=path,
            )
            assert path.exists()
            assert "classification" in report
            assert "trading" in report


class TestStatisticalTests:
    """Tests for StatisticalTests."""

    def test_sharpe_bootstrap(self, returns_series):
        result = StatisticalTests.sharpe_bootstrap_ci(returns_series, n_bootstrap=1000)

        assert "sharpe" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["sharpe"] <= result["ci_upper"]

    def test_mean_return_ttest(self, returns_series):
        result = StatisticalTests.mean_return_ttest(returns_series)

        assert "t_stat" in result
        assert "p_value" in result
        assert "significant_5pct" in result
        assert 0 <= result["p_value"] <= 1

    def test_paired_ttest(self, returns_series):
        np.random.seed(123)
        other_returns = returns_series + np.random.normal(0, 0.001, len(returns_series))

        result = StatisticalTests.paired_model_ttest(returns_series, other_returns)

        assert "t_stat" in result
        assert "p_value" in result
        assert "mean_diff" in result
