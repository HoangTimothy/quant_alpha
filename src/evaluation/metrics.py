"""
MetricsCalculator — comprehensive evaluation metrics for predictions and trading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

from src.utils import get_logger, save_json

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculate prediction, trading, and factor metrics."""

    # ── Classification Metrics ────────────────────────────────────────
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Accuracy, Precision, Recall, F1, ROC-AUC."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["roc_auc"] = 0.5
        return metrics

    # ── Regression Metrics ────────────────────────────────────────────
    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """RMSE, MAE, MAPE."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE — avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float("inf")

        return {"rmse": rmse, "mae": mae, "mape": mape}

    # ── Ranking Metrics ───────────────────────────────────────────────
    @staticmethod
    def ranking_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Information Coefficient (Pearson) and Rank IC (Spearman)."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 5:
            return {"ic": 0.0, "rank_ic": 0.0}

        ic, _ = pearsonr(y_true[mask], y_pred[mask])
        rank_ic, _ = spearmanr(y_true[mask], y_pred[mask])

        return {
            "ic": float(ic) if not np.isnan(ic) else 0.0,
            "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        }

    # ── Trading Metrics ───────────────────────────────────────────────
    @staticmethod
    def trading_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, float]:
        """Sharpe, Sortino, Calmar, MaxDD, CAGR, Vol, PF, WinRate, Turnover."""
        returns = returns.dropna()
        if len(returns) < 2:
            return {}

        # Annualised return
        total_days = len(returns)
        total_return = (1 + returns).prod() - 1
        years = total_days / 252
        cagr = (1 + total_return) ** (1 / max(years, 1e-6)) - 1 if total_return > -1 else -1
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino
        neg = returns[returns < 0]
        downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else 1e-6
        sortino = ann_return / downside_vol

        # Max Drawdown
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_dd = drawdown.min()

        # Calmar
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # Win Rate & Profit Factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / max(len(returns[returns != 0]), 1)
        profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float("inf")

        # Turnover (if we had position data, this would be actual; here approximate)
        turnover_approx = (returns != 0).sum() / total_days

        metrics = {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_dd),
            "cagr": float(cagr),
            "annual_return": float(ann_return),
            "annual_volatility": float(ann_vol),
            "total_return": float(total_return),
            "profit_factor": float(min(profit_factor, 999)),
            "win_rate": float(win_rate),
            "turnover": float(turnover_approx),
        }

        # Information Ratio (vs benchmark)
        if benchmark_returns is not None:
            excess = returns - benchmark_returns.reindex(returns.index).fillna(0)
            ir = excess.mean() * 252 / (excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
            metrics["information_ratio"] = float(ir)

        return metrics

    # ── Factor Metrics ────────────────────────────────────────────────
    @staticmethod
    def factor_metrics(
        feature_importances: dict[str, float] | None = None,
        ic_per_factor: pd.DataFrame | None = None,
    ) -> dict:
        """IC decay summary and feature importance ranking."""
        result = {}
        if feature_importances:
            sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            result["top_features"] = {k: float(v) for k, v in sorted_imp[:20]}

        if ic_per_factor is not None and not ic_per_factor.empty:
            result["top_ic_factors"] = ic_per_factor.head(10).to_dict()

        return result

    # ── Full Report ───────────────────────────────────────────────────
    @classmethod
    def full_report(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        returns: pd.Series | None = None,
        feature_importances: dict[str, float] | None = None,
        output_path: str | Path | None = None,
    ) -> dict:
        """Generate a comprehensive metrics report and optionally save to JSON.

        Parameters
        ----------
        y_true : array
            True labels.
        y_pred : array
            Predicted labels.
        y_prob : array, optional
            Predicted probabilities.
        returns : pd.Series, optional
            Strategy returns for trading metrics.
        feature_importances : dict, optional
            Feature importance scores.
        output_path : str | Path, optional
            Path to save metrics.json.

        Returns
        -------
        dict
            Complete metrics dictionary.
        """
        report = {}

        # Classification
        report["classification"] = cls.classification_metrics(y_true, y_pred, y_prob)

        # Regression (if predictions are continuous)
        if y_prob is not None:
            report["regression"] = cls.regression_metrics(y_true.astype(float), y_prob)

        # Ranking
        if y_prob is not None:
            report["ranking"] = cls.ranking_metrics(y_true.astype(float), y_prob)

        # Trading
        if returns is not None:
            report["trading"] = cls.trading_metrics(returns)

        # Factor
        if feature_importances:
            report["factor"] = cls.factor_metrics(feature_importances)

        if output_path:
            save_json(report, output_path)
            logger.info("Metrics saved → %s", output_path)

        return report
