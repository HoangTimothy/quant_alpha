"""
Standalone backtesting script — run backtests from saved predictions or models.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py backtest.mode=long_only backtest.position_sizing=vol_targeting
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import MarketDataset
from src.features.factor_engine import FactorEngine
from src.features.advanced_factors import AdvancedFactors
from src.features.label_constructor import LabelConstructor
from src.models.model_factory import create_model
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.statistical_tests import StatisticalTests
from src.visualization.plot_engine import PlotEngine
from src.utils import set_seed, ensure_dir, save_json, get_logger, timer

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run backtesting with a saved or freshly trained model."""

    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output.dir)
    plots_dir = ensure_dir(cfg.output.plots_dir)
    ckpt_dir = Path(cfg.output.checkpoint_dir)

    # ── Load or train model ───────────────────────────────────────────
    model_path = ckpt_dir / f"{cfg.model.name}_final.pkl"

    with timer("Data Loading & Features", logger):
        dataset = MarketDataset(
            tickers=list(cfg.data.tickers),
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        dataset.load_yfinance()
        dataset.clean_missing()
        _, _, test_data = dataset.split_time_series(
            train_end=cfg.data.train_end, val_end=cfg.data.val_end,
        )

        factor_engine = FactorEngine()
        test_feat = factor_engine.compute(test_data)

        adv = AdvancedFactors()
        test_feat = adv.compute(test_feat)

    with timer("Label Construction", logger):
        label_kwargs = {}
        if cfg.labels.target_type == "binary":
            label_kwargs["threshold"] = cfg.labels.binary_threshold
        test_labels = LabelConstructor.construct(test_feat, cfg.labels.target_type, **label_kwargs)

    # Prepare features
    exclude_cols = {"open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in test_feat.columns if c not in exclude_cols]
    X_test = test_feat[feature_cols].copy()
    y_test = test_labels.copy()

    common = X_test.index.intersection(y_test.index)
    X_test, y_test = X_test.loc[common], y_test.loc[common]
    valid = X_test.notna().all(axis=1) & y_test.notna()
    X_test, y_test = X_test[valid], y_test[valid]

    # Load model
    with timer("Model Loading", logger):
        if model_path.exists():
            model = create_model(cfg.model.name)
            model.load(model_path)
            logger.info("Loaded saved model from %s", model_path)
        else:
            logger.warning("No saved model found — training fresh model.")
            # Quick train on available data
            from src.data_loader import MarketDataset as MD2
            ds2 = MarketDataset(tickers=list(cfg.data.tickers), start_date=cfg.data.start_date, end_date=cfg.data.end_date)
            ds2.load_yfinance()
            ds2.clean_missing()
            train_d, val_d, _ = ds2.split_time_series(train_end=cfg.data.train_end, val_end=cfg.data.val_end)

            full_train = pd.concat([train_d, val_d]).sort_index()
            full_train = factor_engine.compute(full_train)
            full_train = adv.compute(full_train)
            train_labels = LabelConstructor.construct(full_train, cfg.labels.target_type, **label_kwargs)

            X_tr = full_train[[c for c in feature_cols if c in full_train.columns]]
            y_tr = train_labels
            common_tr = X_tr.index.intersection(y_tr.index)
            X_tr, y_tr = X_tr.loc[common_tr], y_tr.loc[common_tr]
            valid_tr = X_tr.notna().all(axis=1) & y_tr.notna()
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]

            # Match feature sets
            common_feats = [c for c in X_test.columns if c in X_tr.columns]
            X_tr = X_tr[common_feats]
            X_test = X_test[common_feats]

            from omegaconf import OmegaConf
            model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
            model = create_model(cfg.model.name, params=model_params)
            model.fit(X_tr, y_tr)

    # ── Predict ───────────────────────────────────────────────────────
    with timer("Prediction & Backtest", logger):
        preds = model.predict_proba(X_test)
        pred_labels = (preds >= 0.5).astype(int)

        if len(preds) != len(y_test):
            min_len = min(len(preds), len(y_test))
            preds = preds[-min_len:]
            pred_labels = pred_labels[-min_len:]
            y_test = y_test.iloc[-min_len:]
            X_test = X_test.iloc[-min_len:]

        # Backtest
        test_prices = test_feat.loc[X_test.index, "adj_close"]
        signals = pd.Series(preds, index=X_test.index)

        engine = BacktestEngine(
            commission_bps=cfg.backtest.commission_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            stop_loss=cfg.backtest.stop_loss,
            take_profit=cfg.backtest.take_profit,
        )

        bt_result = engine.run(
            signals=signals,
            prices=test_prices,
            mode=cfg.backtest.mode,
            position_sizing=cfg.backtest.position_sizing,
        )

    # ── Report ────────────────────────────────────────────────────────
    with timer("Report Generation", logger):
        metrics = MetricsCalculator.full_report(
            y_true=y_test.values,
            y_pred=pred_labels,
            y_prob=preds,
            returns=bt_result["returns"],
            output_path=output_dir / "backtest_metrics.json",
        )

        stat_tests = StatisticalTests.full_statistical_report(bt_result["returns"])
        metrics["statistical"] = stat_tests
        save_json(metrics, output_dir / "backtest_metrics.json")

        plotter = PlotEngine(output_dir=plots_dir)
        plotter.equity_curve(bt_result["portfolio_value"])
        plotter.drawdown_chart(bt_result["portfolio_value"])
        plotter.returns_distribution(bt_result["returns"])

        logger.info("Backtest — Sharpe=%.3f, MaxDD=%.2f%%",
                     bt_result["stats"].get("sharpe_ratio", 0),
                     bt_result["stats"].get("max_drawdown", 0) * 100)
        logger.info("Results → %s", output_dir)


if __name__ == "__main__":
    main()
