"""
Main training entry point — Hydra-driven end-to-end pipeline.

Usage:
    python scripts/train.py                          # defaults
    python scripts/train.py model=lightgbm           # override model
    python scripts/train.py training.quick_mode=true  # fast smoke test
    python scripts/train.py --multirun model=xgboost,lightgbm  # sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ── Ensure project root is on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import MarketDataset
from src.features.factor_engine import FactorEngine
from src.features.advanced_factors import AdvancedFactors
from src.features.factor_selection import FactorSelector
from src.features.label_constructor import LabelConstructor
from src.features.factor_report import FactorReport
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.statistical_tests import StatisticalTests
from src.visualization.plot_engine import PlotEngine
from src.utils import set_seed, ensure_dir, save_json, get_logger, timer

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the full quantitative research pipeline."""

    logger.info("=" * 70)
    logger.info("Alpha Factor Discovery & Quantitative Backtesting Pipeline")
    logger.info("=" * 70)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Seed ──────────────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── Output dirs ───────────────────────────────────────────────────
    output_dir = ensure_dir(cfg.output.dir)
    plots_dir = ensure_dir(cfg.output.plots_dir)
    ckpt_dir = ensure_dir(cfg.output.checkpoint_dir)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ══════════════════════════════════════════════════════════════════
    with timer("Data Loading", logger):
        dataset = MarketDataset(
            tickers=list(cfg.data.tickers),
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )

        if cfg.data.source == "yfinance":
            dataset.load_yfinance()
        else:
            dataset.load_csv(cfg.data.csv_path)

        dataset.clean_missing(threshold=cfg.data.missing_threshold)

        if cfg.data.resample_freq:
            dataset.resample(cfg.data.resample_freq)

        train_data, val_data, test_data = dataset.split_time_series(
            train_end=cfg.data.train_end,
            val_end=cfg.data.val_end,
        )

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Feature Engineering
    # ══════════════════════════════════════════════════════════════════
    with timer("Feature Engineering", logger):
        # Basic factors
        factor_engine = FactorEngine(
            enable_returns=cfg.features.basic.returns,
            enable_trend=cfg.features.basic.trend,
            enable_momentum=cfg.features.basic.momentum,
            enable_volatility=cfg.features.basic.volatility,
            enable_volume=cfg.features.basic.volume,
            enable_price=cfg.features.basic.price,
            enable_time=cfg.features.basic.time,
            enable_lag=cfg.features.basic.lag,
            enable_cross_sectional=cfg.features.basic.cross_sectional,
        )

        # Compute on full dataset before splitting (to avoid NaN at boundaries)
        full_data = pd.concat([train_data, val_data, test_data]).sort_index()
        full_featured = factor_engine.compute(full_data)

        # Advanced factors
        adv_factors = AdvancedFactors(
            enable_realized_vol=cfg.features.advanced.realized_vol,
            enable_rolling_beta=cfg.features.advanced.rolling_beta,
            enable_downside_deviation=cfg.features.advanced.downside_deviation,
            enable_entropy=cfg.features.advanced.entropy,
            enable_hurst=cfg.features.advanced.hurst,
        )
        full_featured = adv_factors.compute(full_featured)

        # Re-split after feature engineering
        dates = full_featured.index.get_level_values("date")
        train_end = pd.Timestamp(cfg.data.train_end)
        val_end = pd.Timestamp(cfg.data.val_end)

        train_feat = full_featured.loc[dates <= train_end].copy()
        val_feat = full_featured.loc[(dates > train_end) & (dates <= val_end)].copy()
        test_feat = full_featured.loc[dates > val_end].copy()

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Label Construction
    # ══════════════════════════════════════════════════════════════════
    with timer("Label Construction", logger):
        label_kwargs = {}
        if cfg.labels.target_type == "binary":
            label_kwargs["threshold"] = cfg.labels.binary_threshold
        elif cfg.labels.target_type == "regression":
            label_kwargs["horizon"] = cfg.labels.regression_horizon
        elif cfg.labels.target_type == "position":
            label_kwargs["long_threshold"] = cfg.labels.long_threshold
            label_kwargs["short_threshold"] = cfg.labels.short_threshold

        train_labels = LabelConstructor.construct(train_feat, cfg.labels.target_type, **label_kwargs)
        val_labels = LabelConstructor.construct(val_feat, cfg.labels.target_type, **label_kwargs)
        test_labels = LabelConstructor.construct(test_feat, cfg.labels.target_type, **label_kwargs)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Prepare Feature Matrix
    # ══════════════════════════════════════════════════════════════════
    with timer("Feature Preparation", logger):
        # Identify feature columns (exclude raw OHLCV and labels)
        exclude_cols = {"open", "high", "low", "close", "volume", "adj_close"}
        feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

        X_train = train_feat[feature_cols].copy()
        X_val = val_feat[feature_cols].copy()
        X_test = test_feat[feature_cols].copy()

        y_train = train_labels.copy()
        y_val = val_labels.copy()
        y_test = test_labels.copy()

        # Align indices
        common_train = X_train.index.intersection(y_train.index)
        common_val = X_val.index.intersection(y_val.index)
        common_test = X_test.index.intersection(y_test.index)

        X_train, y_train = X_train.loc[common_train], y_train.loc[common_train]
        X_val, y_val = X_val.loc[common_val], y_val.loc[common_val]
        X_test, y_test = X_test.loc[common_test], y_test.loc[common_test]

        # Drop rows with NaN
        valid_train = X_train.notna().all(axis=1) & y_train.notna()
        valid_val = X_val.notna().all(axis=1) & y_val.notna()
        valid_test = X_test.notna().all(axis=1) & y_test.notna()

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_val, y_val = X_val[valid_val], y_val[valid_val]
        X_test, y_test = X_test[valid_test], y_test[valid_test]

        logger.info(
            "Feature matrix — train: %s, val: %s, test: %s, features: %d",
            X_train.shape, X_val.shape, X_test.shape, len(feature_cols),
        )

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Feature Selection
    # ══════════════════════════════════════════════════════════════════
    with timer("Feature Selection", logger):
        selection_method = cfg.features.selection.method
        top_k = cfg.features.selection.top_k

        try:
            selected_features = FactorSelector.select(
                method=selection_method,
                X=X_train,
                y=y_train,
                top_k=top_k,
                task="classification" if cfg.labels.target_type == "binary" else "regression",
            )

            if selection_method != "pca":
                X_train = X_train[selected_features]
                X_val = X_val[selected_features]
                X_test = X_test[selected_features]
                logger.info("Selected %d features via %s", len(selected_features), selection_method)
            else:
                logger.info("PCA: keeping full feature set (transformed during model training)")

        except Exception as e:
            logger.warning("Feature selection failed: %s — using all features.", e)
            selected_features = feature_cols

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Factor Report
    # ══════════════════════════════════════════════════════════════════
    with timer("Factor Report", logger):
        try:
            factor_report = FactorReport(output_dir=output_dir)
            report_data = factor_report.generate(
                train_feat, y_train,
                factor_cols=[c for c in selected_features if c in train_feat.columns],
            )
            save_json(
                {k: v.to_dict() if isinstance(v, pd.DataFrame) else str(v) for k, v in report_data.items()},
                output_dir / "factor_report.json",
            )
        except Exception as e:
            logger.warning("Factor report generation failed: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Train Model
    # ══════════════════════════════════════════════════════════════════
    with timer("Model Training", logger):
        model_name = cfg.model.name
        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)

        if cfg.training.quick_mode:
            # Quick mode: single train/val split, no walk-forward
            logger.info("QUICK MODE — single split training")
            model = create_model(model_name, params=model_params)
            model.fit(X_train, y_train, X_val, y_val)
            cv_results = None
        else:
            # Full walk-forward
            trainer = Trainer(
                model_name=model_name,
                model_params=model_params,
                use_wandb=cfg.experiment.use_wandb,
                use_mlflow=cfg.experiment.use_mlflow,
                experiment_name=cfg.experiment.name,
            )

            validation_method = cfg.training.validation_method
            X_combined = pd.concat([X_train, X_val])
            y_combined = pd.concat([y_train, y_val])

            if validation_method == "walk_forward":
                cv_results = trainer.walk_forward_validation(
                    X_combined, y_combined,
                    n_splits=cfg.training.n_splits,
                )
            elif validation_method == "expanding":
                cv_results = trainer.expanding_window(
                    X_combined, y_combined,
                    initial_train_size=cfg.training.expanding_initial_size,
                    step_size=cfg.training.step_size,
                )
            else:  # rolling
                cv_results = trainer.rolling_window(
                    X_combined, y_combined,
                    window_size=cfg.training.rolling_window_size,
                    step_size=cfg.training.step_size,
                )

            trainer.finish_tracking()

            # Train final model on full train+val data
            model = create_model(model_name, params=model_params)
            model.fit(X_train, y_train, X_val, y_val)

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Predictions
    # ══════════════════════════════════════════════════════════════════
    with timer("Prediction", logger):
        test_pred_proba = model.predict_proba(X_test)
        test_pred_labels = (test_pred_proba >= 0.5).astype(int)

        # Handle length mismatch from sequence models
        if len(test_pred_proba) != len(y_test):
            min_len = min(len(test_pred_proba), len(y_test))
            test_pred_proba = test_pred_proba[-min_len:]
            test_pred_labels = test_pred_labels[-min_len:]
            y_test_eval = y_test.iloc[-min_len:]
            X_test_eval = X_test.iloc[-min_len:]
        else:
            y_test_eval = y_test
            X_test_eval = X_test

        # Save model checkpoint
        model.save(ckpt_dir / f"{model_name}_final.pkl")

    # ══════════════════════════════════════════════════════════════════
    # STEP 9: Backtesting
    # ══════════════════════════════════════════════════════════════════
    with timer("Backtesting", logger):
        # Get test period prices
        test_prices = test_feat.loc[X_test_eval.index, "adj_close"] if "adj_close" in test_feat.columns else None

        if test_prices is not None:
            signals = pd.Series(test_pred_proba, index=X_test_eval.index)

            backtest_engine = BacktestEngine(
                commission_bps=cfg.backtest.commission_bps,
                slippage_bps=cfg.backtest.slippage_bps,
                stop_loss=cfg.backtest.stop_loss,
                take_profit=cfg.backtest.take_profit,
                max_drawdown=cfg.backtest.max_drawdown,
            )

            bt_result = backtest_engine.run(
                signals=signals,
                prices=test_prices,
                mode=cfg.backtest.mode,
                position_sizing=cfg.backtest.position_sizing,
            )

            bt_stats = bt_result["stats"]
            bt_returns = bt_result["returns"]
            portfolio_value = bt_result["portfolio_value"]
        else:
            logger.warning("No price data for backtesting — skipping.")
            bt_stats = {}
            bt_returns = pd.Series(dtype=float)
            portfolio_value = pd.Series(dtype=float)

    # ══════════════════════════════════════════════════════════════════
    # STEP 10: Evaluation
    # ══════════════════════════════════════════════════════════════════
    with timer("Evaluation", logger):
        feature_importances = model.get_feature_importance()

        metrics = MetricsCalculator.full_report(
            y_true=y_test_eval.values,
            y_pred=test_pred_labels,
            y_prob=test_pred_proba,
            returns=bt_returns if len(bt_returns) > 0 else None,
            feature_importances=feature_importances,
            output_path=cfg.output.metrics_file,
        )

        # Statistical tests
        if len(bt_returns) > 0:
            stat_tests = StatisticalTests.full_statistical_report(bt_returns)
            metrics["statistical"] = stat_tests
            save_json(metrics, cfg.output.metrics_file)

        # Log summary
        logger.info("=" * 50)
        logger.info("RESULTS SUMMARY — Model: %s", model_name)
        logger.info("=" * 50)

        if "classification" in metrics:
            clf = metrics["classification"]
            logger.info(
                "Classification — Acc=%.4f, F1=%.4f, AUC=%.4f",
                clf.get("accuracy", 0), clf.get("f1", 0), clf.get("roc_auc", 0),
            )

        if "trading" in metrics:
            trd = metrics["trading"]
            logger.info(
                "Trading — Sharpe=%.3f, Sortino=%.3f, MaxDD=%.2f%%, CAGR=%.2f%%",
                trd.get("sharpe_ratio", 0),
                trd.get("sortino_ratio", 0),
                trd.get("max_drawdown", 0) * 100,
                trd.get("cagr", 0) * 100,
            )

        if cv_results:
            logger.info(
                "Walk-Forward CV — Mean AUC=%.4f, Mean Acc=%.4f",
                cv_results["mean_auc"], cv_results["mean_accuracy"],
            )

    # ══════════════════════════════════════════════════════════════════
    # STEP 11: Visualization
    # ══════════════════════════════════════════════════════════════════
    with timer("Visualization", logger):
        plotter = PlotEngine(output_dir=plots_dir)

        plotter.generate_all(
            portfolio_value=portfolio_value if len(portfolio_value) > 0 else None,
            returns=bt_returns if len(bt_returns) > 0 else None,
            correlation_matrix=report_data.get("correlation_matrix") if "report_data" in dir() else None,
            feature_importances=feature_importances,
            y_true=y_test_eval.values.astype(float),
            y_pred=test_pred_proba,
        )

    logger.info("=" * 70)
    logger.info("Pipeline complete! Outputs → %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
