"""
Ablation study — systematically disable pipeline components and compare.

Runs 4 experiments:
  1. Without technical indicators
  2. Without advanced factors
  3. Without hyperparameter optimization (default params)
  4. Without feature selection (all features)

Produces outputs/ablation_results.csv comparison table.

Usage:
    python scripts/ablation.py
    python scripts/ablation.py model=lightgbm
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import MarketDataset
from src.features.factor_engine import FactorEngine
from src.features.advanced_factors import AdvancedFactors
from src.features.factor_selection import FactorSelector
from src.features.label_constructor import LabelConstructor
from src.models.model_factory import create_model
from src.evaluation.metrics import MetricsCalculator
from src.backtesting.backtest_engine import BacktestEngine
from src.utils import set_seed, ensure_dir, get_logger, save_json, timer

logger = get_logger(__name__)


def run_experiment(
    cfg: DictConfig,
    experiment_name: str,
    enable_technical: bool = True,
    enable_advanced: bool = True,
    enable_selection: bool = True,
    use_default_params: bool = False,
) -> dict:
    """Run a single ablation experiment and return metrics."""
    logger.info("─" * 40)
    logger.info("ABLATION: %s", experiment_name)
    logger.info("─" * 40)

    set_seed(cfg.seed)

    # Load data
    dataset = MarketDataset(
        tickers=list(cfg.data.tickers),
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    dataset.load_yfinance()
    dataset.clean_missing()
    train_data, val_data, test_data = dataset.split_time_series(
        train_end=cfg.data.train_end, val_end=cfg.data.val_end,
    )

    # Features
    full_data = pd.concat([train_data, val_data, test_data]).sort_index()

    factor_engine = FactorEngine(
        enable_trend=enable_technical,
        enable_momentum=enable_technical,
        enable_volatility=enable_technical,
        enable_volume=enable_technical,
    )
    full_featured = factor_engine.compute(full_data)

    if enable_advanced:
        adv = AdvancedFactors()
        full_featured = adv.compute(full_featured)

    # Split
    dates = full_featured.index.get_level_values("date")
    train_end_ts = pd.Timestamp(cfg.data.train_end)
    val_end_ts = pd.Timestamp(cfg.data.val_end)

    train_feat = full_featured.loc[dates <= train_end_ts]
    val_feat = full_featured.loc[(dates > train_end_ts) & (dates <= val_end_ts)]
    test_feat = full_featured.loc[dates > val_end_ts]

    # Labels
    label_kwargs = {"threshold": cfg.labels.binary_threshold} if cfg.labels.target_type == "binary" else {}
    y_train = LabelConstructor.construct(train_feat, cfg.labels.target_type, **label_kwargs)
    y_val = LabelConstructor.construct(val_feat, cfg.labels.target_type, **label_kwargs)
    y_test = LabelConstructor.construct(test_feat, cfg.labels.target_type, **label_kwargs)

    # Prepare features
    exclude = {"open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in train_feat.columns if c not in exclude]

    X_train = train_feat[feature_cols]
    X_val = val_feat[feature_cols]
    X_test = test_feat[feature_cols]

    # Align and clean
    for X, y, name in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        pass

    common = X_train.index.intersection(y_train.index)
    X_train, y_train = X_train.loc[common], y_train.loc[common]
    common = X_val.index.intersection(y_val.index)
    X_val, y_val = X_val.loc[common], y_val.loc[common]
    common = X_test.index.intersection(y_test.index)
    X_test, y_test = X_test.loc[common], y_test.loc[common]

    valid = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train[valid], y_train[valid]
    valid = X_val.notna().all(axis=1) & y_val.notna()
    X_val, y_val = X_val[valid], y_val[valid]
    valid = X_test.notna().all(axis=1) & y_test.notna()
    X_test, y_test = X_test[valid], y_test[valid]

    # Feature selection
    if enable_selection and X_train.shape[1] > cfg.features.selection.top_k:
        try:
            selected = FactorSelector.select(
                method=cfg.features.selection.method,
                X=X_train, y=y_train,
                top_k=cfg.features.selection.top_k,
            )
            if cfg.features.selection.method != "pca":
                X_train = X_train[selected]
                X_val = X_val[selected]
                X_test = X_test[selected]
        except Exception:
            pass

    # Model
    if use_default_params:
        model = create_model(cfg.model.name, params={})
    else:
        model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
        model = create_model(cfg.model.name, params=model_params)

    model.fit(X_train, y_train, X_val, y_val)

    # Predict
    preds = model.predict_proba(X_test)
    pred_labels = (preds >= 0.5).astype(int)

    if len(preds) != len(y_test):
        min_len = min(len(preds), len(y_test))
        preds, pred_labels = preds[-min_len:], pred_labels[-min_len:]
        y_test = y_test.iloc[-min_len:]
        X_test = X_test.iloc[-min_len:]

    # Evaluate
    metrics = MetricsCalculator.classification_metrics(y_test.values, pred_labels, preds)

    # Backtest
    try:
        test_prices = test_feat.loc[X_test.index, "adj_close"]
        signals = pd.Series(preds, index=X_test.index)
        engine = BacktestEngine(commission_bps=cfg.backtest.commission_bps, slippage_bps=cfg.backtest.slippage_bps)
        bt = engine.run(signals, test_prices, mode=cfg.backtest.mode)
        trading = MetricsCalculator.trading_metrics(bt["returns"])
        metrics.update(trading)
    except Exception as e:
        logger.warning("Backtest failed in ablation '%s': %s", experiment_name, e)

    metrics["experiment"] = experiment_name
    metrics["n_features"] = X_test.shape[1]
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run all ablation experiments."""

    logger.info("=" * 70)
    logger.info("ABLATION STUDY")
    logger.info("=" * 70)

    output_dir = ensure_dir(cfg.output.dir)
    results = []

    # 0. Baseline (all features, full config)
    results.append(run_experiment(cfg, "Baseline (Full Pipeline)"))

    # 1. Without technical indicators
    results.append(run_experiment(
        cfg, "Without Technical Indicators",
        enable_technical=False,
    ))

    # 2. Without advanced factors
    results.append(run_experiment(
        cfg, "Without Advanced Factors",
        enable_advanced=False,
    ))

    # 3. Without optimization (default params)
    results.append(run_experiment(
        cfg, "Without Optimization",
        use_default_params=True,
    ))

    # 4. Without feature selection
    results.append(run_experiment(
        cfg, "Without Feature Selection",
        enable_selection=False,
    ))

    # Compile results
    df = pd.DataFrame(results)
    df = df.set_index("experiment")

    display_cols = [
        "accuracy", "f1", "roc_auc", "n_features",
        "sharpe_ratio", "max_drawdown", "cagr", "win_rate",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].round(4)

    csv_path = output_dir / "ablation_results.csv"
    df_display.to_csv(csv_path)

    logger.info("\n" + "=" * 70)
    logger.info("ABLATION COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info("\n%s", df_display.to_string())
    logger.info("\nResults saved → %s", csv_path)

    save_json(df.to_dict(orient="index"), output_dir / "ablation_results.json")


if __name__ == "__main__":
    main()
