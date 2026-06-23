"""
Hyperparameter search: grid search and Bayesian optimization (Optuna).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.base_model import BaseModel
from src.models.model_factory import create_model
from src.utils import get_logger

logger = get_logger(__name__)


def grid_search(
    model_name: str,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric_fn: Callable | None = None,
) -> tuple[dict, float, BaseModel]:
    """Exhaustive grid search over all hyperparameter combinations.

    Parameters
    ----------
    model_name : str
        Name of the model in the registry.
    param_grid : dict
        Mapping parameter name → list of values to try.
    metric_fn : callable, optional
        Scoring function(y_true, y_pred) → float (higher = better).
        Defaults to ROC-AUC.

    Returns
    -------
    (best_params, best_score, best_model)
    """
    if metric_fn is None:
        metric_fn = roc_auc_score

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    logger.info("Grid search: %d combinations for model=%s", len(combinations), model_name)

    best_score = -np.inf
    best_params: dict = {}
    best_model: BaseModel | None = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        try:
            model = create_model(model_name, params=params)
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict_proba(X_val)
            score = metric_fn(y_val, preds)

            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

            logger.info("  [%d/%d] params=%s → score=%.4f", i + 1, len(combinations), params, score)
        except Exception as e:
            logger.warning("  [%d/%d] Failed with %s: %s", i + 1, len(combinations), params, e)

    logger.info("Grid search best: score=%.4f, params=%s", best_score, best_params)
    return best_params, best_score, best_model


def bayesian_search(
    model_name: str,
    search_space: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    metric_fn: Callable | None = None,
) -> tuple[dict, float, BaseModel]:
    """Bayesian hyperparameter optimization using Optuna.

    Parameters
    ----------
    model_name : str
        Name of the model in the registry.
    search_space : dict
        Mapping parameter name → dict with keys:
          - 'type': 'int', 'float', 'categorical', 'loguniform'
          - 'low', 'high': range for int/float
          - 'choices': list for categorical
    n_trials : int
        Number of optimization trials.

    Returns
    -------
    (best_params, best_score, best_model)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed — falling back to grid search stub.")
        raise

    if metric_fn is None:
        metric_fn = roc_auc_score

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in search_space.items():
            stype = spec["type"]
            if stype == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif stype == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif stype == "loguniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif stype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search type: {stype}")

        model = create_model(model_name, params=params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict_proba(X_val)
        return metric_fn(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_score = study.best_value
    logger.info("Bayesian search best: score=%.4f, params=%s", best_score, best_params)

    # Re-train with best params
    best_model = create_model(model_name, params=best_params)
    best_model.fit(X_train, y_train, X_val, y_val)

    return best_params, best_score, best_model
