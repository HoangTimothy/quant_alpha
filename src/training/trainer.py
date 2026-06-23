"""
Trainer — walk-forward, expanding window, and rolling window validation.

Orchestrates model training with proper temporal splits to prevent look-ahead bias.
Supports experiment tracking via wandb and mlflow.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.base_model import BaseModel
from src.models.model_factory import create_model
from src.utils import get_logger, timer

logger = get_logger(__name__)


class Trainer:
    """Train models using temporal cross-validation strategies."""

    def __init__(
        self,
        model_name: str = "xgboost",
        model_params: dict | None = None,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        experiment_name: str = "quant_alpha",
    ) -> None:
        self.model_name = model_name
        self.model_params = model_params or {}
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        self._init_tracking()

    def _init_tracking(self) -> None:
        """Initialize experiment tracking backends."""
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=self.experiment_name, config=self.model_params, reinit=True)
                logger.info("wandb initialized: project=%s", self.experiment_name)
            except Exception as e:
                logger.warning("wandb init failed: %s — disabling.", e)
                self.use_wandb = False

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run(run_name=f"{self.model_name}_run")
                mlflow.log_params(self.model_params)
                logger.info("mlflow initialized: experiment=%s", self.experiment_name)
            except Exception as e:
                logger.warning("mlflow init failed: %s — disabling.", e)
                self.use_mlflow = False

    # ── Walk-Forward Validation ───────────────────────────────────────
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        initial_train_ratio: float = 0.5,
    ) -> dict[str, Any]:
        """Walk-forward validation with expanding training window.

        Split timeline into n_splits periods. For each fold:
          - Train on all data up to fold start
          - Test on the fold period

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must have DatetimeIndex or be sorted temporally).
        y : pd.Series
            Target vector, aligned with X.
        n_splits : int
            Number of forward-testing periods.
        initial_train_ratio : float
            Fraction of data used as initial training set.

        Returns
        -------
        dict
            Contains 'fold_results', 'mean_auc', 'mean_accuracy', 'all_predictions'.
        """
        n = len(X)
        initial_train_size = int(n * initial_train_ratio)
        remaining = n - initial_train_size
        fold_size = remaining // n_splits

        logger.info(
            "Walk-forward: %d samples, initial_train=%d, %d folds of ~%d samples",
            n, initial_train_size, n_splits, fold_size,
        )

        fold_results = []
        all_preds = []
        all_actuals = []

        for fold in range(n_splits):
            train_end = initial_train_size + fold * fold_size
            test_start = train_end
            test_end = min(train_end + fold_size, n)

            if test_start >= n:
                break

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            if len(X_test) == 0:
                continue

            with timer(f"Fold {fold + 1}/{n_splits}", logger):
                model = create_model(self.model_name, params=self.model_params.copy())

                # Use last 20% of training as validation for early stopping
                val_split = int(len(X_train) * 0.8)
                X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
                y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

                model.fit(X_tr, y_tr, X_val, y_val)

                preds = model.predict_proba(X_test)
                pred_labels = (preds >= 0.5).astype(int)

                # Handle length mismatch from sequence models
                if len(preds) != len(y_test):
                    min_len = min(len(preds), len(y_test))
                    preds = preds[-min_len:]
                    pred_labels = pred_labels[-min_len:]
                    y_test_eval = y_test.iloc[-min_len:]
                else:
                    y_test_eval = y_test

                try:
                    auc = roc_auc_score(y_test_eval, preds)
                except ValueError:
                    auc = 0.5

                acc = accuracy_score(y_test_eval, pred_labels)

                fold_result = {
                    "fold": fold + 1,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "auc": auc,
                    "accuracy": acc,
                }
                fold_results.append(fold_result)
                all_preds.extend(preds.tolist())
                all_actuals.extend(y_test_eval.values.tolist())

                logger.info(
                    "  Fold %d — AUC=%.4f, Acc=%.4f (train=%d, test=%d)",
                    fold + 1, auc, acc, len(X_train), len(X_test),
                )

                self._log_fold_metrics(fold + 1, fold_result)

        mean_auc = np.mean([r["auc"] for r in fold_results])
        mean_acc = np.mean([r["accuracy"] for r in fold_results])

        logger.info("Walk-forward complete — Mean AUC=%.4f, Mean Acc=%.4f", mean_auc, mean_acc)

        return {
            "fold_results": fold_results,
            "mean_auc": mean_auc,
            "mean_accuracy": mean_acc,
            "all_predictions": np.array(all_preds),
            "all_actuals": np.array(all_actuals),
        }

    # ── Expanding Window ──────────────────────────────────────────────
    def expanding_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        initial_train_size: int = 504,
        step_size: int = 63,
    ) -> dict[str, Any]:
        """Expanding window cross-validation.

        Start with initial_train_size, then expand by step_size each iteration.
        """
        splits = list(self._expanding_splits(len(X), initial_train_size, step_size))
        return self._run_splits(X, y, splits, "expanding")

    # ── Rolling Window ────────────────────────────────────────────────
    def rolling_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int = 252,
        step_size: int = 63,
    ) -> dict[str, Any]:
        """Rolling (fixed-width) window cross-validation."""
        splits = list(self._rolling_splits(len(X), window_size, step_size))
        return self._run_splits(X, y, splits, "rolling")

    # ── Train single model ────────────────────────────────────────────
    def train_single(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> BaseModel:
        """Train a single model on the provided data (no CV)."""
        model = create_model(self.model_name, params=self.model_params.copy())
        model.fit(X_train, y_train, X_val, y_val)
        return model

    # ── Internal helpers ──────────────────────────────────────────────
    def _run_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[range, range]],
        method_name: str,
    ) -> dict[str, Any]:
        """Execute training across a list of (train_idx, test_idx) splits."""
        fold_results = []
        all_preds = []
        all_actuals = []

        for i, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            model = create_model(self.model_name, params=self.model_params.copy())

            val_split = int(len(X_train) * 0.8)
            model.fit(
                X_train.iloc[:val_split], y_train.iloc[:val_split],
                X_train.iloc[val_split:], y_train.iloc[val_split:],
            )

            preds = model.predict_proba(X_test)
            pred_labels = (preds >= 0.5).astype(int)

            if len(preds) != len(y_test):
                min_len = min(len(preds), len(y_test))
                preds = preds[-min_len:]
                pred_labels = pred_labels[-min_len:]
                y_test = y_test.iloc[-min_len:]

            try:
                auc = roc_auc_score(y_test, preds)
            except ValueError:
                auc = 0.5

            acc = accuracy_score(y_test, pred_labels)
            fold_results.append({"fold": i + 1, "auc": auc, "accuracy": acc})
            all_preds.extend(preds.tolist())
            all_actuals.extend(y_test.values.tolist())

            logger.info("  %s fold %d — AUC=%.4f, Acc=%.4f", method_name, i + 1, auc, acc)

        return {
            "fold_results": fold_results,
            "mean_auc": np.mean([r["auc"] for r in fold_results]),
            "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "all_predictions": np.array(all_preds),
            "all_actuals": np.array(all_actuals),
        }

    @staticmethod
    def _expanding_splits(n: int, initial: int, step: int):
        idx = initial
        while idx + step <= n:
            yield range(0, idx), range(idx, min(idx + step, n))
            idx += step

    @staticmethod
    def _rolling_splits(n: int, window: int, step: int):
        idx = window
        while idx + step <= n:
            yield range(idx - window, idx), range(idx, min(idx + step, n))
            idx += step

    def _log_fold_metrics(self, fold: int, metrics: dict) -> None:
        """Log to experiment trackers."""
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f"fold_{fold}/{k}": v for k, v in metrics.items()})
            except Exception:
                pass

        if self.use_mlflow:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"fold_{fold}_{k}", v)
            except Exception:
                pass

    def finish_tracking(self) -> None:
        """Finalize experiment tracking runs."""
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if self.use_mlflow:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
