"""
Label construction: binary, regression, ranking, and position targets.

All label constructors operate on a MultiIndex (date, ticker) DataFrame
and create forward-looking targets without introducing look-ahead bias
(the shift is applied within each ticker group).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class LabelConstructor:
    """Construct prediction targets from OHLCV data."""

    @staticmethod
    def binary_label(
        df: pd.DataFrame,
        threshold: float = 0.0,
        col: str = "adj_close",
    ) -> pd.Series:
        """Binary label: 1 if next-day return > threshold, else 0.

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex (date, ticker).
        threshold : float
            Return threshold for positive class.
        col : str
            Price column to compute returns from.

        Returns
        -------
        pd.Series
            Named 'label_binary', aligned with df index.
        """
        fwd_return = df.groupby(level="ticker")[col].pct_change().shift(-1)
        label = (fwd_return > threshold).astype(int)
        label.name = "label_binary"
        logger.info(
            "Binary label: threshold=%.4f, class_1 ratio=%.2f%%",
            threshold,
            label.mean() * 100,
        )
        return label

    @staticmethod
    def regression_label(
        df: pd.DataFrame,
        horizon: int = 5,
        col: str = "adj_close",
    ) -> pd.Series:
        """Regression label: future N-day return.

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex (date, ticker).
        horizon : int
            Number of days for forward return computation.
        col : str
            Price column.

        Returns
        -------
        pd.Series
            Named 'label_regression_{horizon}d'.
        """
        fwd_return = (
            df.groupby(level="ticker")[col]
            .pct_change(horizon)
            .groupby(level="ticker")
            .shift(-horizon)
        )
        fwd_return.name = f"label_regression_{horizon}d"
        logger.info(
            "Regression label: horizon=%dd, mean=%.4f, std=%.4f",
            horizon,
            fwd_return.mean(),
            fwd_return.std(),
        )
        return fwd_return

    @staticmethod
    def ranking_label(df: pd.DataFrame, col: str = "adj_close") -> pd.Series:
        """Ranking label: cross-sectional percentile rank of next-day return per date.

        Values in [0, 1], where 1 = highest return among peers that date.
        """
        fwd_return = df.groupby(level="ticker")[col].pct_change().shift(-1)
        ranked = fwd_return.groupby(level="date").rank(pct=True)
        ranked.name = "label_rank"
        logger.info("Ranking label: mean rank=%.4f", ranked.mean())
        return ranked

    @staticmethod
    def position_label(
        df: pd.DataFrame,
        long_threshold: float = 0.01,
        short_threshold: float = -0.01,
        col: str = "adj_close",
    ) -> pd.Series:
        """Position label: long (+1), short (-1), or neutral (0).

        Based on next-day return thresholds.
        """
        fwd_return = df.groupby(level="ticker")[col].pct_change().shift(-1)
        label = pd.Series(0, index=fwd_return.index, name="label_position")
        label[fwd_return > long_threshold] = 1
        label[fwd_return < short_threshold] = -1

        counts = label.value_counts()
        logger.info(
            "Position label: long=%d (%.1f%%), neutral=%d (%.1f%%), short=%d (%.1f%%)",
            counts.get(1, 0),
            counts.get(1, 0) / len(label) * 100,
            counts.get(0, 0),
            counts.get(0, 0) / len(label) * 100,
            counts.get(-1, 0),
            counts.get(-1, 0) / len(label) * 100,
        )
        return label

    @classmethod
    def construct(
        cls,
        df: pd.DataFrame,
        target_type: str = "binary",
        **kwargs,
    ) -> pd.Series:
        """Dispatch to the requested label constructor.

        Parameters
        ----------
        target_type : str
            One of: 'binary', 'regression', 'ranking', 'position'.
        """
        constructors = {
            "binary": cls.binary_label,
            "regression": cls.regression_label,
            "ranking": cls.ranking_label,
            "position": cls.position_label,
        }
        if target_type not in constructors:
            raise ValueError(f"Unknown target type: {target_type}. Choose from {list(constructors)}")

        return constructors[target_type](df, **kwargs)
