"""
Factor analysis report — IC analysis, correlation heatmap, quintile returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class FactorReport:
    """Generate a factor analysis report with IC, correlations, and quintile analysis."""

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = ensure_dir(Path(output_dir))

    def generate(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        factor_cols: list[str],
    ) -> dict:
        """Run full factor analysis and return results dict.

        Parameters
        ----------
        features_df : pd.DataFrame
            MultiIndex (date, ticker) with all computed factors.
        target : pd.Series
            Future return or target variable, aligned with features_df.
        factor_cols : list[str]
            List of factor column names to analyse.

        Returns
        -------
        dict
            Contains 'ic_per_factor', 'ic_decay', 'correlation_matrix', 'quintile_returns'.
        """
        report = {}

        # 1. IC per factor (cross-sectional Spearman per date, averaged)
        report["ic_per_factor"] = self._ic_per_factor(features_df, target, factor_cols)

        # 2. IC decay over horizons
        if "adj_close" in features_df.columns:
            report["ic_decay"] = self._ic_decay(features_df, factor_cols)

        # 3. Factor correlation matrix
        report["correlation_matrix"] = self._correlation_matrix(features_df, factor_cols)

        # 4. Quintile returns
        report["quintile_returns"] = self._quintile_returns(features_df, target, factor_cols[:10])

        logger.info("Factor report generated with %d factors analysed.", len(factor_cols))
        return report

    @staticmethod
    def _ic_per_factor(
        df: pd.DataFrame,
        target: pd.Series,
        factor_cols: list[str],
    ) -> pd.DataFrame:
        """Cross-sectional IC (Spearman) per date, then mean/std/IR."""
        results = {}
        dates = df.index.get_level_values("date").unique()

        for col in factor_cols:
            daily_ics = []
            for dt in dates:
                try:
                    mask = df.index.get_level_values("date") == dt
                    x = df.loc[mask, col].values
                    y = target.loc[mask].values
                    valid = ~(np.isnan(x) | np.isnan(y))
                    if valid.sum() >= 3:
                        ic, _ = spearmanr(x[valid], y[valid])
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                except Exception:
                    continue

            if daily_ics:
                mean_ic = np.mean(daily_ics)
                std_ic = np.std(daily_ics)
                ir = mean_ic / std_ic if std_ic > 0 else 0.0
                results[col] = {
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "ir": ir,
                    "n_obs": len(daily_ics),
                }

        return pd.DataFrame(results).T.sort_values("mean_ic", key=abs, ascending=False)

    @staticmethod
    def _ic_decay(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
        """IC decay: correlate factor values today with future returns at different horizons."""
        close = df.groupby(level="ticker")["adj_close"]
        horizons = [1, 2, 5, 10, 20]
        decay = {}

        for h in horizons:
            fwd_ret = close.pct_change(h).shift(-h)
            ics = {}
            for col in factor_cols[:20]:  # top 20 only for speed
                mask = df[col].notna() & fwd_ret.notna()
                if mask.sum() < 30:
                    continue
                ic, _ = spearmanr(df.loc[mask, col], fwd_ret.loc[mask])
                ics[col] = ic if not np.isnan(ic) else 0.0
            decay[f"horizon_{h}d"] = ics

        return pd.DataFrame(decay)

    @staticmethod
    def _correlation_matrix(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
        """Pairwise Spearman correlation among factors."""
        subset = df[factor_cols].dropna(how="all")
        return subset.corr(method="spearman")

    @staticmethod
    def _quintile_returns(
        df: pd.DataFrame,
        target: pd.Series,
        factor_cols: list[str],
    ) -> dict[str, pd.Series]:
        """For each factor, sort into quintiles and compute mean target return per quintile."""
        quintiles = {}
        for col in factor_cols:
            combined = pd.DataFrame({"factor": df[col], "target": target}).dropna()
            if len(combined) < 50:
                continue
            combined["quintile"] = pd.qcut(combined["factor"], 5, labels=False, duplicates="drop")
            quintiles[col] = combined.groupby("quintile")["target"].mean()
        return quintiles
