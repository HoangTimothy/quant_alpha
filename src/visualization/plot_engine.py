"""
PlotEngine — publication-quality visualizations for quant research.

Generates: equity curve, drawdown chart, factor heatmap, feature importance,
returns distribution, prediction vs actual.

Uses a dark theme with vibrant accent colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── Theme ─────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
DARK_CARD = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_CYAN = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_ORANGE = "#d29922"
ACCENT_PINK = "#f778ba"

PALETTE = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_PINK, ACCENT_RED]


def _apply_dark_theme() -> None:
    """Set matplotlib rcParams for a premium dark theme."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_CARD,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "legend.facecolor": DARK_CARD,
        "legend.edgecolor": GRID_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
    })


class PlotEngine:
    """Generate all required research visualizations."""

    def __init__(self, output_dir: str | Path = "outputs/plots") -> None:
        self.output_dir = ensure_dir(Path(output_dir))
        _apply_dark_theme()

    def equity_curve(
        self,
        portfolio_value: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Equity Curve",
    ) -> Path:
        """Plot portfolio value over time with optional benchmark."""
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(portfolio_value.index, portfolio_value.values, color=ACCENT_CYAN, linewidth=1.8, label="Strategy")
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, color=ACCENT_ORANGE, linewidth=1.2, alpha=0.7, label="Benchmark", linestyle="--")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # Fill area under curve
        ax.fill_between(portfolio_value.index, portfolio_value.values, alpha=0.08, color=ACCENT_CYAN)

        path = self.output_dir / "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def drawdown_chart(
        self,
        portfolio_value: pd.Series,
        title: str = "Drawdown",
    ) -> Path:
        """Plot underwater / drawdown chart."""
        cum = portfolio_value / portfolio_value.iloc[0]
        peak = cum.cummax()
        drawdown = (cum - peak) / peak * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(drawdown.index, drawdown.values, 0, color=ACCENT_RED, alpha=0.4)
        ax.plot(drawdown.index, drawdown.values, color=ACCENT_RED, linewidth=1.2)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(True, alpha=0.3)

        # Annotate max drawdown
        min_dd_idx = drawdown.idxmin()
        min_dd_val = drawdown.min()
        ax.annotate(
            f"Max DD: {min_dd_val:.1f}%",
            xy=(min_dd_idx, min_dd_val),
            xytext=(min_dd_idx, min_dd_val + 2),
            fontsize=10,
            color=ACCENT_RED,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT_RED),
        )

        path = self.output_dir / "drawdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def factor_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Factor Correlation Heatmap",
    ) -> Path:
        """Plot factor-to-factor correlation heatmap."""
        # Limit to 30 factors for readability
        if correlation_matrix.shape[0] > 30:
            correlation_matrix = correlation_matrix.iloc[:30, :30]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            correlation_matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            linecolor=GRID_COLOR,
            cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.tick_params(axis="both", labelsize=8)
        plt.xticks(rotation=45, ha="right")

        path = self.output_dir / "factor_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def feature_importance(
        self,
        importances: dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> Path:
        """Horizontal bar chart of feature importances."""
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [x[0] for x in sorted_imp][::-1]
        values = [x[1] for x in sorted_imp][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [ACCENT_CYAN if v > np.mean(values) else ACCENT_PURPLE for v in values]
        ax.barh(names, values, color=colors, edgecolor=DARK_BG, linewidth=0.5, height=0.7)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Importance Score")
        ax.grid(True, axis="x", alpha=0.3)

        path = self.output_dir / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution",
    ) -> Path:
        """Histogram + KDE of daily returns."""
        fig, ax = plt.subplots(figsize=(12, 6))

        returns_clean = returns.dropna()
        ax.hist(returns_clean, bins=80, color=ACCENT_CYAN, alpha=0.5, density=True, edgecolor=DARK_BG)

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(returns_clean)
        x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
        ax.plot(x_range, kde(x_range), color=ACCENT_PINK, linewidth=2, label="KDE")

        # Stats annotation
        mean_r = returns_clean.mean()
        std_r = returns_clean.std()
        skew = returns_clean.skew()
        kurt = returns_clean.kurtosis()

        stats_text = f"Mean: {mean_r:.4f}\nStd: {std_r:.4f}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
        ax.text(
            0.97, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=DARK_CARD, edgecolor=GRID_COLOR, alpha=0.9),
        )

        ax.axvline(0, color=ACCENT_ORANGE, linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "returns_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Prediction vs Actual",
    ) -> Path:
        """Scatter plot of predicted vs actual values."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=ACCENT_CYAN, edgecolors="none")

        # Perfect prediction line
        lims = [
            min(np.min(y_true), np.min(y_pred)),
            max(np.max(y_true), np.max(y_pred)),
        ]
        ax.plot(lims, lims, color=ACCENT_ORANGE, linestyle="--", linewidth=1.5, alpha=0.8, label="Perfect prediction")

        # Fit line
        if len(y_true) > 5:
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            x_line = np.linspace(lims[0], lims[1], 100)
            ax.plot(x_line, p(x_line), color=ACCENT_GREEN, linewidth=1.5, alpha=0.7, label=f"Fit (slope={z[0]:.3f})")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "prediction_vs_actual.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info("Saved → %s", path)
        return path

    def generate_all(
        self,
        portfolio_value: pd.Series | None = None,
        returns: pd.Series | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        feature_importances: dict[str, float] | None = None,
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
    ) -> list[Path]:
        """Generate all available plots."""
        paths = []

        if portfolio_value is not None:
            paths.append(self.equity_curve(portfolio_value))
            paths.append(self.drawdown_chart(portfolio_value))

        if correlation_matrix is not None:
            paths.append(self.factor_heatmap(correlation_matrix))

        if feature_importances:
            paths.append(self.feature_importance(feature_importances))

        if returns is not None:
            paths.append(self.returns_distribution(returns))

        if y_true is not None and y_pred is not None:
            paths.append(self.prediction_vs_actual(y_true, y_pred))

        logger.info("Generated %d plots → %s", len(paths), self.output_dir)
        return paths
