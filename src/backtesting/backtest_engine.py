"""
BacktestEngine — realistic backtesting using vectorbt with transaction costs,
slippage, position sizing, and risk controls.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    logger.warning("vectorbt not installed — backtesting will use fallback engine.")


class BacktestEngine:
    """Run realistic backtests with configurable costs, sizing, and risk."""

    def __init__(
        self,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
        stop_loss: float | None = 0.05,
        take_profit: float | None = 0.10,
        max_drawdown: float | None = 0.20,
        max_turnover: float | None = 2.0,
    ) -> None:
        self.commission = commission_bps / 10_000
        self.slippage = slippage_bps / 10_000
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_turnover = max_turnover

    def run(
        self,
        signals: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        mode: Literal["long_only", "long_short"] = "long_short",
        position_sizing: Literal["equal_weight", "vol_targeting", "kelly"] = "equal_weight",
        target_volatility: float = 0.15,
        initial_capital: float = 1_000_000,
    ) -> dict:
        """Execute a backtest.

        Parameters
        ----------
        signals : pd.Series or pd.DataFrame
            Signal values: >0 for long, <0 for short, 0 for flat.
            If probabilities, they should be centered around 0.5.
        prices : pd.Series or pd.DataFrame
            Price data aligned with signals.
        mode : str
            'long_only' or 'long_short'.
        position_sizing : str
            Position sizing method.
        target_volatility : float
            Target annual vol for vol_targeting sizing.
        initial_capital : float
            Starting capital.

        Returns
        -------
        dict
            Contains 'portfolio_value', 'returns', 'positions', 'stats'.
        # Unstack MultiIndex Series to DataFrame (columns=tickers, index=date)
        if isinstance(signals.index, pd.MultiIndex) and isinstance(signals, pd.Series):
            signals = signals.unstack(level=-1)
        if isinstance(prices.index, pd.MultiIndex) and isinstance(prices, pd.Series):
            prices = prices.unstack(level=-1)

        # Ensure aligned indices
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        # Convert probability signals to positions
        positions = self._signals_to_positions(signals, mode)

        # Apply position sizing
        sized_positions = self._apply_sizing(
            positions, prices, position_sizing, target_volatility
        )

        # Apply risk controls
        sized_positions = self._apply_risk_controls(sized_positions, prices)

        if HAS_VBT:
            result = self._run_vectorbt(sized_positions, prices, initial_capital)
        else:
            result = self._run_fallback(sized_positions, prices, initial_capital)

        return result

    def _signals_to_positions(
        self, signals: pd.Series | pd.DataFrame, mode: str
    ) -> pd.Series | pd.DataFrame:
        """Convert raw signals to position indicators."""
        if isinstance(signals, pd.DataFrame):
            positions = signals.copy()
        else:
            positions = signals.copy()

        # If signals are probabilities (0 to 1), center around 0
        if hasattr(positions, 'min') and hasattr(positions, 'max'):
            sig_min = positions.min() if isinstance(positions, pd.Series) else positions.min().min()
            sig_max = positions.max() if isinstance(positions, pd.Series) else positions.max().max()
            if 0 <= sig_min and sig_max <= 1:
                positions = positions - 0.5  # Center: >0 long, <0 short

        if mode == "long_only":
            positions = positions.clip(lower=0)

        # Normalize to [-1, 1]
        abs_max = positions.abs().max()
        if isinstance(abs_max, pd.Series):
            abs_max = abs_max.max()
        if abs_max > 0:
            positions = positions / abs_max

        return positions

    def _apply_sizing(
        self,
        positions: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        method: str,
        target_vol: float,
    ) -> pd.Series | pd.DataFrame:
        """Apply position sizing method."""
        if method == "equal_weight":
            return positions

        elif method == "vol_targeting":
            # Inverse volatility sizing
            if isinstance(prices, pd.DataFrame):
                returns = prices.pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            else:
                returns = prices.pct_change()
                rolling_vol = returns.rolling(20).std() * np.sqrt(252)

            vol_scalar = target_vol / rolling_vol.replace(0, np.nan).fillna(target_vol)
            vol_scalar = vol_scalar.clip(0, 2)  # Cap leverage at 2x
            return positions * vol_scalar

        elif method == "kelly":
            # Simplified Kelly: f = (p * b - q) / b where b=1, p=win_rate
            if isinstance(prices, pd.Series):
                returns = prices.pct_change()
                win_rate = (returns > 0).rolling(60).mean()
                kelly_frac = (2 * win_rate - 1).clip(0, 0.5)  # Half-Kelly
                return positions * kelly_frac.fillna(0.1)
            return positions  # Fallback for DataFrames

        return positions

    def _apply_risk_controls(
        self,
        positions: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """Apply turnover constraints."""
        if self.max_turnover is not None:
            if isinstance(positions, pd.Series):
                turnover = positions.diff().abs()
                cum_turnover = turnover.cumsum()
                daily_turnover_budget = self.max_turnover / 252
                excess = turnover > daily_turnover_budget * 3
                if excess.any():
                    positions = positions.copy()
                    positions[excess] = positions.shift(1)[excess]  # Hold previous

        return positions

    def _run_vectorbt(
        self,
        positions: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        initial_capital: float,
    ) -> dict:
        """Run backtest using vectorbt."""
        total_fees = self.commission + self.slippage

        if isinstance(prices, pd.DataFrame):
            # Multi-asset
            entries = (positions > 0) & (positions.shift(1) <= 0)
            exits = (positions <= 0) & (positions.shift(1) > 0)

            pf = vbt.Portfolio.from_signals(
                prices,
                entries=entries,
                exits=exits,
                fees=total_fees,
                init_cash=initial_capital,
                freq="1D",
                sl_stop=self.stop_loss,
                tp_stop=self.take_profit,
            )
        else:
            # Single asset
            entries = (positions > 0) & (positions.shift(1) <= 0)
            exits = (positions <= 0) & (positions.shift(1) > 0)

            pf = vbt.Portfolio.from_signals(
                prices,
                entries=entries,
                exits=exits,
                fees=total_fees,
                init_cash=initial_capital,
                freq="1D",
                sl_stop=self.stop_loss,
                tp_stop=self.take_profit,
            )

        portfolio_value = pf.value()
        returns = pf.returns()

        stats = {
            "total_return": float(pf.total_return()),
            "sharpe_ratio": float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0,
            "sortino_ratio": float(pf.sortino_ratio()) if not np.isnan(pf.sortino_ratio()) else 0.0,
            "max_drawdown": float(pf.max_drawdown()),
            "calmar_ratio": float(pf.calmar_ratio()) if not np.isnan(pf.calmar_ratio()) else 0.0,
            "win_rate": float(pf.trades.win_rate()) if len(pf.trades.records_readable) > 0 else 0.0,
            "profit_factor": float(pf.trades.profit_factor()) if len(pf.trades.records_readable) > 0 else 0.0,
            "total_trades": int(pf.trades.count()),
            "initial_capital": initial_capital,
        }

        return {
            "portfolio_value": portfolio_value,
            "returns": returns,
            "positions": positions,
            "stats": stats,
            "portfolio_object": pf,
        }

    def _run_fallback(
        self,
        positions: pd.Series | pd.DataFrame,
        prices: pd.Series | pd.DataFrame,
        initial_capital: float,
    ) -> dict:
        """Simple fallback backtest without vectorbt."""
        if isinstance(prices, pd.DataFrame):
            price_returns = prices.pct_change().fillna(0)
            # Portfolio return = sum of (position * asset_return) per day
            strategy_returns = (positions.shift(1) * price_returns).sum(axis=1)
        else:
            price_returns = prices.pct_change().fillna(0)
            strategy_returns = positions.shift(1) * price_returns

        # Apply costs
        total_fees = self.commission + self.slippage
        if isinstance(positions, pd.DataFrame):
            turnover = positions.diff().abs().sum(axis=1).fillna(0)
        else:
            turnover = positions.diff().abs().fillna(0)
        cost = turnover * total_fees
        net_returns = strategy_returns - cost

        portfolio_value = initial_capital * (1 + net_returns).cumprod()

        # Apply max drawdown halt
        if self.max_drawdown is not None:
            peak = portfolio_value.cummax()
            dd = (portfolio_value - peak) / peak
            halt_mask = dd < -self.max_drawdown
            if halt_mask.any():
                first_halt = halt_mask.idxmax()
                net_returns.loc[first_halt:] = 0
                portfolio_value = initial_capital * (1 + net_returns).cumprod()
                logger.warning("Max drawdown %.1f%% breached — halted trading.", self.max_drawdown * 100)

        # Compute stats
        ann_return = net_returns.mean() * 252
        ann_vol = net_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        neg_returns = net_returns[net_returns < 0]
        downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 1e-6
        sortino = ann_return / downside_vol

        peak = portfolio_value.cummax()
        max_dd = ((portfolio_value - peak) / peak).min()

        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
        total_return = (portfolio_value.iloc[-1] / initial_capital) - 1

        win_rate = (net_returns > 0).sum() / (net_returns != 0).sum() if (net_returns != 0).sum() > 0 else 0.0

        gross_profit = net_returns[net_returns > 0].sum()
        gross_loss = abs(net_returns[net_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        stats = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "annual_return": float(ann_return),
            "annual_volatility": float(ann_vol),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "initial_capital": initial_capital,
        }

        return {
            "portfolio_value": portfolio_value,
            "returns": net_returns,
            "positions": positions,
            "stats": stats,
        }
