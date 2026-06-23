"""
MarketDataset — unified data loader for yfinance and CSV (JPX) sources.

Returns standardised DataFrames with MultiIndex (date, ticker) and
columns [open, high, low, close, volume, adj_close].
Enforces strict temporal splits to prevent look-ahead leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class MarketDataset:
    """Load, clean, resample, and split market OHLCV data."""

    # Canonical column names used throughout the pipeline
    COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]

    def __init__(
        self,
        tickers: list[str] | None = None,
        start_date: str = "2015-01-01",
        end_date: str = "2026-06-01",
        data_dir: str | Path = "data",
    ) -> None:
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY", "QQQ"]
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.raw_dir = ensure_dir(self.data_dir / "raw")
        self.processed_dir = ensure_dir(self.data_dir / "processed")
        self.data: pd.DataFrame | None = None

    # ── yfinance ──────────────────────────────────────────────────────
    def load_yfinance(self) -> pd.DataFrame:
        """Download multi-ticker OHLCV data from Yahoo Finance.

        Returns
        -------
        pd.DataFrame
            MultiIndex (date, ticker) with standardised column names.
        """
        logger.info(
            "Downloading %d tickers from yfinance: %s → %s",
            len(self.tickers),
            self.start_date,
            self.end_date,
        )

        frames: list[pd.DataFrame] = []
        for ticker in self.tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False,
                )
                if raw.empty:
                    logger.warning("No data returned for %s — skipping.", ticker)
                    continue

                # Flatten MultiIndex columns if present (yfinance sometimes returns them)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = pd.DataFrame(
                    {
                        "open": raw["Open"],
                        "high": raw["High"],
                        "low": raw["Low"],
                        "close": raw["Close"],
                        "volume": raw["Volume"],
                        "adj_close": raw.get("Adj Close", raw["Close"]),
                    }
                )
                df["ticker"] = ticker
                df.index.name = "date"
                frames.append(df.reset_index())
            except Exception as exc:
                logger.error("Failed to download %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No data downloaded for any ticker.")

        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index(["date", "ticker"]).sort_index()

        # Persist raw data
        out_path = self.raw_dir / "yfinance_ohlcv.parquet"
        combined.to_parquet(out_path)
        logger.info("Saved raw yfinance data → %s (%d rows)", out_path, len(combined))

        self.data = combined
        return self.data

    # ── CSV (JPX) ─────────────────────────────────────────────────────
    def load_csv(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load OHLCV data from CSV files (e.g. JPX dataset).

        Expected CSV schema (at minimum):
            Date, SecuritiesCode (or ticker), Open, High, Low, Close, Volume

        Parameters
        ----------
        path : str | Path, optional
            Directory containing CSV files or a single CSV file.
            Defaults to ``data/raw/jpx/``.

        Returns
        -------
        pd.DataFrame
            MultiIndex (date, ticker).
        """
        csv_path = Path(path) if path else self.raw_dir / "jpx"

        if csv_path.is_dir():
            csv_files = sorted(csv_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {csv_path}")
            frames = [pd.read_csv(f) for f in csv_files]
            raw = pd.concat(frames, ignore_index=True)
        elif csv_path.is_file():
            raw = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Path does not exist: {csv_path}")

        logger.info("Loaded CSV data — %d rows, %d columns", len(raw), len(raw.columns))

        # Normalise column names to lowercase
        raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

        # Map common column name variants
        col_map = {
            "securitiescode": "ticker",
            "securities_code": "ticker",
            "symbol": "ticker",
            "adj_close": "adj_close",
            "adjustedclose": "adj_close",
            "adjusted_close": "adj_close",
        }
        raw = raw.rename(columns=col_map)

        # Ensure required columns exist
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw["close"]

        raw["date"] = pd.to_datetime(raw["date"])
        raw["ticker"] = raw["ticker"].astype(str)

        combined = (
            raw[["date", "ticker"] + self.COLUMNS]
            .set_index(["date", "ticker"])
            .sort_index()
        )

        self.data = combined
        return self.data

    # ── Resampling ────────────────────────────────────────────────────
    def resample(self, freq: str = "W") -> pd.DataFrame:
        """Resample OHLCV data to a lower frequency (W = weekly, M = monthly).

        Uses OHLCV-appropriate aggregation (first/max/min/last/sum).
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_yfinance() or load_csv() first.")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "adj_close": "last",
        }

        resampled = (
            self.data.groupby(level="ticker")
            .resample(freq, level="date")
            .agg(agg_dict)
        )

        # Fix index ordering
        resampled = resampled.reorder_levels(["date", "ticker"]).sort_index()
        self.data = resampled
        logger.info("Resampled to %s — %d rows", freq, len(self.data))
        return self.data

    # ── Missing data ──────────────────────────────────────────────────
    def clean_missing(self, threshold: float = 0.3) -> pd.DataFrame:
        """Clean missing data.

        1. Drop columns where >threshold fraction is NaN.
        2. Forward-fill remaining NaNs per ticker.
        3. Back-fill any leading NaNs.
        4. Drop any rows still containing NaN.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")

        n_before = len(self.data)

        # Drop columns with too many NaNs
        col_missing = self.data.isna().mean()
        bad_cols = col_missing[col_missing > threshold].index.tolist()
        if bad_cols:
            logger.warning("Dropping columns with >%.0f%% missing: %s", threshold * 100, bad_cols)
            self.data = self.data.drop(columns=bad_cols)

        # Forward-fill and back-fill per ticker
        self.data = self.data.groupby(level="ticker").ffill()
        self.data = self.data.groupby(level="ticker").bfill()

        # Drop any remaining NaN rows
        self.data = self.data.dropna()

        logger.info(
            "Cleaned missing data: %d → %d rows (dropped %d)",
            n_before,
            len(self.data),
            n_before - len(self.data),
        )
        return self.data

    # ── Temporal split ────────────────────────────────────────────────
    def split_time_series(
        self,
        train_end: str = "2021-12-31",
        val_end: str = "2023-12-31",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train / validation / test using strict date cutoffs.

        No random shuffling — prevents look-ahead leakage.

        Parameters
        ----------
        train_end : str
            Last date (inclusive) for training data.
        val_end : str
            Last date (inclusive) for validation data.
            Everything after val_end becomes the test set.

        Returns
        -------
        (train, val, test) : tuple[pd.DataFrame, ...]
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")

        dates = self.data.index.get_level_values("date")
        train_mask = dates <= pd.Timestamp(train_end)
        val_mask = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(val_end))
        test_mask = dates > pd.Timestamp(val_end)

        train = self.data.loc[train_mask].copy()
        val = self.data.loc[val_mask].copy()
        test = self.data.loc[test_mask].copy()

        logger.info(
            "Split — train: %d rows (%s), val: %d rows (%s), test: %d rows (%s)",
            len(train),
            train_end,
            len(val),
            val_end,
            len(test),
            f">{val_end}",
        )

        # Persist processed splits
        for name, df in [("train", train), ("val", val), ("test", test)]:
            out = self.processed_dir / f"{name}.parquet"
            df.to_parquet(out)

        return train, val, test

    # ── Convenience ───────────────────────────────────────────────────
    def load_and_prepare(
        self,
        source: Literal["yfinance", "csv"] = "yfinance",
        csv_path: str | Path | None = None,
        resample_freq: str | None = None,
        missing_threshold: float = 0.3,
        train_end: str = "2021-12-31",
        val_end: str = "2023-12-31",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """End-to-end: load → clean → (optional resample) → split."""
        if source == "yfinance":
            self.load_yfinance()
        else:
            self.load_csv(csv_path)

        self.clean_missing(threshold=missing_threshold)

        if resample_freq:
            self.resample(resample_freq)

        return self.split_time_series(train_end=train_end, val_end=val_end)

    def __repr__(self) -> str:
        n = len(self.data) if self.data is not None else 0
        return f"MarketDataset(tickers={self.tickers}, rows={n})"
