"""Data loaders: CSV (default), optional yfinance download.

No loader touches a brokerage. yfinance is a public historical data source and
is imported lazily so the package works without it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validate import validate_news, validate_prices


def load_prices(path: str | Path) -> pd.DataFrame:
    return validate_prices(pd.read_csv(path))


def load_news(path: str | Path) -> pd.DataFrame:
    return validate_news(pd.read_csv(path))


def download_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance (optional dependency).

    Raises ImportError with a clear message if yfinance is not installed.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:  # pragma: no cover - optional path
        raise ImportError("yfinance is optional: pip install yfinance") from e
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"no data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.reset_index().rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    return validate_prices(df[["date", "open", "high", "low", "close", "volume"]])
