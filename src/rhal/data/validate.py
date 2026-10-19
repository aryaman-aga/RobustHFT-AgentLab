"""Schema validation for price and news frames.

Both validators return a *cleaned copy* (sorted, typed) and raise ``ValueError``
with an explicit message on structural problems so failures are loud.
"""
from __future__ import annotations

import pandas as pd

PRICE_COLS = ["date", "open", "high", "low", "close", "volume"]
NEWS_COLS = ["id", "date", "text"]


REPAIR_TOL = 0.005  # real vendor data: high/low may miss open/close by float noise; fix up to 0.5%, reject more


def validate_prices(df: pd.DataFrame, repair: bool = False) -> pd.DataFrame:
    """Clean, typed, sorted copy of an OHLCV frame.

    ``repair=True`` is for real downloaded data: rows with missing values are dropped, and a high
    (low) that is below (above) open/close by at most ``REPAIR_TOL`` is widened to fit. Anything
    larger still raises, and so does zero volume (it would break volume_shock and Amihud).
    """
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"prices missing columns: {missing}")
    out = df[PRICE_COLS + [c for c in df.columns if c not in PRICE_COLS]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    if repair:
        out = out.dropna(subset=PRICE_COLS).reset_index(drop=True)
        if (out["volume"] <= 0).any():
            raise ValueError("zero or negative volume in real prices")
        top, bot = out[["open", "close", "low"]].max(axis=1), out[["open", "close", "high"]].min(axis=1)
        if ((top - out["high"]) / top > REPAIR_TOL).any() or ((out["low"] - bot) / bot > REPAIR_TOL).any():
            raise ValueError(f"high/low inconsistent with open/close by more than {REPAIR_TOL:.1%}")
        out["high"] = out[["high", "open", "close", "low"]].max(axis=1)
        out["low"] = out[["low", "open", "close", "high"]].min(axis=1)
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices found")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high < max(open, close, low) for some rows")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low > min(open, close, high) for some rows")
    if out["date"].duplicated().any():
        raise ValueError("duplicate dates in prices")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def validate_news(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NEWS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"news missing columns: {missing}")
    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["text"] = out["text"].astype(str)
    if "provenance" not in out.columns:
        out["provenance"] = "clean"
    if "true_sentiment" not in out.columns:
        out["true_sentiment"] = float("nan")
    if out["id"].duplicated().any():
        raise ValueError("duplicate news ids")
    if (out["text"].str.len() < 10).any():
        raise ValueError("news text too short (<10 chars) for some rows")
    return out.sort_values(["date", "id"]).reset_index(drop=True)
