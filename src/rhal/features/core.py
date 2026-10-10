"""Core time-series features (paper Eq. 19).

NO-LOOKAHEAD CONTRACT: every value in row t is a function of rows <= t only.
Rolling operations are trailing; nothing uses ``.shift(-k)`` or full-sample
statistics. The test-suite verifies this by mutating future rows and asserting
that rows <= t are byte-identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_core_features(
    prices: pd.DataFrame,
    vol_window: int = 5,
    drawdown_window: int = 20,
    short_trend: int = 5,
    long_trend: int = 20,
    anomaly_window: int = 20,
) -> pd.DataFrame:
    """Return a frame indexed like ``prices`` with core features.

    Columns: ret_1, vol_5, dd_20, trend_div, anomaly, mom_5, mom_20
    (names fixed regardless of window params to keep downstream code simple).
    """
    c = prices["close"].astype(float)
    logc = np.log(c)
    out = pd.DataFrame(index=prices.index)
    out["ret_1"] = logc.diff()
    out["vol_5"] = out["ret_1"].rolling(vol_window, min_periods=vol_window).std()
    roll_max = c.rolling(drawdown_window, min_periods=1).max()
    out["dd_20"] = c / roll_max - 1.0  # <= 0
    sma_s = c.rolling(short_trend, min_periods=short_trend).mean()
    sma_l = c.rolling(long_trend, min_periods=long_trend).mean()
    out["trend_div"] = (sma_s - sma_l) / sma_l
    # residual anomaly: deviation of close from trailing mean in trailing-std units.
    # (Simplification of the paper's seasonal-decomposition residual; STL needs
    # the full series and is therefore not lookahead-safe in an online setting.)
    m = c.rolling(anomaly_window, min_periods=anomaly_window).mean()
    s = c.rolling(anomaly_window, min_periods=anomaly_window).std()
    out["anomaly"] = (c - m) / (s + 1e-12)
    out["mom_5"] = logc - logc.shift(short_trend)
    out["mom_20"] = logc - logc.shift(long_trend)
    return out
