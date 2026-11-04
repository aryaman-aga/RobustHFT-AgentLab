"""Rolling normalisation, feature-frame assembly, and the market signal.

Normalisation is a *trailing* z-score (window = ``norm_window``) so that the
value at t never depends on rows after t. Full-sample z-scoring would leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import compute_core_features
from .micro import MICRO_COLS, compute_micro_features

CORE_COLS = ["ret_1", "vol_5", "dd_20", "trend_div", "anomaly", "mom_5", "mom_20"]

# Fixed, documented weights for the market signal (no fitting -> no leakage).
# Sign convention: positive = bullish evidence.
MARKET_SIGNAL_WEIGHTS: dict[str, float] = {
    "mom_5": 0.35,
    "trend_div": 0.35,
    "mom_20": 0.15,
    "anomaly": -0.10,  # extreme deviation from trailing mean is a mild mean-reversion signal
    "ofi_proxy": 0.30,  # micro: buying pressure
    "volume_shock": 0.05,
    "spread_proxy": -0.10,  # micro: wide spread = stress
    "rv_gk": -0.10,  # micro: high realised variance = stress
}


def rolling_zscore(df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    m = df.rolling(window, min_periods=min_periods).mean()
    s = df.rolling(window, min_periods=min_periods).std()
    z = (df - m) / (s + 1e-12)
    return z.clip(-4, 4)


def build_feature_frame(
    prices: pd.DataFrame,
    intraday: pd.DataFrame | None = None,
    *,
    vol_window: int = 5,
    drawdown_window: int = 20,
    short_trend: int = 5,
    long_trend: int = 20,
    anomaly_window: int = 20,
    micro: bool = True,
    rv_window: int = 5,
    volume_window: int = 20,
    norm_window: int = 60,
) -> pd.DataFrame:
    """Return frame with date, raw features, and z_* normalised features."""
    core = compute_core_features(prices, vol_window, drawdown_window, short_trend, long_trend, anomaly_window)
    parts = [core]
    if micro:
        parts.append(compute_micro_features(prices, intraday, rv_window, volume_window))
    raw = pd.concat(parts, axis=1)
    z = rolling_zscore(raw, window=norm_window).add_prefix("z_")
    out = pd.concat(
        [prices[["date"]].reset_index(drop=True), raw.reset_index(drop=True), z.reset_index(drop=True)], axis=1
    )
    return out


def feature_columns(micro: bool) -> list[str]:
    return CORE_COLS + (MICRO_COLS if micro else [])


def market_signal(z_row: pd.Series | dict, micro: bool = True) -> float:
    """Bounded market evidence in [-1, 1] from a row of z-scored features."""
    acc = 0.0
    for k, w in MARKET_SIGNAL_WEIGHTS.items():
        if not micro and k in MICRO_COLS:
            continue
        v = z_row.get(f"z_{k}", np.nan) if isinstance(z_row, dict) else z_row.get(f"z_{k}", np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        acc += w * float(v)
    return float(np.tanh(acc))
