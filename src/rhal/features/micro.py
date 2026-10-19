"""HFT / microstructure-inspired proxy features (our extension over the paper).

All proxies are computed from daily OHLCV (or from intraday bars aggregated to
the day when available) and obey the same trailing-only contract as core.py.

* spread_proxy   : (high - low) / close  -- Corwin-Schultz-style range proxy
* rv_gk          : Garman-Klass realised variance, trailing mean over rv_window
* ofi_proxy      : order-flow imbalance. With intraday bars: (buy - sell)/(buy + sell)
                   via tick rule. Without: close-location value (close-open)/(high-low).
* volume_shock   : log(volume / trailing-mean volume)
* illiquidity    : Amihud |ret| / dollar-volume, trailing mean, scaled
* range_ratio    : today's range / trailing-mean range
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_micro_features(
    prices: pd.DataFrame,
    intraday: pd.DataFrame | None = None,
    rv_window: int = 5,
    volume_window: int = 20,
) -> pd.DataFrame:
    o, h, lo, c, v = (prices[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=prices.index)
    rng_ = (h - lo).clip(lower=1e-9)
    out["spread_proxy"] = rng_ / c
    gk = 0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    out["rv_gk"] = gk.rolling(rv_window, min_periods=rv_window).mean()
    if intraday is not None and {"buy_volume", "sell_volume"}.issubset(intraday.columns):
        agg = intraday.groupby("date")[["buy_volume", "sell_volume"]].sum()
        agg = agg.reindex(prices["date"].to_numpy())
        ofi = ((agg["buy_volume"] - agg["sell_volume"]) / (agg["buy_volume"] + agg["sell_volume"] + 1e-9)).to_numpy()
        out["ofi_proxy"] = ofi
    else:
        out["ofi_proxy"] = ((c - o) / rng_).clip(-1, 1)
    vmean = v.rolling(volume_window, min_periods=volume_window).mean()
    out["volume_shock"] = np.log(v / vmean)
    ret = np.log(c).diff().abs()
    amihud = ret / (c * v) * 1e9
    out["illiquidity"] = amihud.rolling(rv_window, min_periods=rv_window).mean()
    out["range_ratio"] = rng_ / rng_.rolling(volume_window, min_periods=volume_window).mean()
    return out


MICRO_COLS = ["spread_proxy", "rv_gk", "ofi_proxy", "volume_shock", "illiquidity", "range_ratio"]
