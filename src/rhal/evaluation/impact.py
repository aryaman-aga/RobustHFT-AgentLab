"""Price-impact stress test: misinformation that also moves the market.

In the main experiment injected news never moves prices, so the market evidence used by
grounding is clean by construction. Here a recorded attack (an AttackTape) defines a moved
price path that BOTH twins trade, while only the treated agent reads the fake text:

    shock_t = gamma * sigma_{t-1} * sum_{fake d published on day t} [ s(d) - s(parent(d)) ]

where s is the lexicon sentiment (the market's reading of the text) and sigma_{t-1} the
trailing 20-day volatility of daily log returns up to the previous day. Each published fake
article moves the price (a day's edits often push in opposite directions, so a mean would
cancel them). The shock moves the
close of day t (news published during day t is priced in by the close the agent sees) and is
permanent: every later open/high/low/close is scaled by exp(cumulative shock). gamma = 0
returns the dataset unchanged; gamma = 1 moves the price by one daily sigma per unit of
sentiment shift. Features are rebuilt from the moved prices with the usual trailing windows,
so nothing at day t depends on news or prices after t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agent.sentiment import score_sentiment
from ..config import Config
from ..features import build_feature_frame

# The tape is an AttackTape from experiment.py; typed loosely to avoid a circular import.


def sentiment_shifts(ds, tape) -> dict[int, float]:
    """Day index -> summed lexicon-sentiment shift of the fake articles published that day."""
    parent = dict(zip(ds.news["id"].astype(str), ds.news["text"].astype(str)))
    idx = {d: i for i, d in enumerate(ds.dates)}
    out: dict[int, float] = {}
    for date, docs in tape.docs.items():
        docs = [d for d in docs if d.parent_id in parent]
        if docs and date in idx:
            out[idx[date]] = float(sum(score_sentiment(d.text) - score_sentiment(parent[d.parent_id]) for d in docs))
    return out


def impact_shocks(ds, tape, gamma: float, vol_window: int = 20) -> np.ndarray:
    """Per-day log-price shock implied by the tape (zero on days without injected documents)."""
    shock = np.zeros(len(ds.dates))
    if gamma == 0.0:
        return shock
    close = ds.prices["close"].to_numpy(dtype=float)
    lr = np.diff(np.log(close), prepend=np.nan)
    sigma = pd.Series(lr).rolling(vol_window, min_periods=5).std().shift(1).to_numpy()  # up to t-1 only
    for t, delta in sentiment_shifts(ds, tape).items():
        if np.isfinite(sigma[t]):
            shock[t] = gamma * sigma[t] * delta
    return shock


def impacted_dataset(ds, tape, gamma: float, cfg: Config):
    """Copy of ``ds`` whose prices carry the tape's permanent price impact; features rebuilt."""
    if gamma == 0.0:
        return ds
    from .experiment import Dataset  # local import: experiment imports this module's users

    shock = impact_shocks(ds, tape, gamma)
    after_close = np.exp(np.cumsum(shock))  # factor on the close of day t (includes today's shock)
    before_open = np.exp(np.cumsum(shock) - shock)  # factor on the open of day t (earlier shocks only)
    p = ds.prices.copy()
    p["open"] = p["open"] * before_open
    p["close"] = p["close"] * after_close
    p["high"] = np.maximum(p["high"] * before_open, p[["open", "close"]].max(axis=1))
    p["low"] = np.minimum(p["low"] * before_open, p[["open", "close"]].min(axis=1))
    f = cfg.features
    feats = build_feature_frame(p, ds.intraday, vol_window=f.vol_window, drawdown_window=f.drawdown_window, short_trend=f.short_trend,
                                long_trend=f.long_trend, anomaly_window=f.anomaly_window, micro=True, rv_window=f.micro.rv_window,
                                volume_window=f.micro.volume_window)
    feats["close"] = p["close"].to_numpy()
    return Dataset(p, ds.news, ds.intraday, feats, ds.dates)
