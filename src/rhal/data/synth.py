"""Synthetic market + news generator (fully offline, deterministic).

Price process
-------------
Regime-switching drift (bull / sideways / bear via a 3-state Markov chain)
with GARCH(1,1)-style conditional volatility. Overnight gaps are small and
independent of the day's drift so that most of a day's return is realised
*after* the next-open execution point -- this keeps news informative after
execution without any lookahead in the agent.

News process
------------
Each day t emits ``news_per_day`` items whose latent sentiment is
``s = rho * z_{t+1} + sqrt(1 - rho^2) * eps``  (rho ~ 0.35), where z_{t+1} is the
standardised *next-day* return. That is, news is genuinely but noisily
predictive. Items are rendered from finance templates that always contain
entities, numbers, causes and time references so every MisGen perturbation
type has something to act on.

Optional intraday bars
----------------------
Brownian-bridge minute bars from open to close with a U-shaped volume profile
and a tick-rule buy/sell proxy, used only to derive microstructure features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import lexicon as L
from ..seeding import rng


@dataclass
class SynthConfig:
    n_days: int = 400
    news_per_day: int = 5
    start_date: str = "2023-01-02"
    rho: float = 0.35  # news informativeness
    intraday: bool = False
    bars_per_day: int = 78
    symbol: str = "SYNTH"


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def _simulate_prices(cfg: SynthConfig, seed: int) -> pd.DataFrame:
    g = rng(seed, "prices")
    n = cfg.n_days
    # regimes: 0 bull, 1 sideways, 2 bear ; daily drift and base vol
    drift = np.array([0.0012, 0.0000, -0.0010])
    P = np.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.02, 0.97]])
    regime = np.zeros(n, dtype=int)
    regime[0] = g.integers(0, 3)
    for t in range(1, n):
        regime[t] = g.choice(3, p=P[regime[t - 1]])

    # GARCH(1,1): sigma^2_t = w + a*eps^2_{t-1} + b*sigma^2_{t-1}
    omega, a, b = 0.000004, 0.08, 0.90
    sigma2 = np.empty(n)
    ret = np.empty(n)
    sigma2[0] = omega / (1 - a - b)
    eps_prev = 0.0
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + a * eps_prev**2 + b * sigma2[t - 1]
        eps = np.sqrt(sigma2[t]) * g.standard_normal()
        ret[t] = drift[regime[t]] + eps
        eps_prev = eps

    close = 100.0 * np.exp(np.cumsum(ret))
    prev_close = np.concatenate([[100.0], close[:-1]])
    gap = g.normal(0, 0.0015, n)  # small overnight gap, independent of drift
    open_ = prev_close * np.exp(gap)
    # intraday range from realised vol
    sig = np.sqrt(sigma2)
    rng_frac = np.abs(g.normal(0, 1, n)) * sig * 1.2 + 0.002
    hi = np.maximum(open_, close) * (1 + rng_frac * g.uniform(0.3, 1.0, n))
    lo = np.minimum(open_, close) * (1 - rng_frac * g.uniform(0.3, 1.0, n))
    base_vol = 2_000_000
    volume = base_vol * (1 + 2.5 * np.abs(ret) / (sig + 1e-9) * 0.3) * np.exp(g.normal(0, 0.25, n))
    dates = pd.bdate_range(cfg.start_date, periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": hi,
            "low": lo,
            "close": close,
            "volume": volume.round(0),
            "regime": regime,  # ground-truth latent regime (diagnostics only; never fed to the agent)
        }
    )
    return df


def _simulate_intraday(daily: pd.DataFrame, cfg: SynthConfig, seed: int) -> pd.DataFrame:
    """Brownian-bridge minute bars consistent with daily OHLC."""
    g = rng(seed, "intraday")
    rows = []
    m = cfg.bars_per_day
    # U-shaped volume profile
    x = np.linspace(-1, 1, m)
    profile = 0.6 + 0.8 * x**2
    profile /= profile.sum()
    for _, r in daily.iterrows():
        o, c, h, l, v = r["open"], r["close"], r["high"], r["low"], r["volume"]
        lo_ret = np.log(c / o)
        # Brownian bridge in log space
        z = g.standard_normal(m)
        w = np.cumsum(z) / np.sqrt(m)
        t = np.arange(1, m + 1) / m
        bridge = w - t * w[-1]
        vol_scale = max(abs(np.log(h / l)), 1e-4) * 0.35
        path = np.log(o) + t * lo_ret + vol_scale * bridge
        px = np.exp(path)
        # clip to the day's range so intraday bars respect daily high/low
        px = np.clip(px, l, h)
        px[-1] = c
        bar_open = np.concatenate([[o], px[:-1]])
        bar_close = px
        bar_high = np.maximum(bar_open, bar_close) * (1 + np.abs(g.normal(0, 0.0004, m)))
        bar_low = np.minimum(bar_open, bar_close) * (1 - np.abs(g.normal(0, 0.0004, m)))
        bar_vol = np.maximum(1, (v * profile * np.exp(g.normal(0, 0.3, m))).round())
        # tick-rule buy/sell proxy: sign of price change classifies the bar's volume
        sgn = np.sign(bar_close - bar_open)
        sgn[sgn == 0] = 1
        buy_vol = np.where(sgn > 0, bar_vol, 0.0)
        sell_vol = np.where(sgn < 0, bar_vol, 0.0)
        for i in range(m):
            rows.append((r["date"], i, bar_open[i], bar_high[i], bar_low[i], bar_close[i], bar_vol[i], buy_vol[i], sell_vol[i]))
    return pd.DataFrame(rows, columns=["date", "bar", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def _bucket(s: float) -> str:
    if s > 0.5:
        return "strong_pos"
    if s > 0.15:
        return "mild_pos"
    if s < -0.5:
        return "strong_neg"
    if s < -0.15:
        return "mild_neg"
    return "neutral"


_VERBS = {
    "strong_pos": ["surged", "rallied", "jumped", "soared"],
    "mild_pos": ["rose", "gained", "climbed", "edged higher"],
    "neutral": ["held steady", "was little changed", "traded flat"],
    "mild_neg": ["slipped", "fell", "declined", "edged lower"],
    "strong_neg": ["plunged", "slumped", "tumbled", "collapsed"],
}
_ADJ = {
    "strong_pos": ["strong", "robust", "record"],
    "mild_pos": ["solid", "improved", "resilient"],
    "neutral": ["in line", "stable", "mixed"],
    "mild_neg": ["muted", "softened", "cautious"],
    "strong_neg": ["weak", "disappointing", "fragile"],
}
_RESULT = {
    "strong_pos": "beat", "mild_pos": "modestly beat", "neutral": "matched", "mild_neg": "slightly missed", "strong_neg": "missed",
}
_BANKS = ["Meridian Capital", "Alder & Rowe", "Cobalt Securities", "Fairhaven Partners"]
_REGULATORS = ["The federal competition authority", "The securities regulator", "The data-protection office"]


def _render_news(g: np.random.Generator, s: float, date: str, prev_close: float) -> tuple[str, str]:
    """Render one headline-plus-lede paragraph for latent sentiment s. Returns (text, category)."""
    b = _bucket(s)
    E = L.ENTITY
    verb = g.choice(_VERBS[b])
    adj = g.choice(_ADJ[b])
    pct = round(float(abs(s) * g.uniform(1.5, 6.0) + g.uniform(0.2, 1.0)), 1)
    cause_pool = L.CAUSES_POS if s > 0.15 else L.CAUSES_NEG if s < -0.15 else L.CAUSES_NEUTRAL
    cause = g.choice(cause_pool)
    conn = g.choice(L.CAUSAL_CONNECTORS)
    q = g.choice(L.QUARTERS)
    yr = int(g.choice(L.YEARS))
    hedge = g.choice(L.HEDGES) if abs(s) < 0.4 else g.choice(L.INTENSIFIERS)
    rev = round(float(g.uniform(1.2, 4.8)), 2)
    yoy = round(float(np.sign(s) * pct * g.uniform(1.0, 2.5) + (0 if abs(s) > 0.15 else g.uniform(-1, 1))), 1)
    margin = round(float(g.uniform(18, 34)), 1)
    pt = round(prev_close * (1 + s * g.uniform(0.05, 0.2)), 0)
    kind = g.choice(["earnings", "guidance", "analyst", "sector", "product", "regulatory", "flow", "margin"])

    if kind == "earnings":
        text = (
            f"{E['company']} ({E['ticker']}) reported {q} {yr} revenue of ${rev}B, {'up' if yoy >= 0 else 'down'} {abs(yoy)}% year over year, "
            f"{conn} {cause}. The result {_RESULT[b]} consensus estimates and analysts described the quarter as {adj}. "
            f"Shares {verb} {pct}% in early trading."
        )
    elif kind == "guidance":
        action = "raised" if s > 0.15 else "cut" if s < -0.15 else "reaffirmed"
        text = (
            f"{E['company']} {action} its full-year {yr} revenue guidance to ${round(rev * 4, 1)}B {conn} {cause}. "
            f"Management sounded {adj} on the call, and the stock {verb} {pct}% after hours."
        )
    elif kind == "analyst":
        act = "upgraded" if s > 0.15 else "downgraded" if s < -0.15 else "maintained"
        rating = "outperform" if s > 0.15 else "underperform" if s < -0.15 else "market perform"
        text = (
            f"{g.choice(_BANKS)} {act} {E['company']} to {rating}, citing {cause}. The price target was set at ${pt:.0f}, "
            f"and the analyst called the setup {hedge} {adj}. Shares {verb} {pct}% on the note."
        )
    elif kind == "sector":
        rank = "best" if s > 0.15 else "worst" if s < -0.15 else "middle"
        text = (
            f"Shares of {E['sector']} companies including {E['company']} {verb} {pct}% {conn} {cause}, "
            f"with {E['company']} among the {rank} performers in the group. Sentiment across the sector was {adj}."
        )
    elif kind == "product":
        ev = "a major expansion of its cloud platform" if s > 0 else "a delay to its next platform release"
        text = (
            f"{E['company']} announced {ev} {conn} {cause}. The company said the move would {hedge} "
            f"{'accelerate' if s > 0 else 'weigh on'} bookings in {q} {yr}; shares {verb} {pct}%."
        )
    elif kind == "regulatory":
        ra = "cleared" if s > 0.15 else "opened an investigation into" if s < -0.15 else "requested information from"
        text = (
            f"{g.choice(_REGULATORS)} {ra} {E['company']} {conn} {cause}. The company expects the {adj} outcome to "
            f"{'support' if s > 0 else 'pressure'} operating margin, and the stock {verb} {pct}%."
        )
    elif kind == "flow":
        vd = "heavy" if abs(s) > 0.4 else "light"
        text = (
            f"Trading volume in {E['ticker']} was {vd} as the stock {verb} {pct}% {conn} {cause}. "
            f"Order flow appeared {g.choice(_ADJ[b][:2]) if b != 'strong_pos' else 'strong'}, and dealers noted {hedge} {'buying' if s > 0 else 'selling'} interest into the close."
        )
    else:  # margin
        mv = "expanded" if s > 0.15 else "contracted" if s < -0.15 else "was flat at"
        text = (
            f"{E['company']}'s operating margin {mv}{"" if mv.endswith("at") else " to"} {margin}% in {q} {yr}, {conn} {cause}. "
            f"Free cash flow was {adj}, and shares {verb} {pct}% following the release."
        )
    return text, kind


def _simulate_news(prices: pd.DataFrame, cfg: SynthConfig, seed: int) -> pd.DataFrame:
    g = rng(seed, "news")
    close = prices["close"].to_numpy()
    ret = np.diff(np.log(close))  # ret[t] = log(close[t+1]/close[t])
    z = (ret - ret.mean()) / (ret.std() + 1e-12)
    rows = []
    n = len(prices)
    for t in range(n - 1):  # last day has no next-day return; emit none
        z_next = float(np.clip(z[t], -3, 3)) / 3.0  # scale to ~[-1, 1]
        for j in range(cfg.news_per_day):
            eps = g.normal(0, 1) / 3.0
            s = float(np.clip(cfg.rho * z_next * 3 + np.sqrt(1 - cfg.rho**2) * eps * 3, -1, 1))
            text, kind = _render_news(g, s, prices.loc[t, "date"], float(prices.loc[t, "close"]))
            rows.append(
                {
                    "id": f"n{t:04d}_{j}",
                    "date": prices.loc[t, "date"],
                    "text": text,
                    "provenance": "clean",
                    "true_sentiment": round(s, 4),
                    "category": kind,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(cfg: SynthConfig, seed: int) -> dict[str, pd.DataFrame]:
    """Return ``{"prices": df, "news": df, ["intraday": df]}``."""
    prices = _simulate_prices(cfg, seed)
    news = _simulate_news(prices, cfg, seed)
    out = {"prices": prices, "news": news}
    if cfg.intraday:
        out["intraday"] = _simulate_intraday(prices, cfg, seed)
    return out
