"""Turn features + analogs into structured temporal evidence and a consistency score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.windows import market_signal
from ..schemas import MarketEvidence
from .analogs import AnalogRetriever


def consistency_score(text_signal: float, market_signal_: float) -> float:
    """1 when text and market agree perfectly, 0 when maximally opposed."""
    return float(np.clip(1.0 - abs(text_signal - market_signal_) / 2.0, 0.0, 1.0))


class EvidenceBuilder:
    def __init__(self, feats: pd.DataFrame, z_cols: list[str], *, micro: bool, window: int, horizon: int, top_k: int, use_analogs: bool = True):
        self.feats = feats
        self.z_cols = z_cols
        self.micro = micro
        self.use_analogs = use_analogs
        self.retriever = AnalogRetriever(feats, z_cols, window=window, horizon=horizon, top_k=top_k)

    def build(self, t: int) -> MarketEvidence:
        row = self.feats.iloc[t]
        zdict = {c: float(row[c]) for c in self.z_cols}
        base = market_signal(row, micro=self.micro)
        analogs = self.retriever.retrieve(t) if self.use_analogs else []
        mean_fwd = None
        hit = None
        sig = base
        if analogs:
            fwd = np.array([a.forward_return for a in analogs])
            mean_fwd = float(fwd.mean())
            hit = float((fwd > 0).mean())
            # blend analog outcome (scaled by a typical 5-day move ~3%) into the evidence
            sig = float(np.tanh(0.7 * np.arctanh(np.clip(base, -0.99, 0.99)) + 0.3 * np.tanh(mean_fwd / 0.03)))
        summary = self._summarise(row, sig, mean_fwd, hit, len(analogs))
        return MarketEvidence(
            date=str(row["date"]),
            features=zdict,
            market_signal=sig,
            analog_mean_forward_return=mean_fwd,
            analog_hit_rate=hit,
            analog_dates=[a.end_date for a in analogs],
            summary=summary,
        )

    def _summarise(self, row: pd.Series, sig: float, mean_fwd, hit, n_analogs: int) -> str:
        def f(k: str) -> str:
            v = row.get(k, np.nan)
            return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"

        parts = [
            f"5d momentum {f('mom_5')}, 20d trend divergence {f('trend_div')}, 5d vol {f('vol_5')}, 20d drawdown {f('dd_20')}, anomaly z {f('anomaly')}",
        ]
        if self.micro:
            parts.append(f"micro: OFI {f('ofi_proxy')}, spread proxy {f('spread_proxy')}, volume shock {f('volume_shock')}, GK realised var {f('rv_gk')}")
        if n_analogs:
            parts.append(f"{n_analogs} historical analogs: mean {5}-day forward return {mean_fwd:+.2%}, hit rate {hit:.0%}")
        parts.append(f"market evidence signal {sig:+.2f} (bullish>0)")
        return "; ".join(parts)
