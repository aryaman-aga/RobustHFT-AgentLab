"""Historical analog retrieval with a *strict* no-leak rule.

At decision time t we may only use analog windows whose forward outcome is
fully observed by t. Concretely a candidate window ending at index e with
outcome horizon H is admissible iff  e + H <= t.  This is stricter than the
common (and leaky) practice of only requiring e < t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Analog:
    end_idx: int
    end_date: str
    similarity: float
    forward_return: float


class AnalogRetriever:
    def __init__(self, feats: pd.DataFrame, z_cols: list[str], window: int = 30, horizon: int = 5, top_k: int = 5):
        self.window = window
        self.horizon = horizon
        self.top_k = top_k
        self.dates = feats["date"].tolist()
        Z = feats[z_cols].to_numpy(dtype=float)
        self.Z = np.nan_to_num(Z, nan=0.0)
        self.logc = np.log(feats["close"].to_numpy(dtype=float)) if "close" in feats.columns else None
        self.n = len(feats)

    def _vec(self, end_idx: int) -> np.ndarray | None:
        start = end_idx - self.window + 1
        if start < 0:
            return None
        return self.Z[start : end_idx + 1].ravel()

    def retrieve(self, t: int) -> list[Analog]:
        """Top-k analogs for the window ending at t (inclusive)."""
        if self.logc is None:
            return []
        q = self._vec(t)
        if q is None:
            return []
        qn = np.linalg.norm(q) + 1e-12
        cands: list[Analog] = []
        last_admissible = t - self.horizon  # e + H <= t
        for e in range(self.window - 1, last_admissible + 1):
            v = self._vec(e)
            if v is None:
                continue
            sim = float(q @ v / (qn * (np.linalg.norm(v) + 1e-12)))
            fwd = float(self.logc[e + self.horizon] - self.logc[e])
            cands.append(Analog(e, self.dates[e], sim, fwd))
        cands.sort(key=lambda a: a.similarity, reverse=True)
        return cands[: self.top_k]
