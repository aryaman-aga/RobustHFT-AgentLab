"""History-aware strategy sampler (paper §2.2.3, Pólya-urn reinforcement).

    w_s = 1 + alpha * h_s ;  P(s) = w_s / sum w
Return-dependent adjustment: CR > 1 boosts overconfidence by delta, CR < 1
boosts loss aversion by delta, then re-normalise.
"""

from __future__ import annotations

import numpy as np


class StrategySampler:
    def __init__(self, strategies: list[str], alpha: float = 1.0, use_feedback: bool = True):
        self.strategies = list(strategies)
        self.alpha = alpha
        self.use_feedback = use_feedback
        self.counts: dict[str, int] = {s: 0 for s in self.strategies}

    def probabilities(self, cr: float | None = None, delta: float = 0.0) -> np.ndarray:
        if self.use_feedback:
            w = np.array([1.0 + self.alpha * self.counts[s] for s in self.strategies])
        else:
            w = np.ones(len(self.strategies))
        p = w / w.sum()
        if cr is not None and delta > 0:
            if cr > 1.0 and "overconfidence" in self.strategies:
                p[self.strategies.index("overconfidence")] += delta
            elif cr < 1.0 and "loss_aversion" in self.strategies:
                p[self.strategies.index("loss_aversion")] += delta
            p = p / p.sum()
        return p

    def sample(self, g: np.random.Generator, cr: float | None = None, delta: float = 0.0) -> str:
        p = self.probabilities(cr, delta)
        return str(g.choice(self.strategies, p=p))

    def reinforce(self, strategy: str, amount: int = 1) -> None:
        if strategy in self.counts:
            self.counts[strategy] += amount
