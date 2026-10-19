"""Lexicon sentiment scorer used by the surrogate (mock-LLM) policy.

Handles multi-word phrases, negation within a 3-token window, intensifiers
and hedges, and explicit 'up X%' / 'down X%' numeric cues. Output in [-1, 1].
This deliberately mimics the *shape* of an LLM's reaction to financial text
(direction words dominate, numbers and tone modulate) while being transparent.
"""

from __future__ import annotations

import re

import numpy as np

from .. import lexicon as L

_UPDOWN = re.compile(
    r"\b(up|down|rose|fell|gained|dropped|surged|plunged|jumped|tumbled|climbed|declined|soared|slumped|rallied|collapsed)\s+(\d+(?:\.\d+)?)%"
)
# longest first; ties broken alphabetically so the order (and float summation) never depends on PYTHONHASHSEED
_PHRASES = sorted(set(L.POSITIVE) | set(L.NEGATIVE), key=lambda p: (-len(p), p))
_WEIGHT = {**L.POSITIVE, **L.NEGATIVE}


def score_sentiment(text: str) -> float:
    t = text.lower()
    total = 0.0
    n_hits = 0
    # phrase matching with negation / modifier context
    for ph in _PHRASES:
        for m in re.finditer(rf"\b{re.escape(ph)}\b", t):
            w = _WEIGHT[ph]
            # context window: 3 tokens before
            pre = t[: m.start()].split()[-3:]
            if any(neg in " ".join(pre) for neg in L.NEGATORS):
                w = -w
            if any(i in pre for i in L.INTENSIFIERS):
                w *= 1.35
            if any(h in pre for h in L.HEDGES):
                w *= 0.7
            total += w
            n_hits += 1
    # numeric direction cues: 'up 4.2%' etc.
    for m in _UPDOWN.finditer(t):
        word, pct = m.group(1), float(m.group(2))
        sign = 1 if word in ("up", "rose", "gained", "surged", "jumped", "climbed", "soared", "rallied") else -1
        total += sign * min(pct / 10.0, 0.6)
        n_hits += 1
    if n_hits == 0:
        return 0.0
    # normalise: average polarity, then squash. Keep magnitude sensitive to intensity.
    avg = total / np.sqrt(n_hits)
    return float(np.tanh(avg))
