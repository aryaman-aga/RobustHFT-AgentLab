"""Lexicon sentiment scorer used by the surrogate (mock-LLM) policy.

Handles multi-word phrases, negation within a 3-token window, intensifiers
and hedges, and explicit 'up X%' / 'down X%' numeric cues. Output in [-1, 1].
This deliberately mimics the *shape* of an LLM's reaction to financial text
(direction words dominate, numbers and tone modulate) while being transparent.

``sentiment_hits`` lists every cue the scorer counts, in counting order, so a reader
(e.g. the interactive demo) can see why a headline got its score; ``score_sentiment``
aggregates exactly those cues.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

from .. import lexicon as L

_UPDOWN = re.compile(
    r"\b(up|down|rose|fell|gained|dropped|surged|plunged|jumped|tumbled|climbed|declined|soared|slumped|rallied|collapsed)\s+(\d+(?:\.\d+)?)%"
)
# longest first; ties broken alphabetically so the order (and float summation) never depends on PYTHONHASHSEED
_PHRASES = sorted(set(L.POSITIVE) | set(L.NEGATIVE), key=lambda p: (-len(p), p))
_WEIGHT = {**L.POSITIVE, **L.NEGATIVE}


class Hit(NamedTuple):
    """One counted cue: the matched text, its signed weight after modifiers, and which modifiers applied."""

    phrase: str
    weight: float
    notes: tuple[str, ...]


def sentiment_hits(text: str) -> list[Hit]:
    """Every cue ``score_sentiment`` counts for ``text``, in the order it sums them."""
    t = text.lower()
    hits: list[Hit] = []
    # phrase matching with negation / modifier context
    for ph in _PHRASES:
        for m in re.finditer(rf"\b{re.escape(ph)}\b", t):
            w = _WEIGHT[ph]
            notes: list[str] = []
            # context window: 3 tokens before
            pre = t[: m.start()].split()[-3:]
            if any(neg in " ".join(pre) for neg in L.NEGATORS):
                w = -w
                notes.append("negated")
            if any(i in pre for i in L.INTENSIFIERS):
                w *= 1.35
                notes.append("intensified")
            if any(h in pre for h in L.HEDGES):
                w *= 0.7
                notes.append("hedged")
            hits.append(Hit(ph, w, tuple(notes)))
    # numeric direction cues: 'up 4.2%' etc.
    for m in _UPDOWN.finditer(t):
        word, pct = m.group(1), float(m.group(2))
        sign = 1 if word in ("up", "rose", "gained", "surged", "jumped", "climbed", "soared", "rallied") else -1
        hits.append(Hit(m.group(0), sign * min(pct / 10.0, 0.6), ("number",)))
    return hits


def score_sentiment(text: str) -> float:
    hits = sentiment_hits(text)
    if not hits:
        return 0.0
    total = 0.0
    for h in hits:
        total += h.weight
    # normalise: average polarity, then squash. Keep magnitude sensitive to intensity.
    avg = total / np.sqrt(len(hits))
    return float(np.tanh(avg))
