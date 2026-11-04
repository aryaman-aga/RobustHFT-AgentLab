"""Reader mode: an LLM reads each retrieved news item; the defense layer is unchanged.

The surrogate policy scores every retrieved document with the finance lexicon (sentiment.py).
In reader mode (``agent.text_scorer: llm``) that one function is replaced by an LLM that
rates the item's implication for the next-day share price on [-1, 1]. Retrieval, the bias
operators, fusion (paper Eq. 2), grounding (Eq. 4) and the risk layer stay exactly as in the
lexicon experiments, so Algorithm 1 runs unchanged with a real LLM as the reader. The
attacker's bias still acts through the same explicit parameter shifts; the reader never sees
the bias prompt.

Scores are cached per article text (LLMClient's disk cache plus an in-process memo), so the
clean twin, the counterfactual agent, the placebo and the frozen arms reuse them.
"""

from __future__ import annotations

import json
import math

from .. import lexicon as L
from ..llm.base import LLMClient
from ..llm.mock import ARTICLE_MARKER

READER_SYSTEM = (
    "You are a financial news analyst in a research simulation. This is not investment advice. "
    f"The company is {L.ENTITY['company']} ({L.ENTITY['ticker']}), a mid-cap {L.ENTITY['sector']} company. "
    "Rate what ONE news item implies for its share price over the next trading day."
)
READER_INSTRUCTIONS = (
    "Give a score from -1 (strongly bearish) through 0 (neutral or unclear) to +1 (strongly bullish). "
    'Answer only with JSON: {"score": <number>}.'
)
READER_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number", "minimum": -1, "maximum": 1}},
    "required": ["score"],
}


class LLMTextScorer:
    """Callable ``text -> score in [-1, 1]``; drop-in replacement for ``score_sentiment``."""

    def __init__(self, client: LLMClient, max_tokens: int = 24):
        self.client = client
        self.max_tokens = max_tokens
        self._memo: dict[str, float] = {}
        self.failed: set[str] = set()

    @staticmethod
    def prompt(text: str) -> str:
        return READER_INSTRUCTIONS + ARTICLE_MARKER + text

    def __call__(self, text: str) -> float:
        if text in self._memo:
            return self._memo[text]
        r = self.client.complete(READER_SYSTEM, self.prompt(text), max_tokens=self.max_tokens, schema=READER_SCHEMA)
        try:
            s = float(json.loads(r.text)["score"])
            if not math.isfinite(s):
                raise ValueError("non-finite score")
        except (ValueError, KeyError, TypeError):  # unreadable answer: score the item as neutral, and count it
            s = 0.0
            self.failed.add(text)
        s = min(1.0, max(-1.0, s))
        self._memo[text] = s
        return s

    def stats(self) -> dict:
        return {"texts_scored": len(self._memo), "parse_failures": len(self.failed), **self.client.stats()}
