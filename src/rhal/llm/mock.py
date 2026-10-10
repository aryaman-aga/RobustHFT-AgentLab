"""Deterministic mock LLM.

For the *agent*, mock mode does not go through this class at all -- the
surrogate policy is used directly. This client exists so that any code path
written against ``LLMClient`` (e.g. LLM-mode MisGen prompts) still runs
offline and returns well-formed JSON derived from a lexicon sentiment score.
"""
from __future__ import annotations

import json

from ..agent.sentiment import score_sentiment
from .base import LLMClient


class MockLLM(LLMClient):
    provider = "mock"

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        s = score_sentiment(user)
        action = "Buy" if s > 0.15 else "Sell" if s < -0.15 else "Hold"
        return json.dumps({"action": action, "confidence": round(min(1.0, abs(s)), 3), "rationale": "mock: lexicon sentiment of provided context"})
