"""Deterministic mock LLM.

For the *agent*, mock mode does not go through this class at all -- the surrogate policy is
used directly. This client exists so that every code path written against ``LLMClient``
(reader mode, LLM-decider mode, their tests) runs offline and returns well-formed JSON
derived from the lexicon sentiment score:

* reader requests (schema with ``score``) get exactly ``score_sentiment(article)``, so reader
  mode with this client reproduces the lexicon results bit for bit (tests/test_llm.py);
* decider requests get an action, a confidence (0..100) and a ``news_stance`` (-100..100) from
  the whole prompt.
"""

from __future__ import annotations

import json

from ..agent.sentiment import score_sentiment
from .base import LLMClient

ARTICLE_MARKER = "\nArticle:\n"  # reader prompts end with this marker followed by the article text


class MockLLM(LLMClient):
    provider = "mock"

    def _complete(self, system: str, user: str, max_tokens: int, schema: dict | None = None) -> str:
        props = (schema or {}).get("properties", {})
        if "score" in props:
            article = user.split(ARTICLE_MARKER, 1)[1] if ARTICLE_MARKER in user else user
            return json.dumps({"score": score_sentiment(article)})
        s = score_sentiment(user)
        action = "Buy" if s > 0.15 else "Sell" if s < -0.15 else "Hold"
        return json.dumps(
            {"action": action, "confidence": int(round(100 * min(1.0, abs(s)))), "news_stance": int(round(100 * s))}
        )
