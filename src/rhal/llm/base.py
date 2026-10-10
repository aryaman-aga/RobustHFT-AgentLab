"""Provider-agnostic LLM interface with a disk cache.

The cache makes API-mode runs reproducible (same prompt -> same completion)
and makes re-runs free. Keys are hashes of (provider, model, system, user).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..config import LLMCfg
from ..seeding import stable_hash


@dataclass
class LLMResponse:
    text: str
    cached: bool = False
    provider: str = "mock"


class LLMClient(ABC):
    provider: str = "abstract"

    def __init__(self, model: str, cache_dir: str | None = None):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def complete(self, system: str, user: str, max_tokens: int = 400) -> LLMResponse:
        key = stable_hash(self.provider, self.model, system, user)
        if self.cache_dir:
            f = self.cache_dir / f"{key}.json"
            if f.exists():
                return LLMResponse(json.loads(f.read_text())["text"], cached=True, provider=self.provider)
        text = self._complete(system, user, max_tokens)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps({"system": system, "user": user, "text": text}))
        return LLMResponse(text, cached=False, provider=self.provider)

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int) -> str: ...


def build_client(cfg: LLMCfg) -> LLMClient:
    """Return a client for the configured provider; falls back to mock when keys are missing."""
    from .mock import MockLLM

    if cfg.provider == "anthropic" and cfg.api_key:
        from .api import AnthropicClient

        return AnthropicClient(cfg.model, cfg.cache_dir, cfg.api_key)
    if cfg.provider == "openai" and cfg.api_key:
        from .api import OpenAICompatibleClient

        return OpenAICompatibleClient(cfg.model, cfg.cache_dir, cfg.api_key, cfg.base_url)
    return MockLLM(cfg.model, cfg.cache_dir)
