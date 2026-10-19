"""Provider-agnostic LLM interface with a disk cache.

The cache makes LLM-mode runs reproducible (same request -> same completion) and makes
re-runs free. The key hashes everything that can change a completion: provider, model,
decoding options, the requested JSON schema, the token budget and both prompts. Changing
any of them is a cache miss, never a silently reused answer.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        self.options: dict[str, Any] = {}  # decoding options; part of the cache key
        self.hits = 0
        self.misses = 0

    def cache_key(self, system: str, user: str, max_tokens: int, schema: dict | None) -> str:
        opts = json.dumps(self.options, sort_keys=True)
        fmt = json.dumps(schema, sort_keys=True) if schema is not None else ""
        return stable_hash(self.provider, self.model, opts, fmt, max_tokens, system, user)

    def complete(self, system: str, user: str, max_tokens: int = 400, schema: dict | None = None) -> LLMResponse:
        """Return the completion for (system, user); ``schema`` asks for JSON matching a JSON schema."""
        key = self.cache_key(system, user, max_tokens, schema)
        f = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if f is not None and f.exists():
            self.hits += 1
            return LLMResponse(json.loads(f.read_text())["text"], cached=True, provider=self.provider)
        text = self._complete(system, user, max_tokens, schema)  # network errors propagate: never cache a failure
        self.misses += 1
        if f is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            entry = {"provider": self.provider, "model": self.model, "options": self.options, "schema": schema,
                     "max_tokens": max_tokens, "system": system, "user": user, "text": text}
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(entry))
            os.replace(tmp, f)  # atomic: an interrupted run never leaves a half-written entry
        return LLMResponse(text, cached=False, provider=self.provider)

    def stats(self) -> dict:
        return {"provider": self.provider, "model": self.model, "cache_hits": self.hits, "cache_misses": self.misses}

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int, schema: dict | None) -> str: ...


def build_client(cfg: LLMCfg) -> LLMClient:
    """Return the client for the configured provider.

    A provider that cannot be used raises instead of quietly falling back to the mock: a
    fallback made every "LLM" decision a lexicon decision without any visible error.
    """
    from .mock import MockLLM

    if cfg.provider == "mock":
        return MockLLM(cfg.model, cfg.cache_dir)
    if cfg.provider == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(cfg.model, cfg.cache_dir, base_url=cfg.base_url, timeout=cfg.timeout)
    if not cfg.api_key:
        raise ValueError(f"llm.provider={cfg.provider!r} needs an API key in .env (see .env.example)")
    if cfg.provider == "anthropic":
        from .api import AnthropicClient

        return AnthropicClient(cfg.model, cfg.cache_dir, cfg.api_key)
    if cfg.provider == "openai":
        from .api import OpenAICompatibleClient

        return OpenAICompatibleClient(cfg.model, cfg.cache_dir, cfg.api_key, cfg.base_url)
    raise ValueError(f"unknown llm.provider {cfg.provider!r}")
