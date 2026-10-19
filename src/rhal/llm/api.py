"""Optional API-backed clients (imported lazily; SDKs are optional extras).

They ignore ``schema`` and rely on the prompt asking for JSON; the parsers tolerate code fences.
"""

from __future__ import annotations

from .base import LLMClient


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, cache_dir: str | None, api_key: str):
        super().__init__(model, cache_dir)
        try:
            import anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key)

    def _complete(
        self, system: str, user: str, max_tokens: int, schema: dict | None = None
    ) -> str:  # pragma: no cover - network
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content)


class OpenAICompatibleClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, cache_dir: str | None, api_key: str, base_url: str | None = None):
        super().__init__(model, cache_dir)
        try:
            import openai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai") from e
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _complete(
        self, system: str, user: str, max_tokens: int, schema: dict | None = None
    ) -> str:  # pragma: no cover - network
        r = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""
