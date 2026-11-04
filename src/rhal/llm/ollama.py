"""Local Ollama client (standard library only).

Talks to an Ollama server on THIS machine; ``make llm-setup`` installs one inside .venv.
Only loopback hosts are accepted, so prompts -- which contain the synthetic misinformation --
never leave the computer (CLAUDE.md: offline only; synthetic text stays synthetic).

Decoding is greedy (temperature 0) with a fixed seed, and answers are constrained to a JSON
schema through Ollama's ``format`` field. The disk cache in ``LLMClient`` makes re-runs exact.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .base import LLMClient

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 4096}
_VERIFIED: dict[tuple[str, str], str] = {}  # (base_url, model) -> model digest; one /api/tags check per process
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        cache_dir: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        options: dict | None = None,
        verify: bool = True,
    ):
        super().__init__(model, cache_dir)
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        host = urlparse(self.base_url).hostname
        if host not in LOCAL_HOSTS:
            raise ValueError(f"Ollama must run on this machine (got host {host!r}); prompts may not leave it")
        self.timeout = timeout
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.digest = self._verify() if verify else ""

    # -- HTTP -----------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with _NO_PROXY.open(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:  # e.g. 404 unknown model, 400 bad request
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Ollama {method} {path} failed: HTTP {e.code} {detail}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"cannot reach Ollama at {self.base_url} ({e.reason}); start it with: make llm-up"
            ) from e

    def _verify(self) -> str:
        """Fail early (and loudly) if the server is down or the model has not been pulled."""
        key = (self.base_url, self.model)
        if key not in _VERIFIED:
            tags = self._request("GET", "/api/tags")
            models = {m.get("name", ""): m.get("digest", "") for m in tags.get("models", [])}
            name = self.model if ":" in self.model else f"{self.model}:latest"
            if name not in models:
                raise RuntimeError(
                    f"model {self.model!r} is not available in Ollama; run: make llm-pull LLM_MODELS={self.model}"
                )
            _VERIFIED[key] = models[name]
        return _VERIFIED[key]

    def _complete(self, system: str, user: str, max_tokens: int, schema: dict | None) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {**self.options, "num_predict": max_tokens},
        }
        if schema is not None:
            body["format"] = schema  # structured output: the reply is JSON matching the schema
        out = self._request("POST", "/api/chat", body)
        return str(out.get("message", {}).get("content", ""))

    def stats(self) -> dict:
        return {**super().stats(), "base_url": self.base_url, "digest": self.digest, "options": self.options}
