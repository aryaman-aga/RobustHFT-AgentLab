"""LLM plumbing: reader/decider modes, the cache, and the local Ollama client (no network).

The Ollama tests talk to a fake server on 127.0.0.1, so they run offline and fast.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rhal.agent.llm_reader import READER_SCHEMA, LLMTextScorer
from rhal.agent.sentiment import score_sentiment
from rhal.config import LLMCfg, _llm_from_env
from rhal.evaluation.experiment import RunSpec, Simulation
from rhal.llm import MockLLM, OllamaClient, build_client
from rhal.llm.base import LLMClient


# ---------------------------------------------------------------------------
# reader mode
# ---------------------------------------------------------------------------
def test_reader_mode_with_mock_client_reproduces_the_lexicon_bit_for_bit(small_cfg, small_dataset, detector):
    """Swapping the lexicon for an LLM reader changes nothing else: with a reader that returns
    the lexicon score, every metric is identical to the lexicon run."""
    spec = RunSpec("injected__grounded_risk", "misgen", grounded=True, risk=True)
    lex = Simulation(small_cfg, small_dataset, detector, 42).run(spec)
    cfg = small_cfg.with_overrides({"agent.text_scorer": "llm", "llm.provider": "mock", "llm.cache_dir": ""})
    sim = Simulation(cfg, small_dataset, detector, 42)
    rd = sim.run(spec)
    assert rd.metrics == lex.metrics
    assert [s["action"] for s in rd.treated_steps] == [s["action"] for s in lex.treated_steps]
    assert sim.llm_stats()["texts_scored"] > 0 and sim.llm_stats()["parse_failures"] == 0


def test_reader_scorer_counts_unreadable_answers_as_neutral():
    class Garbage(LLMClient):
        provider = "fake"

        def _complete(self, system, user, max_tokens, schema):
            return "I think it is bullish"

    sc = LLMTextScorer(Garbage("x"))
    assert sc("Nordwind Systems shares rose 3%.") == 0.0
    assert sc.stats()["parse_failures"] == 1


def test_reader_score_is_clipped():
    class Wild(LLMClient):
        provider = "fake"

        def _complete(self, system, user, max_tokens, schema):
            return json.dumps({"score": 7.5})

    assert LLMTextScorer(Wild("x"))("any text") == 1.0


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_cache_key_covers_options_schema_and_budget(tmp_path):
    c = MockLLM("m", str(tmp_path))
    a = c.complete("sys", "user text", max_tokens=10, schema=READER_SCHEMA)
    b = c.complete("sys", "user text", max_tokens=10, schema=READER_SCHEMA)
    assert not a.cached and b.cached and (c.hits, c.misses) == (1, 1)
    c.complete("sys", "user text", max_tokens=11, schema=READER_SCHEMA)  # budget differs -> new entry
    c.complete("sys", "user text", max_tokens=10, schema=None)  # schema differs -> new entry
    c.options = {"temperature": 0.5}
    c.complete("sys", "user text", max_tokens=10, schema=READER_SCHEMA)  # options differ -> new entry
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 4 and not list(tmp_path.glob("*.tmp*"))
    entry = json.loads(files[0].read_text())
    assert {"provider", "model", "options", "schema", "max_tokens", "system", "user", "text"} <= set(entry)


# ---------------------------------------------------------------------------
# loud failures
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_instead_of_falling_back_to_mock():
    with pytest.raises(ValueError, match="API key"):
        build_client(LLMCfg(provider="openai", model="gpt-x"))


def test_unknown_provider_in_env_raises(monkeypatch):
    monkeypatch.setenv("RHAL_LLM_PROVIDER", "olama")
    with pytest.raises(ValueError, match="not one of"):
        _llm_from_env()


def test_ollama_provider_from_env(monkeypatch):
    monkeypatch.setenv("RHAL_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("RHAL_LLM_MODEL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11999")
    cfg = _llm_from_env()
    assert cfg.provider == "ollama" and cfg.base_url == "http://127.0.0.1:11999" and cfg.model == "qwen2.5:7b"


# ---------------------------------------------------------------------------
# Ollama client against a fake local server
# ---------------------------------------------------------------------------
class _FakeOllama(BaseHTTPRequestHandler):
    requests: list = []
    reply = json.dumps({"score": 0.25})

    def log_message(self, *a):  # keep pytest output clean
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "tiny:latest", "digest": "sha256:abc"}, {"name": "qwen2.5:7b", "digest": "sha256:def"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        self._send({"model": body["model"], "message": {"role": "assistant", "content": type(self).reply}, "done": True})


@pytest.fixture()
def fake_ollama():
    _FakeOllama.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_ollama_client_sends_schema_and_greedy_options(fake_ollama, tmp_path):
    c = OllamaClient("tiny", str(tmp_path), base_url=fake_ollama)
    assert c.digest == "sha256:abc"
    sc = LLMTextScorer(c)
    assert sc("Nordwind Systems raised guidance.") == 0.25
    req = _FakeOllama.requests[-1]
    assert req["format"] == READER_SCHEMA and req["stream"] is False
    assert req["options"]["temperature"] == 0.0 and req["options"]["seed"] == 0 and req["options"]["num_predict"] == sc.max_tokens
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    # the same article again is served from the cache: no second request
    n = len(_FakeOllama.requests)
    assert LLMTextScorer(c)("Nordwind Systems raised guidance.") == 0.25 and len(_FakeOllama.requests) == n


def test_ollama_unknown_model_fails_loudly(fake_ollama):
    with pytest.raises(RuntimeError, match="make llm-pull"):
        OllamaClient("not-pulled:1b", base_url=fake_ollama)


def test_ollama_refuses_remote_hosts():
    with pytest.raises(ValueError, match="this machine"):
        OllamaClient("tiny", base_url="http://example.com:11434", verify=False)


def test_ollama_server_down_raises_connection_error(tmp_path):
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens on this port now
    with pytest.raises(ConnectionError, match="make llm-up"):
        OllamaClient("tiny", str(tmp_path), base_url=f"http://127.0.0.1:{port}", timeout=2)


# ---------------------------------------------------------------------------
# decider mode
# ---------------------------------------------------------------------------
class _Recorder(LLMClient):
    """Fake decider: fixed answer, records every prompt."""

    provider = "fake"

    def __init__(self, reply: str):
        super().__init__("fake")
        self.reply = reply
        self.prompts: list[str] = []

    def _complete(self, system, user, max_tokens, schema):
        self.prompts.append(user)
        return self.reply


def _decider(small_cfg, small_dataset, detector, reply):
    sim = Simulation(small_cfg.with_overrides({"agent.mode": "llm"}), small_dataset, detector, 42)
    sim._llm = _Recorder(reply)  # inject the fake client (build_client is tested above)
    return sim


def test_decider_memory_shows_the_llms_own_stance_not_lexicon_numbers(small_cfg, small_dataset, detector):
    reply = json.dumps({"action": "Buy", "confidence": 80, "news_stance": 12})
    sim = _decider(small_cfg, small_dataset, detector, reply)
    r = sim.run(RunSpec("injected__none", "misgen"))
    texts = [float(x) for p in sim._llm.prompts for x in re.findall(r"text ([+-]\d\.\d\d)", p)]
    assert texts and all(abs(x - 0.12) < 1e-9 for x in texts)  # only the LLM's own stance, never a lexicon score
    assert all(s["text_signal"] == 0.12 and s["confidence"] == 0.8 and not s["parse_failed"] for s in r.treated_steps)
    assert r.metrics["parse_fail_rate"] == 0.0


def test_decider_unreadable_answer_becomes_flagged_hold(small_cfg, small_dataset, detector):
    sim = _decider(small_cfg, small_dataset, detector, "Buy, definitely.")
    r = sim.run(RunSpec("clean__none", "none"))
    assert all(s["action"] == "Hold" and s["parse_failed"] for s in r.treated_steps)
    assert r.metrics["parse_fail_rate"] == 1.0


def test_decider_connection_error_is_not_swallowed(small_cfg, small_dataset, detector):
    class Down(LLMClient):
        provider = "fake"

        def _complete(self, system, user, max_tokens, schema):
            raise ConnectionError("server down")

    sim = Simulation(small_cfg.with_overrides({"agent.mode": "llm"}), small_dataset, detector, 42)
    sim._llm = Down("fake")
    with pytest.raises(ConnectionError):
        sim.run(RunSpec("clean__none", "none"))
