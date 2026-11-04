"""Interactive demo (rhal.demo, scripts/demo_server.py): the presets tell the story their notes claim,
the clean twin never sees the fake headlines, bad requests fail with a readable message, and the server
answers on loopback. The demo reuses the experiments' agent code; these tests guard its wiring."""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from rhal.agent.sentiment import score_sentiment, sentiment_hits
from rhal.config import load_config
from rhal.demo import PRESETS, Demo, DemoError


@pytest.fixture(scope="module")
def demo(detector) -> Demo:
    # The presets were written for the default market (seed 42); the detector only labels headlines.
    return Demo(load_config("configs/default.yaml"), detector=detector)


def _run(demo: Demo, preset: dict, **kw) -> dict:
    req = {k: preset[k] for k in ("scenario", "genuine", "injected", "bias")}
    return demo.analyze({**req, "reader": "lexicon", **kw})


def test_scenarios_match_their_names(demo):
    s = {x["id"]: x for x in demo.scenario_payload()}
    assert s["rising"]["s_mkt"] > 0.3 and s["rising"]["basic_read"] > 0
    assert s["falling"]["s_mkt"] < -0.3 and s["falling"]["basic_read"] < 0
    assert abs(s["flat"]["s_mkt"]) < 0.1
    assert all(x["chart"]["dates"][-1] == x["date"] for x in s.values())  # the chart ends on the decision day


@pytest.mark.parametrize(
    "pid,none_flips,defended_flips",
    [("flip", True, False), ("aligned", True, False), ("limit", True, True)],
)
def test_presets_tell_their_story(demo, pid, none_flips, defended_flips):
    r = _run(demo, next(p for p in PRESETS if p["id"] == pid))
    assert r["pairs"]["none"]["flipped"] is none_flips
    assert r["pairs"]["defended"]["flipped"] is defended_flips


def test_conflict_rule_backfires_and_shrink_fixes_it(demo):
    p = next(p for p in PRESETS if p["id"] == "flip")
    hold, shrink = _run(demo, p, risk="hold"), _run(demo, p, risk="shrink")
    assert hold["pairs"]["defended"]["flipped"] and hold["pairs"]["defended"]["attacked"]["final"] == "Hold"
    assert not shrink["pairs"]["defended"]["flipped"]
    assert any(o.startswith("shrink_on_conflict") for o in shrink["pairs"]["defended"]["attacked"]["overrides"])


def test_clean_twin_never_reads_the_fake(demo):
    p = next(p for p in PRESETS if p["id"] == "flip")
    with_fake, without = _run(demo, p), _run(demo, {**p, "injected": []})
    for key in ("none", "defended"):
        assert with_fake["pairs"][key]["clean"] == without["pairs"][key]["clean"]
        assert not without["pairs"][key]["flipped"]  # nothing injected, nothing flipped


def test_headline_scores_and_words(demo):
    p = next(p for p in PRESETS if p["id"] == "flip")
    r = _run(demo, p)
    gen, fake = r["headlines"]
    assert gen["role"] == "genuine" and gen["score"] > 0.5 and fake["role"] == "injected" and fake["score"] < -0.5
    assert fake["agreement"] < r["params"]["conflict_threshold"] < gen["agreement"]  # the fake contradicts the market
    assert {w["phrase"] for w in fake["words"]} >= {"plunged", "downgraded"}


@pytest.mark.parametrize(
    "req,msg",
    [
        ({"scenario": "nope", "genuine": ["x"]}, "unknown market scenario"),
        ({"scenario": "rising"}, "at least one headline"),
        ({"scenario": "rising", "genuine": ["x"], "reader": "gpt"}, "unknown reader"),
        ({"scenario": "rising", "genuine": ["x"], "bias": "herding"}, "unknown bias"),
        ({"scenario": "rising", "genuine": [1, 2]}, "list of strings"),
        ({"scenario": "rising", "genuine": ["x"] * 7}, "at most"),
        ({"scenario": "rising", "genuine": ["x" * 401]}, "at most 400"),
    ],
)
def test_bad_requests_fail_readably(demo, req, msg):
    with pytest.raises(DemoError, match=msg):
        demo.analyze(req)


def test_sentiment_hits_reproduce_the_score(small_dataset):
    for text in small_dataset.news["text"].tolist()[:300] + ["Shares rose 4.2% but not strongly", ""]:
        hits = sentiment_hits(text)
        total = 0.0
        for h in hits:
            total += h.weight
        expect = float(np.tanh(total / np.sqrt(len(hits)))) if hits else 0.0
        assert score_sentiment(text) == expect  # exact: the demo's word list is what the agent counts


def test_server_answers_on_loopback(demo):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import demo_server

    demo_server.Handler.demo = demo
    server = ThreadingHTTPServer(("127.0.0.1", 0), demo_server.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        setup = json.load(urllib.request.urlopen(base + "/api/setup", timeout=10))
        assert {s["id"] for s in setup["scenarios"]} == {"rising", "falling", "flat"} and setup["presets"]
        body = json.dumps({"scenario": "rising", "genuine": PRESETS[0]["genuine"], "injected": PRESETS[0]["injected"]})
        req = urllib.request.Request(
            base + "/api/analyze", data=body.encode(), headers={"Content-Type": "application/json"}
        )
        assert json.load(urllib.request.urlopen(req, timeout=10))["pairs"]["none"]["flipped"]
        bad = urllib.request.Request(base + "/api/analyze", data=b"[]", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad, timeout=10)
        assert e.value.code == 400
        assert (
            b"Can one fake headline" in urllib.request.urlopen(base + "/", timeout=10).read()
        )  # / serves the demo page
    finally:
        server.shutdown()
        server.server_close()
