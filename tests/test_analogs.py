import numpy as np

from rhal.features import build_feature_frame
from rhal.features.windows import feature_columns
from rhal.grounding.analogs import AnalogRetriever


def _retriever(synth, window=20, horizon=5, k=4):
    p = synth["prices"]
    f = build_feature_frame(p, synth["intraday"], micro=True)
    f["close"] = p["close"].to_numpy()
    return AnalogRetriever(f, ["z_" + c for c in feature_columns(True)], window=window, horizon=horizon, top_k=k)


def test_admissibility_rule(synth):
    r = _retriever(synth)
    for t in (60, 90, 140):
        for a in r.retrieve(t):
            assert a.end_idx + r.horizon <= t, "analog outcome must be fully observed by t"
            assert a.end_idx >= r.window - 1


def test_sorted_and_bounded(synth):
    r = _retriever(synth, k=3)
    an = r.retrieve(120)
    assert len(an) == 3
    sims = [a.similarity for a in an]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in sims)


def test_insufficient_history_returns_empty(synth):
    r = _retriever(synth, window=30, horizon=5)
    assert r.retrieve(10) == []
    assert r.retrieve(29) == []  # window fits but no admissible analog (needs e+H <= t with e >= 29)
    assert len(r.retrieve(34)) == 1


def test_forward_return_is_from_the_past(synth):
    r = _retriever(synth)
    t = 100
    logc = np.log(synth["prices"]["close"].to_numpy())
    for a in r.retrieve(t):
        assert abs(a.forward_return - (logc[a.end_idx + 5] - logc[a.end_idx])) < 1e-12
