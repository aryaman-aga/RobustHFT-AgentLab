import numpy as np

from rhal.agent.sentiment import score_sentiment
from rhal.misgen import BIAS_TYPES, MINOR_TYPES, REWRITE_TYPES, MisGenEngine, apply_minor, apply_placebo, apply_rewrite, bias_config
from rhal.misgen.detector import DetectabilityModel, detect_features, FEATURE_NAMES
from rhal.misgen.filters import SimilarityEvaluator
from rhal.schemas import NewsItem
from rhal.seeding import rng

SRC = "Nordwind Systems (NWS) reported Q2 2024 revenue of $2.4B, up 8.5% year over year, driven by stronger-than-expected enterprise demand. The result beat consensus estimates and analysts described the quarter as strong. Shares surged 4.1% in early trading."


def test_every_minor_operator_changes_text():
    for m in MINOR_TYPES:
        out = apply_minor(SRC, m, rng(1, m))
        assert out != SRC, m


def test_flipping_reverses_sentiment_sign():
    flipped = apply_minor(SRC, "flipping", rng(2))
    assert score_sentiment(SRC) > 0.2 and score_sentiment(flipped) < -0.2


def test_numerical_keeps_direction_and_changes_numbers():
    out = apply_minor(SRC, "numerical", rng(3))
    assert "up" in out and "8.5%" not in out and "$2.4B" not in out
    assert np.sign(score_sentiment(out)) == np.sign(score_sentiment(SRC))


def test_entity_and_temporal_and_concept():
    assert "Nordwind Systems" not in apply_minor(SRC, "entity", rng(4))
    assert "Q2 2024" not in apply_minor(SRC, "temporal", rng(5))
    assert "net income" in apply_minor(SRC, "concept", rng(6))


def test_placebo_preserves_meaning():
    out = apply_placebo(SRC, rng(7))
    assert out != SRC
    assert abs(score_sentiment(out) - score_sentiment(SRC)) < 0.15
    assert "8.5%" in out and "$2.4B" in out and "Q2 2024" in out


def test_rewrite_keeps_perturbation_and_length_neutral():
    flipped = apply_minor(SRC, "flipping", rng(8))
    for r in REWRITE_TYPES:
        out = apply_rewrite(flipped, r, rng(9, r))
        assert score_sentiment(out) < 0, r  # does not undo the flip
        assert len(out.split()) <= len(flipped.split()) + 6, r  # no long appended clauses


def test_bias_configs():
    assert bias_config(None).name is None
    for b in BIAS_TYPES:
        c = bias_config(b)
        assert c.name == b and c.prompt
    assert bias_config("overconfidence").threshold_scale < 1
    assert bias_config("loss_aversion").downside_weight > 1


def test_detector_features_and_training(synth):
    x = detect_features(SRC)
    assert x.shape == (len(FEATURE_NAMES),)
    texts = synth["news"]["text"].tolist()
    pert = [apply_minor(t, "flipping", rng(10, i)) for i, t in enumerate(texts[:100])] + [apply_minor(t, "causal", rng(11, i)) for i, t in enumerate(texts[:100])]
    det = DetectabilityModel().fit(texts, pert, target_fpr=0.1, seed=0)
    assert 0 < det.threshold < 1
    assert abs(det.meta["holdout_clean_fpr_at_threshold"] - 0.1) < 0.06
    assert 0 <= det.score(SRC) <= 1


def test_similarity_evaluator(synth):
    sim = SimilarityEvaluator(synth["news"]["text"].tolist())
    assert sim(SRC, SRC) > 0.999
    assert sim(SRC, apply_placebo(SRC, rng(1))) > sim(SRC, "completely unrelated text about gardening tools")


def _engine(det, texts, **kw):
    return MisGenEngine(seed=1, background_corpus=texts, detector=det, **kw)


def test_engine_labels_and_determinism(synth, detector):
    texts = synth["news"]["text"].tolist()
    items = [NewsItem(id=f"n{i}", date="2024-01-02", text=t) for i, t in enumerate(texts[:6])]
    a = _engine(detector, texts).generate_for_items(items, "2024-01-02")
    b = _engine(detector, texts).generate_for_items(items, "2024-01-02")
    assert [x.text for x in a] == [x.text for x in b]
    assert all(x.provenance == "synthetic" and x.parent_id and x.strategy and x.safety_label for x in a)
    assert all(x.strategy["minor"] in MINOR_TYPES for x in a)


def test_engine_placebo_and_no_filter(synth, detector):
    texts = synth["news"]["text"].tolist()
    items = [NewsItem(id=f"n{i}", date="2024-01-02", text=t) for i, t in enumerate(texts[:5])]
    pl = _engine(detector, texts, placebo=True).generate_for_items(items, "2024-01-02")
    assert len(pl) == 5 and all(x.provenance == "placebo" for x in pl)
    nf = _engine(detector, texts, use_filter=False, max_tries=1).generate_for_items(items, "2024-01-02")
    assert len(nf) == 5  # everything accepted on first try


def test_engine_feedback_reinforces(synth, detector):
    texts = synth["news"]["text"].tolist()
    eng = _engine(detector, texts, bias_prob=1.0)
    eng.sample_bias(cr=1.1, date="2024-01-02")
    items = [NewsItem(id="n0", date="2024-01-02", text=texts[0])]
    syn = eng.generate_for_items(items, "2024-01-02")
    before = dict(eng.s_minor.counts)
    eng.update_history(True, syn)
    assert sum(eng.s_minor.counts.values()) == sum(before.values()) + len(syn)
    assert sum(eng.history.bias.values()) == 1
    eng.update_history(False, syn)  # no change when decision unchanged
    assert sum(eng.history.bias.values()) == 1


def test_injection_ratio(synth, detector):
    texts = synth["news"]["text"].tolist()
    items = [NewsItem(id=f"n{i}", date="2024-01-02", text=t) for i, t in enumerate(texts[:10])]
    out = _engine(detector, texts, injection_ratio=0.5, use_filter=False).generate_for_items(items, "2024-01-02")
    assert len(out) == 5
    assert _engine(detector, texts, injection_ratio=0.0).generate_for_items(items, "2024-01-02") == []
