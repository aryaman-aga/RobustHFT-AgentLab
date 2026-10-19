"""Baseline defenses (text-free agent, detection filter, robust aggregation) and the defense-aware attacker."""
import numpy as np
import pandas as pd
import pytest

from rhal.agent.policy import weighted_median, weighted_trimmed_mean
from rhal.evaluation.experiment import Dataset, RunSpec, Simulation
from test_controls import _mutate_future


@pytest.fixture(scope="module")
def sim(small_cfg, small_dataset, detector):
    return Simulation(small_cfg, small_dataset, detector, 42)


# ---------------------------------------------------------------------------
# B1: text-free agent
# ---------------------------------------------------------------------------
def test_text_free_agent_cannot_be_attacked(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg.with_overrides({"agent.mode": "rule"}), small_dataset, detector, 42)
    r = sim.run(RunSpec("injected__text_free", "misgen", grounded=True, micro=True))
    assert r.metrics["ASR"] == 0.0 and r.metrics["CR_delta_vs_clean"] == 0.0
    assert r.metrics["MER"] > 0  # the fakes are retrieved; the agent just never reads them


# ---------------------------------------------------------------------------
# B2: detection filter
# ---------------------------------------------------------------------------
def test_detection_filter_is_identical_for_both_twins(small_cfg, small_dataset, detector):
    cfg = small_cfg.with_overrides({"defense.detect_filter": True, "defense.detect_fpr": 0.25})
    r = Simulation(cfg, small_dataset, detector, 42).run(RunSpec("injected__detect", "misgen"))
    for s, t in zip(r.shadow_steps, r.treated_steps):
        assert all(p == "clean" for p in s["filter_dropped"])
        # the same clean docs are visible to both twins, so the filter drops the same number of them
        assert s["filter_dropped"].count("clean") == t["filter_dropped"].count("clean")
    assert 0.0 <= r.metrics["filter_recall"] <= 1.0 and 0.0 < r.metrics["filter_clean_drop"] < 1.0


def test_detection_filter_off_changes_nothing(sim):
    r = sim.run(RunSpec("injected__none", "misgen"))
    assert "filter_recall" not in r.metrics and all(not s["filter_visible"] for s in r.treated_steps)


def test_detection_threshold_uses_only_pre_window_news(sim, small_cfg, detector):
    """Rewrite every article in the evaluation window: the calibrated threshold must not move."""
    news = sim.ds.news.copy()
    future = news["date"].astype(str) >= sim.ds.dates[sim.t0]
    news.loc[future, "text"] = "Nordwind Systems said nothing material today, according to people familiar with the matter."
    other = Simulation(small_cfg, Dataset(sim.ds.prices, news, sim.ds.intraday, sim.ds.feats, sim.ds.dates), detector, 42)
    sim.detect_filter(0.2)
    other.detect_filter(0.2)
    assert sim.filter_thresholds[0.2] == other.filter_thresholds[0.2]


# ---------------------------------------------------------------------------
# B3: robust aggregation
# ---------------------------------------------------------------------------
def test_weighted_median_and_trimmed_mean_by_hand():
    x = np.array([0.9, -0.8, 0.1, 0.2])
    w = np.full(4, 0.25)
    assert weighted_median(x, w) == 0.1  # lower median of {-0.8, 0.1, 0.2, 0.9}
    assert weighted_trimmed_mean(x, w, 0.25) == pytest.approx(0.15)  # mean of the middle two
    assert weighted_trimmed_mean(x, w, 0.0) == pytest.approx(x.mean())
    # one heavy outlier moves the mean but not the median
    w2 = np.array([0.1, 0.6, 0.15, 0.15])
    assert weighted_median(x, w2) == -0.8 and weighted_median(np.array([0.9, 0.1, 0.2]), np.array([0.2, 0.4, 0.4])) == 0.2


def test_median_aggregation_runs_and_differs_from_mean(small_cfg, small_dataset, detector):
    spec = RunSpec("injected__none", "misgen")
    mean = Simulation(small_cfg, small_dataset, detector, 42).run(spec)
    med = Simulation(small_cfg.with_overrides({"agent.text_agg": "median"}), small_dataset, detector, 42).run(spec)
    assert [s["text_signal"] for s in med.treated_steps] != [s["text_signal"] for s in mean.treated_steps]


# ---------------------------------------------------------------------------
# A1/A2: defense-aware attacker
# ---------------------------------------------------------------------------
def _aware(small_cfg, ds, detector, mode):
    return Simulation(small_cfg.with_overrides({"misgen.evidence_aware": mode}), ds, detector, 42)


def test_aware_attacker_is_deterministic_and_ignores_arm_names(small_cfg, small_dataset, detector):
    a = _aware(small_cfg, small_dataset, detector, "aligned").run(RunSpec("injected__x", "misgen", True, False, True))
    b = _aware(small_cfg, small_dataset, detector, "aligned").run(RunSpec("injected__renamed", "misgen", True, False, True))
    assert a.metrics == b.metrics


def test_aligned_edits_agree_with_the_market_more_than_conflicting_ones(small_cfg, small_dataset, detector):
    def mean_align(mode):
        r = _aware(small_cfg, small_dataset, detector, mode).run(RunSpec("injected__none", "misgen"))
        vals = [d.strategy["align"] for docs in r.tape.docs.values() for d in docs if d.strategy and "align" in d.strategy]
        assert vals, "the aware attacker recorded no alignment"
        return float(np.mean(vals))

    assert mean_align("aligned") > mean_align("conflict")


def test_aware_attacker_does_not_see_future_prices(sim, small_cfg, detector):
    """Prices after a cut-off change: the attacker's documents for earlier days must not."""
    cut = (sim.t0 + sim.t1) // 2
    mutated = _mutate_future(sim.ds, small_cfg, cut)
    spec = RunSpec("injected__none", "misgen")
    a = _aware(small_cfg, sim.ds, detector, "aligned").run(spec).tape
    b = _aware(small_cfg, mutated, detector, "aligned").run(spec).tape
    early = [d for d in sim.ds.dates[sim.t0:cut]]
    assert [[x.text for x in a.docs[d]] for d in early] == [[x.text for x in b.docs[d]] for d in early]
