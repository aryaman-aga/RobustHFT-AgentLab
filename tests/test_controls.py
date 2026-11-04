"""Matched placebo controls (count- and bias-matched to an attack tape) and the one-step counterfactual ASR."""

import numpy as np
import pandas as pd
import pytest

from rhal.evaluation.experiment import TAPE_INJECTIONS, Dataset, RunSpec, Simulation
from rhal.evaluation.metrics import attack_metrics
from rhal.features import build_feature_frame
from rhal.memory import NewsPool


@pytest.fixture(scope="module")
def sim(small_cfg, small_dataset, detector):
    return Simulation(small_cfg, small_dataset, detector, 42)


@pytest.fixture(scope="module")
def source(sim):
    return sim.run(RunSpec("injected__none", "misgen"))


def _cf_keys(m):
    return {k: v for k, v in m.items() if not k.startswith("ASR_cf")}


def test_matched_placebo_matches_attack_count_and_bias(sim, source):
    tape = source.tape
    frozen = sim.run(RunSpec("frozen__none", "replay"), tape=tape)
    matched = sim.run(RunSpec("placebo_matched__none", "placebo_matched"), tape=tape)
    biased = sim.run(RunSpec("placebo_bias__none", "placebo_bias"), tape=tape)
    # same number of injected docs every day -> identical synthetic share of the visible pool
    for r in (matched, biased):
        assert [s["visible_synthetic_frac"] for s in r.treated_steps] == [
            s["visible_synthetic_frac"] for s in frozen.treated_steps
        ]
        assert all(p in ("clean", "placebo") for s in r.treated_steps for p in s["retrieved_provenance"])
    assert all(s["bias"] is None for s in matched.treated_steps)
    assert [s["bias"] for s in biased.treated_steps] == [tape.bias[s["date"]] for s in biased.treated_steps]


def test_matched_paraphrases_are_a_subset_of_the_classic_placebo(sim, source):
    pool = NewsPool.from_frame(sim.ds.news)
    n_items = lambda d: sum(1 for it in pool.items() if it.date == d)  # noqa: E731
    dated = [d for d, docs in source.tape.docs.items() if docs]
    date = next(
        (d for d in dated if len(source.tape.docs[d]) < n_items(d)), dated[0]
    )  # prefer a day with rejected candidates
    items = [it for it in pool.items() if it.date == date]
    parents = [pool.get(d.parent_id) for d in source.tape.docs[date]]
    classic = sim._misgen(sim.cfg, RunSpec("p", "placebo")).generate_for_items(items, date)
    matched = sim._misgen(sim.cfg, RunSpec("m", "placebo_matched")).generate_for_items(parents, date)
    # one paraphrase stream per (day, item): the matched pool is the classic pool restricted to the attacked items
    assert len(matched) == len(parents) and {d.text for d in matched} <= {d.text for d in classic}


def test_tape_required_for_matched_placebos(sim):
    for inj in ("placebo_matched", "placebo_bias"):
        with pytest.raises(ValueError):
            sim.run(RunSpec(inj, inj))


@pytest.mark.parametrize(
    "injection, grounded, risk",
    [
        ("misgen", False, False),
        ("misgen", True, True),
        ("replay", True, False),
        ("placebo", True, True),
        ("placebo_matched", False, False),
        ("placebo_bias", True, True),
    ],
)
def test_counterfactual_asr_equals_asr_on_day_one(sim, source, injection, grounded, risk):
    """Day one: the throw-away agent holds what the treated agent holds (empty memory, initial cash) and reads the same pool and bias."""
    r = sim.run(
        RunSpec("x", injection, grounded, risk, grounded), tape=source.tape if injection in TAPE_INJECTIONS else None
    )
    assert r.treated_steps[0]["cf_action"] == r.treated_steps[0]["action"]
    day_one = attack_metrics(r.treated_steps[:1], r.shadow_steps[:1])
    assert day_one["ASR_cf"] == day_one["ASR"]


def test_counterfactual_is_zero_for_clean(sim):
    clean = sim.run(RunSpec("clean__none", "none"))
    assert clean.metrics["ASR_cf"] == 0.0 and clean.metrics["ASR"] == 0.0


def _mutate_future(ds: Dataset, cfg, cut: int) -> Dataset:
    """Copy of ``ds`` whose prices and volumes change from row ``cut`` on (news untouched); features rebuilt as in prepare_dataset."""
    prices = ds.prices.copy()
    cols = ["open", "high", "low", "close"]
    ramp = pd.Series(np.linspace(0.6, 1.8, len(prices) - cut), index=prices.index[cut:])
    prices.loc[cut:, cols] = prices.loc[cut:, cols].mul(ramp, axis=0)
    prices.loc[cut:, "volume"] *= 50.0
    f = cfg.features
    feats = build_feature_frame(
        prices,
        ds.intraday,
        vol_window=f.vol_window,
        drawdown_window=f.drawdown_window,
        short_trend=f.short_trend,
        long_trend=f.long_trend,
        anomaly_window=f.anomaly_window,
        micro=True,
        rv_window=f.micro.rv_window,
        volume_window=f.micro.volume_window,
    )
    feats["close"] = prices["close"].to_numpy()
    return Dataset(prices, ds.news, ds.intraday, feats, ds.dates)


def test_counterfactual_does_not_depend_on_future(sim, small_cfg, detector):
    """Mutate every price row from the middle of the evaluation window on: no decision before it may move."""
    cut = (sim.t0 + sim.t1) // 2
    mutated = Simulation(small_cfg, _mutate_future(sim.ds, small_cfg, cut), detector, 42)
    spec = RunSpec("g", "misgen", True, True, True)
    a, b = sim.run(spec), mutated.run(spec)
    k = cut - sim.t0  # steps 0..k-1 are decided at t < cut
    for key in ("action", "cf_action", "raw_signal"):
        assert [s[key] for s in a.treated_steps[:k]] == [s[key] for s in b.treated_steps[:k]]
    assert [s["action"] for s in a.shadow_steps[:k]] == [s["action"] for s in b.shadow_steps[:k]]
    assert [s["raw_signal"] for s in a.treated_steps[k:]] != [
        s["raw_signal"] for s in b.treated_steps[k:]
    ]  # the mutation was seen


def test_counterfactual_does_not_change_any_other_result(sim, source, small_cfg, small_dataset, detector):
    off = Simulation(small_cfg.with_overrides({"experiment.counterfactual_asr": False}), small_dataset, detector, 42)
    r_off = off.run(RunSpec("injected__none", "misgen"))
    assert "ASR_cf" not in r_off.metrics and all(s["cf_action"] is None for s in r_off.treated_steps)
    assert _cf_keys(source.metrics) == r_off.metrics  # every other metric is bit-identical
    assert [s["action"] for s in source.treated_steps] == [s["action"] for s in r_off.treated_steps]


def test_new_arms_are_deterministic(small_cfg, small_dataset, detector):
    """Two fresh simulations, each recording its own tape, give identical matched controls and counterfactual actions."""

    def run_controls():
        s = Simulation(small_cfg, small_dataset, detector, 42)
        tape = s.run(RunSpec("injected__none", "misgen")).tape
        return [
            s.run(RunSpec(inj, inj, True, False, True), tape=tape)
            for inj in ("placebo_matched", "placebo_bias", "replay")
        ]

    for a, b in zip(run_controls(), run_controls()):
        assert a.metrics == b.metrics
        assert [s["cf_action"] for s in a.treated_steps] == [s["cf_action"] for s in b.treated_steps]
