"""Matched placebo controls (count- and bias-matched to an attack tape) and the one-step counterfactual ASR."""
import pytest

from rhal.evaluation.experiment import RunSpec, Simulation
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
        assert [s["visible_synthetic_frac"] for s in r.treated_steps] == [s["visible_synthetic_frac"] for s in frozen.treated_steps]
        assert all(p in ("clean", "placebo") for s in r.treated_steps for p in s["retrieved_provenance"])
    assert all(s["bias"] is None for s in matched.treated_steps)
    assert [s["bias"] for s in biased.treated_steps] == [tape.bias[s["date"]] for s in biased.treated_steps]


def test_matched_paraphrases_are_a_subset_of_the_classic_placebo(sim, source):
    pool = NewsPool.from_frame(sim.ds.news)
    n_items = lambda d: sum(1 for it in pool.items() if it.date == d)  # noqa: E731
    dated = [d for d, docs in source.tape.docs.items() if docs]
    date = next((d for d in dated if len(source.tape.docs[d]) < n_items(d)), dated[0])  # prefer a day with rejected candidates
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


def test_counterfactual_equals_treated_on_day_one_and_is_zero_for_clean(sim, source):
    first = source.treated_steps[0]
    assert first["cf_action"] == first["action"]  # day one: same memory, same portfolio, same text
    clean = sim.run(RunSpec("clean__none", "none"))
    assert clean.metrics["ASR_cf"] == 0.0 and clean.metrics["ASR"] == 0.0


def test_counterfactual_does_not_change_any_other_result(sim, source, small_cfg, small_dataset, detector):
    off = Simulation(small_cfg.with_overrides({"experiment.counterfactual_asr": False}), small_dataset, detector, 42)
    r_off = off.run(RunSpec("injected__none", "misgen"))
    assert "ASR_cf" not in r_off.metrics and all(s["cf_action"] is None for s in r_off.treated_steps)
    assert _cf_keys(source.metrics) == r_off.metrics  # every other metric is bit-identical
    assert [s["action"] for s in source.treated_steps] == [s["action"] for s in r_off.treated_steps]


def test_new_arms_are_deterministic(sim, source):
    a = sim.run(RunSpec("placebo_bias__grounded", "placebo_bias", True, False, True), tape=source.tape)
    b = sim.run(RunSpec("placebo_bias__grounded", "placebo_bias", True, False, True), tape=source.tape)
    assert a.metrics == b.metrics
