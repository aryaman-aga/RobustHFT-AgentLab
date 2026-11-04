"""Price-impact stress test: the frozen attack also moves the shared price path."""

import copy

import numpy as np
import pytest

from rhal.evaluation.experiment import RunSpec, Simulation
from rhal.evaluation.impact import impact_shocks, impacted_dataset, sentiment_shifts


@pytest.fixture(scope="module")
def sim(small_cfg, small_dataset, detector):
    return Simulation(small_cfg, small_dataset, detector, 42)


@pytest.fixture(scope="module")
def tape(sim):
    return sim.run(RunSpec("injected__none", "misgen")).tape


def test_gamma_zero_changes_nothing(sim, tape, small_cfg):
    assert impacted_dataset(sim.ds, tape, 0.0, small_cfg) is sim.ds


def test_shock_follows_the_sentiment_shift(sim, tape):
    shifts = sentiment_shifts(sim.ds, tape)
    shock = impact_shocks(sim.ds, tape, 1.0)
    days = [t for t, v in shifts.items() if abs(v) > 0.2]
    assert days and all(np.sign(shock[t]) == np.sign(shifts[t]) for t in days)
    assert np.allclose(impact_shocks(sim.ds, tape, 2.0), 2 * shock)


def test_impact_is_causal(sim, tape, small_cfg):
    """Rewriting the tape's documents after a cut-off cannot move prices or features before it."""
    cut = (sim.t0 + sim.t1) // 2
    late = copy.deepcopy(tape)
    for date in list(late.docs):
        if sim.ds.dates.index(date) > cut:
            for d in late.docs[date]:
                d.text = "Nordwind Systems shares collapsed after a record loss and a downgrade."
    a, b = impacted_dataset(sim.ds, tape, 1.0, small_cfg), impacted_dataset(sim.ds, late, 1.0, small_cfg)
    assert a.prices.iloc[: cut + 1].equals(b.prices.iloc[: cut + 1])
    assert a.feats.iloc[: cut + 1].fillna(-999).equals(b.feats.iloc[: cut + 1].fillna(-999))
    assert not a.prices.iloc[cut + 1 :].equals(b.prices.iloc[cut + 1 :])


def test_both_twins_trade_the_moved_prices(sim, tape, small_cfg, detector):
    moved = impacted_dataset(sim.ds, tape, 2.0, small_cfg)
    assert not np.allclose(moved.prices["close"], sim.ds.prices["close"])
    r = Simulation(small_cfg, moved, detector, 42).run(
        RunSpec("frozen__grounded", "replay", True, False, True), tape=tape
    )
    assert [s["executed_price"] for s in r.treated_steps] == [s["executed_price"] for s in r.shadow_steps]
    assert r.treated_steps[-1]["executed_price"] == pytest.approx(moved.prices["open"].iloc[sim.t1])
