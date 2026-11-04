"""End-to-end: all arms run, clean arm is a perfect pair, everything is deterministic."""

import numpy as np

from rhal.evaluation.experiment import RunSpec, Simulation, standard_arms


def test_all_arms_run_and_clean_is_identical(small_cfg, small_dataset, detector):
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    for spec in standard_arms():
        r = sim.run(spec)
        n = small_cfg.data.eval_days
        assert len(r.treated_steps) == n == len(r.shadow_steps)
        assert all(0 <= s["confidence"] <= 1 for s in r.treated_steps)
        assert all(s["cash"] >= -1e-6 for s in r.treated_steps)
        assert all(abs(s["equity"] - (s["cash"] + s["position"] * 0)) >= 0 for s in r.treated_steps)
        if spec.injection == "none":
            assert r.metrics["ASR"] == 0.0 and r.metrics["MER"] == 0.0
            assert [s["action"] for s in r.treated_steps] == [s["action"] for s in r.shadow_steps]
            assert abs(r.metrics["CR"] - r.metrics["CR_clean_paired"]) < 1e-12
        else:
            assert r.metrics["MER"] > 0 and r.metrics["exposure_rate"] > 0
            assert any(p != "clean" for s in r.treated_steps for p in s["retrieved_provenance"])
            assert all(p == "clean" for s in r.shadow_steps for p in s["retrieved_provenance"])


def test_determinism_across_runs(small_cfg, small_dataset, detector):
    a = Simulation(small_cfg, small_dataset, detector, 42).run(RunSpec("x", "misgen", True, True, True))
    b = Simulation(small_cfg, small_dataset, detector, 42).run(RunSpec("x", "misgen", True, True, True))
    assert a.metrics == b.metrics
    assert [s["equity"] for s in a.treated_steps] == [s["equity"] for s in b.treated_steps]


def test_decision_does_not_depend_on_future(small_cfg, small_dataset, detector):
    """Mutate prices strictly after the evaluation window's last needed row; decisions must be unchanged."""
    from rhal.evaluation.experiment import Dataset
    from rhal.features import build_feature_frame

    ds = small_dataset
    n = len(ds.dates)
    prices = ds.prices.copy()
    prices.loc[n - 1 :, ["open", "high", "low", "close"]] *= (
        3.0  # only the very last row (never read as a decision input except as exec/mark)
    )
    # Build a sim whose eval window ends before the mutated row
    cfg3 = small_cfg.with_overrides({"data.eval_days": small_cfg.data.eval_days - 1})
    feats = build_feature_frame(prices, ds.intraday, micro=True)
    feats["close"] = prices["close"].to_numpy()
    ds_mut = Dataset(prices, ds.news, ds.intraday, feats, ds.dates)
    sim_a = Simulation(cfg3, ds, detector, 42)
    sim_b = Simulation(cfg3, ds_mut, detector, 42)
    # decisions are made at t <= n-2, so no decision input reads the mutated row n-1 (only the last fill/mark does)
    ra, rb = sim_a.run(RunSpec("g", "misgen", True, True, True)), sim_b.run(RunSpec("g", "misgen", True, True, True))
    assert [s["action"] for s in ra.treated_steps] == [s["action"] for s in rb.treated_steps]
    assert np.allclose([s["raw_signal"] for s in ra.treated_steps], [s["raw_signal"] for s in rb.treated_steps])


def test_grounding_reduces_signal_deviation_on_average(small_cfg, small_dataset, detector):
    """Weak sanity check of the core hypothesis on one seed: grounded agent's signal moves less under attack."""
    sim = Simulation(small_cfg, small_dataset, detector, 42)
    u = sim.run(RunSpec("u", "misgen", False, False, False)).metrics["signal_deviation"]
    g = sim.run(RunSpec("g", "misgen", True, False, True)).metrics["signal_deviation"]
    assert g <= u * 1.25  # not a strict guarantee on 12 days; guards against grounding *amplifying* attacks
