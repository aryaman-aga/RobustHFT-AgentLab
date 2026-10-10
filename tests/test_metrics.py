import math

from rhal.evaluation.metrics import attack_metrics, compute_metrics, portfolio_metrics


def _step(action, prov, equity, ret=0.0, syn_frac=0.5, strategies=None, bias=None, raw=0.3):
    return {
        "action": action, "retrieved_provenance": prov, "equity": equity, "daily_return": ret, "turnover": 0.1, "shares_delta": 1.0,
        "cost_paid": 0.0, "visible_synthetic_frac": syn_frac, "retrieved_strategies": strategies or [], "bias": bias, "risk_overrides": [], "raw_signal": raw,
    }


def test_mer_asr_by_hand():
    treated = [
        _step("Buy", ["synthetic", "clean", "clean", "clean"], 101, strategies=[{"minor": "flipping"}], raw=0.5),
        _step("Sell", ["synthetic", "synthetic", "clean", "clean"], 100, strategies=[{"minor": "flipping"}, {"minor": "causal"}], raw=-0.4),
        _step("Hold", ["clean", "clean", "clean", "clean"], 100, raw=0.0),
        _step("Buy", ["placebo", "clean", "clean", "clean"], 102, raw=0.3),
    ]
    shadow = [_step("Buy", ["clean"] * 4, 100, raw=0.5), _step("Buy", ["clean"] * 4, 100, raw=0.4), _step("Hold", ["clean"] * 4, 100, raw=0.0), _step("Sell", ["clean"] * 4, 100, raw=-0.2)]
    m = attack_metrics(treated, shadow)
    assert math.isclose(m["MER"], 4 / 16)
    assert math.isclose(m["MER_base_rate"], 0.5) and math.isclose(m["MER_lift"], 0.5)
    assert math.isclose(m["ASR"], 2 / 4)
    assert math.isclose(m["ASR_active"], 2 / 4)  # both flips (Sell, Buy) are active
    assert math.isclose(m["exposure_rate"], 3 / 4)
    assert math.isclose(m["ASR_given_exposed"], 2 / 3)  # flips on steps 2 and 4, both exposed; step 1 exposed no flip
    assert m["exposure_by_type"] == {"flipping": 2, "causal": 1}
    assert m["flips_by_type"] == {"flipping": 1, "causal": 1}
    assert math.isclose(m["signal_deviation"], (0 + 0.8 + 0 + 0.5) / 4)


def test_defensive_hold_counts_in_asr_but_not_asr_active():
    treated = [_step("Hold", ["synthetic"], 100), _step("Sell", ["synthetic"], 100)]
    shadow = [_step("Buy", ["clean"], 100), _step("Buy", ["clean"], 100)]
    m = attack_metrics(treated, shadow)
    assert m["ASR"] == 1.0 and m["ASR_active"] == 0.5


def test_portfolio_metrics():
    steps = [_step("Buy", [], 110, 0.10), _step("Hold", [], 99, -0.10), _step("Buy", [], 108.9, 0.10)]
    m = portfolio_metrics(steps, 100.0)
    assert math.isclose(m["CR"], 0.089)
    assert math.isclose(m["max_drawdown"], 99 / 110 - 1)
    assert m["n_trades"] == 3 and math.isclose(m["hold_rate"], 1 / 3)


def test_compute_metrics_paired_delta():
    t = [_step("Buy", ["clean"], 105, 0.05)]
    s = [_step("Buy", ["clean"], 102, 0.02)]
    m = compute_metrics(t, s, 100.0)
    assert math.isclose(m["CR_delta_vs_clean"], 0.03)
    assert m["ASR"] == 0.0
