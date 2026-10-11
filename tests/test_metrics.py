import math

from rhal.evaluation.experiment import NET_COLUMNS, attach_net_asr
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


def test_counterfactual_asr_by_hand():
    shadow = [_step(a, ["clean"], 100) for a in ("Buy", "Buy", "Hold", "Sell")]
    treated = [_step("Buy", ["synthetic"], 100), _step("Sell", ["synthetic"], 100), _step("Hold", ["synthetic"], 100), _step("Buy", ["synthetic"], 100)]
    for s, cf in zip(treated, ("Buy", "Hold", "Sell", "Buy")):
        s["cf_action"] = cf
    m = attack_metrics(treated, shadow)
    assert math.isclose(m["ASR"], 2 / 4)  # sequential: steps 2 and 4 differ from the shadow
    assert math.isclose(m["ASR_cf"], 3 / 4)  # counterfactual: steps 2 (Hold), 3 (Sell), 4 (Buy) differ
    assert math.isclose(m["ASR_cf_active"], 2 / 4)  # the Hold on step 2 is not an active flip
    del treated[0]["cf_action"]
    assert "ASR_cf" not in attack_metrics(treated, shadow)  # only reported when every step has it


def test_net_columns_by_hand():
    """Net columns subtract the control arm of the same seed and defence level; matched ones only for arms facing the tape."""
    def row(arm, seed, asr, cf):
        return {"arm": arm, "seed": seed, "ASR": asr, "ASR_cf": cf, "ASR_active": asr / 2, "signal_deviation": 0.1}

    rows = [
        row("frozen__grounded", 0, 0.40, 0.35), row("placebo__grounded", 0, 0.20, 0.25), row("placebo_bias__grounded", 0, 0.30, 0.20),
        row("injected__grounded", 0, 0.45, 0.40), row("injected__none", 0, 0.38, 0.36), row("placebo__none", 0, 0.22, 0.21),
        row("placebo_bias__none", 0, 0.26, 0.30),
        row("frozen__grounded", 1, 0.50, 0.45), row("placebo__grounded", 1, 0.10, 0.15), row("placebo_bias__grounded", 1, 0.25, 0.30),
        row("injected__none", 1, 0.30, 0.30),  # seed 1 has no placebo__none / placebo_bias__none
    ]
    out = {(r["seed"], r["arm"]): r for r in attach_net_asr(rows)}
    f0, f1 = out[(0, "frozen__grounded")], out[(1, "frozen__grounded")]
    assert math.isclose(f0["ASR_net"], 0.40 - 0.20) and math.isclose(f0["ASR_cf_net"], 0.35 - 0.25)
    assert math.isclose(f0["ASR_net_matched"], 0.40 - 0.30) and math.isclose(f0["ASR_cf_net_matched"], 0.35 - 0.20)
    assert math.isclose(f0["ASR_active_net"], 0.20 - 0.10) and math.isclose(f0["signal_deviation_net"], 0.0, abs_tol=1e-12)
    assert math.isclose(f1["ASR_net"], 0.50 - 0.10) and math.isclose(f1["ASR_net_matched"], 0.50 - 0.25)  # same-seed partners only
    # injected__none is the arm the tape was recorded from, so its matched columns are exact
    n0, n1 = out[(0, "injected__none")], out[(1, "injected__none")]
    assert math.isclose(n0["ASR_net"], 0.38 - 0.22) and math.isclose(n0["ASR_net_matched"], 0.38 - 0.26) and math.isclose(n0["ASR_cf_net_matched"], 0.36 - 0.30)
    assert all(math.isnan(n1[c]) for c in NET_COLUMNS)  # partners missing in seed 1
    # an adaptive attack on a defended agent is not the tape: classic net ASR yes, matched columns no
    g0 = out[(0, "injected__grounded")]
    assert math.isclose(g0["ASR_net"], 0.45 - 0.20) and math.isclose(g0["ASR_cf_net"], 0.40 - 0.25)
    assert math.isnan(g0["ASR_net_matched"]) and math.isnan(g0["ASR_cf_net_matched"])
    assert all(math.isnan(out[(0, a)][c]) for a in ("placebo__grounded", "placebo_bias__grounded") for c in NET_COLUMNS)  # controls get none
