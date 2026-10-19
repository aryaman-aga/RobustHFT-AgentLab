import pytest

from rhal.agent.policy import PolicyOutput
from rhal.agent.risk import PortfolioState, RiskLayer
from rhal.schemas import Action


def _risk(**kw):
    base = dict(enabled=True, max_position=1.0, vol_target=0.015, stop_loss=0.06, take_profit=0.12, hold_on_conflict=True, conflict_threshold=0.6, min_confidence=0.3, drawdown_throttle=0.10)
    base.update(kw)
    return RiskLayer(**base)


def _out(action=Action.BUY, conf=0.9, cons=0.9):
    return PolicyOutput(action, conf, 0.5, 0.5, 0.4, cons, 0.15, "t")


def test_disabled_passes_through():
    r = _risk(enabled=False)
    s = PortfolioState(cash=100.0)
    a, tgt, ov = r.apply(_out(), s, 0.05)
    assert a == Action.BUY and tgt == 1.0 and ov == []
    a, tgt, _ = r.apply(_out(Action.SELL), s, 0.05)
    assert tgt == 0.0


def test_stop_loss_and_take_profit():
    r = _risk()
    s = PortfolioState(cash=0.0, shares=10, entry_price=100.0, last_price=93.0)
    a, tgt, ov = r.apply(_out(Action.BUY), s, 0.01)
    assert a == Action.SELL and tgt == 0.0 and ov[0].startswith("stop_loss")
    s.last_price = 113.0
    a, tgt, ov = r.apply(_out(Action.HOLD), s, 0.01)
    assert a == Action.SELL and ov[0].startswith("take_profit")


def test_hold_on_conflict_and_low_confidence():
    r = _risk()
    s = PortfolioState(cash=100.0, last_price=10.0)
    a, tgt, ov = r.apply(_out(cons=0.4), s, 0.01)
    assert a == Action.HOLD and any(o.startswith("hold_on_conflict") for o in ov)
    a, tgt, ov = r.apply(_out(conf=0.1), s, 0.01)
    assert a == Action.HOLD and any(o.startswith("low_confidence") for o in ov)
    r2 = _risk(hold_on_conflict=False)
    a, _, ov = r2.apply(_out(cons=0.4), s, 0.01)
    assert a == Action.BUY and ov == []


def test_vol_scaling_and_drawdown_throttle():
    r = _risk()
    s = PortfolioState(cash=100.0, last_price=10.0, peak_equity=100.0)
    _, tgt_lowvol, _ = r.apply(_out(conf=1.0), s, 0.005)
    _, tgt_hivol, _ = r.apply(_out(conf=1.0), s, 0.03)
    assert tgt_lowvol == 1.0 and abs(tgt_hivol - 0.5) < 1e-9  # 0.015/0.03
    s2 = PortfolioState(cash=85.0, last_price=10.0, peak_equity=100.0)  # 15% drawdown
    _, tgt_dd, ov = r.apply(_out(conf=1.0), s2, 0.005)
    assert abs(tgt_dd - 0.5) < 1e-9 and any(o.startswith("dd_throttle") for o in ov)


def test_hold_keeps_current_position():
    r = _risk()
    s = PortfolioState(cash=50.0, shares=5, entry_price=10.0, last_price=10.0, peak_equity=100.0)
    a, tgt, _ = r.apply(_out(Action.HOLD), s, 0.01)
    assert a == Action.HOLD and abs(tgt - 0.5) < 1e-9


def test_shrink_on_conflict_keeps_the_action_and_moves_part_of_the_way():
    s = PortfolioState(cash=100.0, last_price=10.0)  # flat: current position 0
    full_a, full_t, _ = _risk(hold_on_conflict=False).apply(_out(cons=0.4), s, 0.01)
    a, t, ov = _risk(conflict_mode="shrink").apply(_out(cons=0.4), s, 0.01)
    assert a == full_a == Action.BUY and any(o.startswith("shrink_on_conflict") for o in ov)
    assert t == pytest.approx(0.4 * full_t)  # c = 0.4 of the way from 0 to the unconstrained target
    a, t2, ov = _risk(conflict_mode="shrink").apply(_out(cons=0.9), s, 0.01)  # no conflict: unchanged
    assert t2 == pytest.approx(full_t) and not any("conflict" in o for o in ov)
