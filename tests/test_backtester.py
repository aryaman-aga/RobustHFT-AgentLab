import pytest

from rhal.agent.risk import PortfolioState
from rhal.backtest import Backtester


def test_equity_identity_and_no_trade():
    bt = Backtester(fee_bps=0, slippage_bps=0)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    f = bt.execute(s, 0.0, 10.0)
    assert f.shares_delta == 0 and s.equity == 1000.0
    bt.mark(s, 12.0)
    assert s.equity == 1000.0  # no position -> price moves don't matter


def test_full_buy_then_sell_roundtrip_costs():
    bt = Backtester(fee_bps=10, slippage_bps=5)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    f = bt.execute(s, 1.0, 10.0)
    assert s.cash >= -1e-9, "cash must never go negative"
    assert abs(s.equity - (s.cash + s.shares * s.last_price)) < 1e-9
    assert f.price == 10.0 * 1.0005
    bt.execute(s, 0.0, 10.0)
    assert s.shares == 0 and s.entry_price is None
    assert s.cash < 1000.0  # fees + slippage strictly reduce equity on a round trip
    assert s.cash > 995.0


def test_partial_rebalance_and_turnover():
    bt = Backtester(fee_bps=0, slippage_bps=0)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    f1 = bt.execute(s, 0.5, 10.0)
    assert abs(s.shares - 50) < 1e-9 and abs(f1.turnover - 0.5) < 1e-9
    bt.mark(s, 20.0)  # equity = 500 + 50*20 = 1500
    assert abs(s.equity - 1500) < 1e-9
    f2 = bt.execute(s, 0.5, 20.0)  # target 750/20 = 37.5 shares -> sell 12.5
    assert abs(s.shares - 37.5) < 1e-9 and f2.shares_delta < 0


def test_long_only_clamps_short():
    bt = Backtester(fee_bps=0, slippage_bps=0, allow_short=False)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    bt.execute(s, -1.0, 10.0)
    assert s.shares == 0.0


def test_entry_price_weighted_average():
    bt = Backtester(fee_bps=0, slippage_bps=0)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    bt.execute(s, 0.5, 10.0)  # 50 @10
    bt.mark(s, 20.0)  # equity 1500
    bt.execute(s, 1.0, 20.0)  # target 75 shares -> buy 25 @20 (cash 500 -> 0)
    assert abs(s.shares - 75) < 1e-9
    assert abs(s.entry_price - (50 * 10 + 25 * 20) / 75) < 1e-9


def test_min_notional_skips_dust():
    bt = Backtester(fee_bps=0, slippage_bps=0, min_notional=5.0)
    s = PortfolioState(cash=1000.0)
    bt.mark(s, 10.0)
    f = bt.execute(s, 0.001, 10.0)  # $1 -> below min notional
    assert f.shares_delta == 0.0
