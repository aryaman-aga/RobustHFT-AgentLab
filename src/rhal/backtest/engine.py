"""Event-driven accounting engine (daily; optional intraday features are upstream).

Invariants (tested):
* equity == cash + shares * price at every mark,
* cash never negative (no leverage): buys are capped by available cash incl. fees,
* with zero costs and no trades, equity is constant,
* a round trip with costs strictly reduces equity by fees + slippage.
No broker, no live data, ever.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agent.risk import PortfolioState
from .costs import fee, fill_price


@dataclass
class Fill:
    price: float
    shares_delta: float
    notional: float
    fee_paid: float
    turnover: float


class Backtester:
    def __init__(self, *, fee_bps: float = 10.0, slippage_bps: float = 5.0, allow_short: bool = False, min_notional: float = 1.0):
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.allow_short = allow_short
        self.min_notional = min_notional

    def execute(self, state: PortfolioState, target_frac: float, price: float) -> Fill:
        """Rebalance to ``target_frac`` of equity at ``price`` (pre-slippage)."""
        equity = state.cash + state.shares * price
        if equity <= 0:
            return Fill(price, 0.0, 0.0, 0.0, 0.0)
        target_shares = target_frac * equity / price
        if not self.allow_short:
            target_shares = max(0.0, target_shares)
        delta = target_shares - state.shares
        if abs(delta) * price < self.min_notional:
            return Fill(price, 0.0, 0.0, 0.0, 0.0)
        side = 1 if delta > 0 else -1
        px = fill_price(price, side, self.slippage_bps)
        if delta > 0:  # cap by cash so cash never goes negative
            max_affordable = state.cash / (px * (1 + self.fee_bps / 1e4))
            delta = min(delta, max_affordable)
            if delta * price < self.min_notional:
                return Fill(price, 0.0, 0.0, 0.0, 0.0)
        notional = abs(delta) * px
        f = fee(notional, self.fee_bps)
        state.cash -= delta * px + f
        prev_shares = state.shares
        state.shares += delta
        # entry-price bookkeeping (long side)
        if prev_shares <= 1e-12 and state.shares > 1e-12:
            state.entry_price = px
        elif delta > 0 and state.shares > 1e-12:
            state.entry_price = (prev_shares * (state.entry_price or px) + delta * px) / state.shares
        elif state.shares <= 1e-12:
            state.shares = 0.0
            state.entry_price = None
        return Fill(px, delta, notional, f, notional / equity)

    @staticmethod
    def mark(state: PortfolioState, price: float) -> float:
        state.last_price = price
        eq = state.equity
        state.peak_equity = max(state.peak_equity, eq)
        return eq
