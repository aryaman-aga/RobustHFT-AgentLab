"""Explicit risk-control layer (our extension).

Runs *after* the policy and can only make the position smaller or force an exit;
it never initiates a trade the policy did not ask for (except stop/take-profit).
Every override is logged so the evaluation can attribute behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import Action
from .policy import PolicyOutput


@dataclass
class PortfolioState:
    cash: float
    shares: float = 0.0
    entry_price: float | None = None
    last_price: float = 0.0
    peak_equity: float = 0.0

    @property
    def equity(self) -> float:
        return self.cash + self.shares * self.last_price

    @property
    def position_frac(self) -> float:
        eq = self.equity
        return 0.0 if eq <= 0 else self.shares * self.last_price / eq

    @property
    def drawdown(self) -> float:
        return 0.0 if self.peak_equity <= 0 else self.equity / self.peak_equity - 1.0


class RiskLayer:
    def __init__(self, *, enabled: bool, max_position: float, vol_target: float, stop_loss: float, take_profit: float,
                 hold_on_conflict: bool, conflict_threshold: float, min_confidence: float, drawdown_throttle: float, allow_short: bool = False,
                 conflict_mode: str = "hold"):
        self.enabled = enabled
        self.max_position = max_position
        self.vol_target = vol_target
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.hold_on_conflict = hold_on_conflict
        self.conflict_threshold = conflict_threshold
        self.min_confidence = min_confidence
        self.drawdown_throttle = drawdown_throttle
        self.allow_short = allow_short
        # what the conflict rule does when consistency c < conflict_threshold:
        #   hold   -- no new trade (the paper's rule; changes the action, so an attacker can trigger it)
        #   shrink -- keep the action but move the position only a fraction c of the way to its target
        self.conflict_mode = conflict_mode

    def apply(self, out: PolicyOutput, state: PortfolioState, vol_5: float) -> tuple[Action, float, list[str]]:
        """Return (final_action, target_position_fraction, overrides)."""
        overrides: list[str] = []
        action = out.action
        cur = state.position_frac
        if not self.enabled:
            size = self.max_position
            target = size if action == Action.BUY else (-size if (action == Action.SELL and self.allow_short) else 0.0) if action == Action.SELL else cur
            return action, float(target), overrides

        # 1. stop-loss / take-profit on an open long position (checked at close t, executed t+1 open)
        if state.shares > 0 and state.entry_price:
            pnl = state.last_price / state.entry_price - 1.0
            if pnl <= -self.stop_loss:
                overrides.append(f"stop_loss({pnl:+.2%})")
                return Action.SELL, 0.0, overrides
            if pnl >= self.take_profit:
                overrides.append(f"take_profit({pnl:+.2%})")
                return Action.SELL, 0.0, overrides

        # 2. confidence gate
        if action != Action.HOLD and out.confidence < self.min_confidence:
            overrides.append(f"low_confidence({out.confidence:.2f})")
            action = Action.HOLD

        # 3. text/market conflict: hold (default) or shrink the move toward the target by c
        step_frac = 1.0
        if self.hold_on_conflict and action != Action.HOLD and out.consistency < self.conflict_threshold:
            if self.conflict_mode == "shrink":
                overrides.append(f"shrink_on_conflict(c={out.consistency:.2f})")
                step_frac = float(out.consistency)
            else:
                overrides.append(f"hold_on_conflict(c={out.consistency:.2f})")
                action = Action.HOLD

        # 4. volatility-scaled sizing + drawdown throttle
        size = self.max_position
        if vol_5 > 0:
            size = min(self.max_position, self.vol_target / vol_5)
        if state.drawdown <= -self.drawdown_throttle:
            overrides.append(f"dd_throttle({state.drawdown:+.2%})")
            size *= 0.5
        size = float(np.clip(size, 0.0, self.max_position))

        if action == Action.BUY:
            target = size * min(1.0, 0.5 + out.confidence)  # confidence-scaled
        elif action == Action.SELL:
            target = -size * min(1.0, 0.5 + out.confidence) if self.allow_short else 0.0
        else:
            target = cur
        target = cur + step_frac * (target - cur)  # shrink: only part of the way (step_frac=1 otherwise)
        return action, float(target), overrides
