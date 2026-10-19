"""Agent memory (FinMem-style, simplified).

Working memory : last ``k`` step records (date, action, signals, realised return, reflection).
Long-term      : running *text-trust* estimate = EWMA of sign-agreement between
                 yesterday's text signal and today's realised return, plus insights.
The trust estimate is only consumed when ``adaptive_trust`` is on (ablation knob).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..schemas import Action


@dataclass
class StepRecord:
    date: str
    action: Action
    text_signal: float
    market_signal: float
    raw_signal: float
    realized_return: float | None = None
    reflection: str = ""


@dataclass
class AgentMemory:
    k: int = 5
    working: deque = field(default_factory=deque)
    text_trust: float = 0.5  # in [0,1]
    insights: list[str] = field(default_factory=list)
    ewma: float = 0.2

    def __post_init__(self) -> None:
        self.working = deque(maxlen=self.k)

    def record(self, rec: StepRecord) -> None:
        self.working.append(rec)

    def reflect(self, realized_return: float) -> None:
        """Attach realised return to the latest record and update long-term trust."""
        if not self.working:
            return
        last = self.working[-1]
        last.realized_return = realized_return
        agree = np.sign(last.text_signal) == np.sign(realized_return) if abs(last.text_signal) > 0.05 else None
        if agree is not None:
            self.text_trust = (1 - self.ewma) * self.text_trust + self.ewma * float(agree)
        consistent = abs(last.text_signal - last.market_signal) < 0.5
        last.reflection = (
            f"On {last.date} chose {last.action.value} (text {last.text_signal:+.2f}, market {last.market_signal:+.2f}, "
            f"{'consistent' if consistent else 'CONFLICTING'}); realised {realized_return:+.2%}."
        )
        if agree is False and not consistent:
            self.insights.append(f"{last.date}: text signal contradicted market evidence and was wrong.")

    # -- accessors used by the bias operators --------------------------------
    def prev_text_signal(self) -> float | None:
        return self.working[-1].text_signal if self.working else None

    def majority_action_sign(self) -> int:
        if not self.working:
            return 0
        s = sum(r.action.sign for r in self.working)
        return int(np.sign(s))

    def last_action(self) -> Action | None:
        return self.working[-1].action if self.working else None

    def summary(self) -> str:
        if not self.working:
            return "No prior steps."
        lines = [r.reflection or f"{r.date}: {r.action.value}" for r in list(self.working)[-3:]]
        lines.append(f"Long-term text-trust estimate: {self.text_trust:.2f}.")
        if self.insights:
            lines.append("Insights: " + " | ".join(self.insights[-2:]))
        return "\n".join(lines)
