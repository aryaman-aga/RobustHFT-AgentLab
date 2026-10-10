"""Typed records shared across modules (pydantic v2).

Provenance labels are mandatory on every news record so that synthetic text
can never be confused with real text anywhere in the pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Provenance = Literal["clean", "synthetic", "placebo"]


class Action(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"

    @property
    def sign(self) -> int:
        return {"Buy": 1, "Sell": -1, "Hold": 0}[self.value]


class NewsItem(BaseModel):
    """One news document in the pool."""

    id: str
    date: str  # ISO date, the day the item becomes visible
    text: str
    provenance: Provenance = "clean"
    parent_id: Optional[str] = None  # for synthetic docs: the clean source
    strategy: Optional[dict] = None  # {"bias":..., "minor":..., "rewrite":...}
    true_sentiment: Optional[float] = None  # latent sentiment used by the data generator
    safety_label: str = "SYNTHETIC-FOR-ROBUSTNESS-TESTING-ONLY"

    @field_validator("safety_label")
    @classmethod
    def _label_required(cls, v: str) -> str:
        if not v:
            raise ValueError("safety_label must be non-empty")
        return v


class RetrievedDoc(BaseModel):
    item: NewsItem
    score: float
    age_days: int


class MarketEvidence(BaseModel):
    """Structured temporal evidence for one decision step (paper Eq. 19 + our micro features)."""

    date: str
    features: dict[str, float]  # normalized feature vector (z-scores), no lookahead
    market_signal: float  # in [-1, 1]
    analog_mean_forward_return: Optional[float] = None
    analog_hit_rate: Optional[float] = None  # fraction of analogs with positive forward return
    analog_dates: list[str] = Field(default_factory=list)
    summary: str = ""


class Decision(BaseModel):
    date: str
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    raw_signal: float
    text_signal: float
    market_signal: float
    consistency: float = Field(ge=0.0, le=1.0)
    target_position: float  # fraction of equity, after risk layer
    rationale: str = ""
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_provenance: list[str] = Field(default_factory=list)
    risk_overrides: list[str] = Field(default_factory=list)


class BiasConfig(BaseModel):
    """Numeric operationalization of a behavioral bias applied to the agent context.

    In LLM mode these map to system-prompt text (paper App. E.2). In mock mode
    they map to explicit parameter shifts in the surrogate policy so that
    Phi_Bias is measurable without an LLM.
    """

    name: Optional[str] = None
    threshold_scale: float = 1.0  # overconfidence lowers the action threshold
    downside_weight: float = 1.0  # loss aversion amplifies negative signals
    anchor_weight: float = 0.0  # anchoring blends with previous signal
    herd_weight: float = 0.0  # herding pulls toward recent majority action
    confirm_weight: float = 0.0  # confirmation bias boosts agreeing docs
    prompt: str = ""


class StepResult(BaseModel):
    date: str
    decision: Decision
    executed_price: Optional[float] = None
    position: float
    cash: float
    equity: float
    daily_return: float
    turnover: float
    cost_paid: float
