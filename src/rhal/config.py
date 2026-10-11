"""Typed configuration loaded from YAML, with dotted-path overrides.

Usage::

    cfg = load_config("configs/default.yaml", overrides={"grounding.enabled": False})
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .utils.io import load_yaml


class IntradayCfg(BaseModel):
    enabled: bool = False
    bars_per_day: int = 78


class DataCfg(BaseModel):
    source: Literal["synthetic", "csv", "yfinance"] = "synthetic"
    csv_prices: str = "data/sample/prices.csv"
    csv_news: str = "data/sample/news.csv"
    symbol: str = "SYNTH"
    n_days: int = 400
    eval_days: int = 60
    news_per_day: int = 5
    intraday: IntradayCfg = IntradayCfg()


class MicroCfg(BaseModel):
    enabled: bool = True
    rv_window: int = 5
    volume_window: int = 20


class FeaturesCfg(BaseModel):
    vol_window: int = 5
    drawdown_window: int = 20
    short_trend: int = 5
    long_trend: int = 20
    anomaly_window: int = 20
    micro: MicroCfg = MicroCfg()


class GroundingCfg(BaseModel):
    enabled: bool = True
    window: int = 30
    top_k: int = 5
    horizon: int = 5
    strength: float = Field(0.7, ge=0.0, le=1.0)


class RetrievalCfg(BaseModel):
    method: Literal["tfidf", "bm25", "embed"] = "tfidf"
    top_k: int = 4
    recency_days: int = 3
    recency_decay: float = 0.5


class MisGenCfg(BaseModel):
    enabled: bool = True
    mode: Literal["mock", "llm"] = "mock"
    bias_prob: float = 0.7
    use_bias: bool = True
    use_minor: bool = True
    use_rewrite: bool = True
    use_filter: bool = True
    use_feedback: bool = True
    alpha: float = 1.0
    delta: float = 0.15
    max_tries: int = 3
    sim_threshold: float = 0.35
    det_threshold: Optional[float] = None  # None -> use detector's calibrated threshold
    injection_ratio: float = 1.0
    placebo: bool = False


class AgentCfg(BaseModel):
    mode: Literal["rule", "mock", "llm"] = "mock"
    action_threshold: float = 0.15
    text_weight: float = 0.75
    memory_len: int = 5
    reflection: bool = True
    adaptive_trust: bool = False  # long-term memory modulates text reliance (ablation knob)


class RiskCfg(BaseModel):
    enabled: bool = True
    max_position: float = 1.0
    vol_target: float = 0.015
    stop_loss: float = 0.06
    take_profit: float = 0.12
    hold_on_conflict: bool = True
    conflict_threshold: float = 0.6
    min_confidence: float = 0.3
    drawdown_throttle: float = 0.10


class BacktestCfg(BaseModel):
    initial_cash: float = 100_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    execution: Literal["next_open", "same_close"] = "next_open"


class ExperimentCfg(BaseModel):
    n_seeds: int = 100
    counterfactual_asr: bool = True  # also compute the one-step counterfactual ASR (+1 agent step/day; doubles LLM calls in llm mode)


class LLMCfg(BaseModel):
    """Populated from environment (.env); never from YAML so keys stay out of configs."""

    provider: Literal["mock", "anthropic", "openai"] = "mock"
    model: str = "claude-haiku-4-5-20251001"
    cache_dir: str = "outputs/llm_cache"
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class Config(BaseModel):
    seed: int = 42
    output_dir: str = "outputs"
    data: DataCfg = DataCfg()
    features: FeaturesCfg = FeaturesCfg()
    grounding: GroundingCfg = GroundingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    misgen: MisGenCfg = MisGenCfg()
    agent: AgentCfg = AgentCfg()
    risk: RiskCfg = RiskCfg()
    backtest: BacktestCfg = BacktestCfg()
    experiment: ExperimentCfg = ExperimentCfg()
    llm: LLMCfg = LLMCfg()

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with dotted-path overrides applied."""
        d = self.model_dump()
        for k, v in overrides.items():
            _set_dotted(d, k, v)
        return Config.model_validate(d)


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _llm_from_env() -> LLMCfg:
    load_dotenv(override=False)
    provider = os.getenv("RHAL_LLM_PROVIDER", "mock").strip().lower() or "mock"
    if provider not in ("mock", "anthropic", "openai"):
        provider = "mock"
    key = None
    base_url = None
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY") or None
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY") or None
        base_url = os.getenv("OPENAI_BASE_URL") or None
    return LLMCfg(
        provider=provider,  # type: ignore[arg-type]
        model=os.getenv("RHAL_LLM_MODEL", "claude-haiku-4-5-20251001"),
        cache_dir=os.getenv("RHAL_LLM_CACHE_DIR", "outputs/llm_cache"),
        api_key=key,
        base_url=base_url,
    )


def load_config(path: str | Path = "configs/default.yaml", overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load YAML -> Config, merge LLM settings from env, apply overrides."""
    raw = copy.deepcopy(load_yaml(path)) if Path(path).exists() else {}
    raw.pop("llm", None)  # never accept keys from YAML
    cfg = Config.model_validate(raw)
    cfg = cfg.model_copy(update={"llm": _llm_from_env()})
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg
