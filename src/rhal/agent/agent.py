"""POMDP-style trading agent assembling Perception, Profile, Memory, Action (+ risk layer)."""
from __future__ import annotations

from ..grounding import EvidenceBuilder
from ..memory import NewsPool
from ..schemas import Action, BiasConfig, Decision
from .memory_agent import AgentMemory, StepRecord
from .perception import Perceiver
from .policy import LLMPolicy, PolicyOutput, RulePolicy, SurrogatePolicy
from .profile import Profile
from .risk import PortfolioState, RiskLayer


class TradingAgent:
    def __init__(
        self,
        *,
        perceiver: Perceiver,
        evidence: EvidenceBuilder | None,
        policy: RulePolicy | SurrogatePolicy | LLMPolicy,
        risk: RiskLayer,
        profile: Profile | None = None,
        memory_len: int = 5,
        grounded: bool = True,
        kappa: float = 0.7,
        text_weight: float = 0.75,
        adaptive_trust: bool = False,
    ):
        self.perceiver = perceiver
        self.evidence = evidence
        self.policy = policy
        self.risk = risk
        self.profile = profile or Profile()
        self.memory = AgentMemory(k=memory_len)
        self.grounded = grounded and evidence is not None
        self.kappa = kappa
        self.text_weight = text_weight
        self.adaptive_trust = adaptive_trust
        self.last_output: PolicyOutput | None = None

    def set_bias(self, bias: BiasConfig) -> None:
        """Attacker-controlled context update (Phi_Bias)."""
        self.profile.bias = bias

    def step(self, t: int, pool: NewsPool, state: PortfolioState) -> Decision:
        ev = self.evidence.build(t) if self.evidence is not None else None
        last = self.memory.last_action()
        query = self.profile.query(ev if self.grounded else None, last.value if last else None)
        obs = self.perceiver.observe(t, pool, query, ev)
        out = self.policy.decide(obs, self.profile, self.memory, grounded=self.grounded, kappa=self.kappa, text_weight=self.text_weight, adaptive_trust=self.adaptive_trust)
        self.last_output = out
        action, target, overrides = self.risk.apply(out, state, obs.vol_5)
        self.memory.record(StepRecord(obs.date, action, out.text_signal, out.market_signal, out.raw_signal))
        return Decision(
            date=obs.date, action=action, confidence=out.confidence, raw_signal=out.raw_signal, text_signal=out.text_signal,
            market_signal=out.market_signal, consistency=out.consistency, target_position=target, rationale=out.rationale,
            retrieved_ids=obs.retrieved_ids, retrieved_provenance=obs.retrieved_provenance, risk_overrides=overrides,
            parse_failed=out.parse_failed, filter_visible=obs.filter_visible, filter_dropped=obs.filter_dropped,
        )

    def observe_return(self, realized_return: float) -> None:
        self.memory.reflect(realized_return)
