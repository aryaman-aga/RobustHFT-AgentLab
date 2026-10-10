"""Decision policies.

RulePolicy      : ignores text; acts on market evidence only (technical baseline).
SurrogatePolicy : deterministic mock-LLM. Reads retrieved docs through the lexicon
                  scorer, applies the day's BiasConfig as explicit parameter shifts,
                  fuses with market evidence, and (if grounded) shrinks inconsistent
                  text toward market evidence:
                      s_g = s - kappa * (1 - c) * (s - s_mkt)
LLMPolicy       : builds the prompt (Perception + Profile + Memory + Evidence + docs),
                  calls the provider, parses JSON; falls back to the surrogate on failure.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from ..grounding.evidence import consistency_score
from ..llm.base import LLMClient
from ..schemas import Action, BiasConfig
from .memory_agent import AgentMemory
from .perception import Observation
from .profile import Profile
from .sentiment import score_sentiment


@dataclass
class PolicyOutput:
    action: Action
    confidence: float
    raw_signal: float
    text_signal: float
    market_signal: float
    consistency: float
    threshold: float
    rationale: str


def _basic_market_read(obs: Observation) -> float:
    """What an *ungrounded* agent sees numerically: recent return and drawdown only."""
    return float(np.tanh(8.0 * obs.ret_1 + 2.0 * obs.dd_20))


class RulePolicy:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def decide(self, obs: Observation, profile: Profile, memory: AgentMemory, *, grounded: bool, kappa: float, text_weight: float, adaptive_trust: bool) -> PolicyOutput:
        s_mkt = obs.evidence.market_signal if obs.evidence else _basic_market_read(obs)
        a = Action.BUY if s_mkt > self.threshold else Action.SELL if s_mkt < -self.threshold else Action.HOLD
        return PolicyOutput(a, min(1.0, abs(s_mkt)), s_mkt, 0.0, s_mkt, 1.0, self.threshold, "rule: market evidence only")


class SurrogatePolicy:
    def __init__(self, threshold: float = 0.15):
        self.base_threshold = threshold

    # -- text signal with bias operators -------------------------------------
    def text_signal(self, obs: Observation, bias: BiasConfig, memory: AgentMemory) -> float:
        if not obs.docs:
            return 0.0
        scores = np.array([score_sentiment(d.item.text) for d in obs.docs])
        w = np.array([max(d.score, 1e-6) for d in obs.docs])
        last = memory.last_action()
        if bias.confirm_weight > 0 and last is not None and last != Action.HOLD:
            agree = np.sign(scores) == last.sign
            w = w * (1 + bias.confirm_weight * agree)
        w = w / w.sum()
        s = float(np.sum(w * scores))
        if bias.downside_weight != 1.0 and s < 0:
            s *= bias.downside_weight
        if bias.anchor_weight > 0:
            prev = memory.prev_text_signal()
            if prev is not None:
                s = (1 - bias.anchor_weight) * s + bias.anchor_weight * prev
        if bias.herd_weight > 0:
            s += bias.herd_weight * memory.majority_action_sign()
        return float(np.clip(s, -1, 1))

    def decide(self, obs: Observation, profile: Profile, memory: AgentMemory, *, grounded: bool, kappa: float, text_weight: float, adaptive_trust: bool) -> PolicyOutput:
        bias = profile.bias
        s_text = self.text_signal(obs, bias, memory)
        if adaptive_trust:
            s_text *= 0.5 + memory.text_trust  # long-term memory modulates reliance on text
        s_mkt = obs.evidence.market_signal if (grounded and obs.evidence) else _basic_market_read(obs)
        c = consistency_score(s_text, s_mkt)
        s = text_weight * s_text + (1 - text_weight) * s_mkt
        if grounded:
            s = s - kappa * (1 - c) * (s - s_mkt)
        thr = self.base_threshold * bias.threshold_scale
        a = Action.BUY if s > thr else Action.SELL if s < -thr else Action.HOLD
        conf = float(np.clip(abs(s) / max(2 * thr, 1e-6), 0, 1))
        if bias.name == "overconfidence":
            conf = float(np.clip(conf * 1.3 + 0.1, 0, 1))
        rat = f"text {s_text:+.2f}, market {s_mkt:+.2f}, consistency {c:.2f}, fused {s:+.2f}" + (f", bias={bias.name}" if bias.name else "")
        return PolicyOutput(a, conf, float(s), s_text, s_mkt, c, thr, rat)


class LLMPolicy:
    def __init__(self, client: LLMClient, fallback: SurrogatePolicy):
        self.client = client
        self.fallback = fallback

    def build_prompt(self, obs: Observation, profile: Profile, memory: AgentMemory, grounded: bool) -> tuple[str, str]:
        sys_p = profile.system_prompt(grounded)
        lines = [f"Date: {obs.date}. Close {obs.close:.2f}, 1d return {obs.ret_1:+.2%}, 5d vol {obs.vol_5:.3f}, 20d drawdown {obs.dd_20:+.2%}."]
        if grounded and obs.evidence:
            lines.append("Temporal evidence: " + obs.evidence.summary)
        lines.append("Memory:\n" + memory.summary())
        lines.append("Retrieved news (most relevant first):")
        for i, d in enumerate(obs.docs, 1):
            lines.append(f"[{i}] ({d.age_days}d ago) {d.item.text}")
        lines.append("Decide now. JSON only.")
        return sys_p, "\n".join(lines)

    def decide(self, obs: Observation, profile: Profile, memory: AgentMemory, *, grounded: bool, kappa: float, text_weight: float, adaptive_trust: bool) -> PolicyOutput:
        sur = self.fallback.decide(obs, profile, memory, grounded=grounded, kappa=kappa, text_weight=text_weight, adaptive_trust=adaptive_trust)
        sys_p, user = self.build_prompt(obs, profile, memory, grounded)
        try:
            r = self.client.complete(sys_p, user)
            m = re.search(r"\{.*\}", r.text, re.S)
            obj = json.loads(m.group(0)) if m else {}
            a = Action(obj.get("action", "Hold").capitalize())
            conf = float(np.clip(float(obj.get("confidence", 0.5)), 0, 1))
            rat = str(obj.get("rationale", ""))[:300]
        except Exception as e:  # malformed output -> conservative hold, keep surrogate diagnostics
            a, conf, rat = Action.HOLD, 0.0, f"parse-failure fallback: {e}"
        raw = a.sign * max(conf, 0.01)
        return PolicyOutput(a, conf, raw, sur.text_signal, sur.market_signal, sur.consistency, sur.threshold, f"llm:{rat}")
