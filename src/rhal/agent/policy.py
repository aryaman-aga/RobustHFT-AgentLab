"""Decision policies.

RulePolicy      : ignores text; acts on market evidence only (technical baseline).
SurrogatePolicy : deterministic mock-LLM. Reads retrieved docs through a text scorer (the
                  finance lexicon, or an LLM in reader mode -- llm_reader.py), applies the
                  day's BiasConfig as explicit parameter shifts, fuses with market evidence,
                  and (if grounded) shrinks inconsistent text toward market evidence:
                      s_g = s - kappa * (1 - c) * (s - s_mkt)
LLMPolicy       : decider mode. Builds the prompt (Perception + Profile + Memory + Evidence +
                  docs), asks the LLM for schema-checked JSON {action, confidence, news_stance,
                  rationale}. The LLM's own news_stance is the text signal (so the risk layer
                  and memory never see lexicon numbers). Connection errors raise; an unreadable
                  answer becomes Hold and is flagged (parse_failed -> metric parse_fail_rate).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

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
    parse_failed: bool = False  # decider mode: the LLM answer could not be read


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


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Smallest value whose cumulative weight reaches half; ties broken by position (stable)."""
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def weighted_trimmed_mean(x: np.ndarray, w: np.ndarray, trim: float) -> float:
    """Weighted mean after removing ``trim`` of the total weight from each tail."""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order] / w.sum()
    hi = np.cumsum(ws)
    lo = hi - ws
    kept = np.clip(np.minimum(hi, 1 - trim) - np.maximum(lo, trim), 0.0, None)  # weight of each item inside [trim, 1-trim]
    return float(np.sum(kept * xs) / kept.sum()) if kept.sum() > 0 else weighted_median(x, w)


class SurrogatePolicy:
    def __init__(self, threshold: float = 0.15, scorer: Callable[[str], float] | None = None, text_agg: str = "mean", trim_frac: float = 0.25):
        self.base_threshold = threshold
        self.scorer = scorer or score_sentiment  # lexicon by default; LLMTextScorer in reader mode
        self.text_agg = text_agg  # mean (paper) | median | trimmed: robust-aggregation baseline
        self.trim_frac = trim_frac

    # -- text signal with bias operators -------------------------------------
    def text_signal(self, obs: Observation, bias: BiasConfig, memory: AgentMemory) -> float:
        if not obs.docs:
            return 0.0
        scores = np.array([self.scorer(d.item.text) for d in obs.docs])
        w = np.array([max(d.score, 1e-6) for d in obs.docs])
        last = memory.last_action()
        if bias.confirm_weight > 0 and last is not None and last != Action.HOLD:
            agree = np.sign(scores) == last.sign
            w = w * (1 + bias.confirm_weight * agree)
        w = w / w.sum()
        if self.text_agg == "median":
            s = weighted_median(scores, w)
        elif self.text_agg == "trimmed":
            s = weighted_trimmed_mean(scores, w, self.trim_frac)
        else:
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


DECIDER_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["Buy", "Sell", "Hold"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "news_stance": {"type": "number", "minimum": -1, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["action", "confidence", "news_stance", "rationale"],
}


def _parse_json(text: str) -> dict:
    """JSON object from a reply; tolerates code fences or prose around it (API providers)."""
    try:
        obj = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


class LLMPolicy:
    def __init__(self, client: LLMClient, fallback: SurrogatePolicy, max_tokens: int = 160):
        self.client = client
        self.fallback = fallback  # only supplies the action threshold
        self.max_tokens = max_tokens

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
        thr = self.fallback.base_threshold * profile.bias.threshold_scale
        s_mkt = obs.evidence.market_signal if (grounded and obs.evidence) else _basic_market_read(obs)
        sys_p, user = self.build_prompt(obs, profile, memory, grounded)
        r = self.client.complete(sys_p, user, max_tokens=self.max_tokens, schema=DECIDER_SCHEMA)  # network errors propagate
        try:
            obj = _parse_json(r.text)
            a = Action(str(obj["action"]).strip().capitalize())
            conf = float(np.clip(float(obj["confidence"]), 0, 1))
            stance = float(np.clip(float(obj["news_stance"]), -1, 1))
            rat, failed = str(obj.get("rationale", ""))[:300], False
        except (ValueError, KeyError, TypeError) as e:  # unreadable answer -> conservative Hold, flagged
            a, conf, stance, rat, failed = Action.HOLD, 0.0, 0.0, f"parse-failure fallback: {e}", True
        c = consistency_score(stance, s_mkt)  # the LLM's own reading of the news, not the lexicon's
        raw = a.sign * max(conf, 0.01)
        return PolicyOutput(a, conf, raw, stance, s_mkt, c, thr, f"llm:{rat}", parse_failed=failed)
