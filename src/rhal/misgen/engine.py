"""MisGen engine: Algorithm 1 (paper App. E.1) with closed-loop feedback.

Per trading day the engine:
  1. samples a day-level bias b_t (with prob p) -> BiasConfig for the agent context,
  2. for each clean item n_i visible that day, samples m_i and produces
     Minor(n_i, m_i); if it fails the filter, applies Rewrite(., r_i) and re-checks,
  3. returns accepted synthetic items with provenance='synthetic' and the
     strategy tuple attached for later attribution.

After the agent acts, ``update_history`` reinforces the strategies of the
synthetic docs that were retrieved on a day where the injected decision
differed from the paired clean decision (paper Eq. 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schemas import BiasConfig, NewsItem
from ..seeding import rng
from .detector import DetectabilityModel
from .filters import QualityFilter, SimilarityEvaluator
from .operators import BIAS_TYPES, MINOR_TYPES, REWRITE_TYPES, apply_minor, apply_placebo, apply_rewrite, bias_config
from .sampler import StrategySampler


@dataclass
class StrategyHistory:
    """HistoryEffect = {H_B, H_M, H_R}."""

    bias: dict[str, int] = field(default_factory=lambda: {b: 0 for b in BIAS_TYPES})
    minor: dict[str, int] = field(default_factory=lambda: {m: 0 for m in MINOR_TYPES})
    rewrite: dict[str, int] = field(default_factory=lambda: {r: 0 for r in REWRITE_TYPES})


@dataclass
class GenStats:
    generated: int = 0
    accepted: int = 0
    accepted_after_rewrite: int = 0
    rejected: int = 0
    sim_scores: list[float] = field(default_factory=list)
    det_scores: list[float] = field(default_factory=list)


class MisGenEngine:
    def __init__(
        self,
        *,
        seed: int,
        background_corpus: list[str],
        detector: DetectabilityModel,
        bias_prob: float = 0.7,
        use_bias: bool = True,
        use_minor: bool = True,
        use_rewrite: bool = True,
        use_filter: bool = True,
        use_feedback: bool = True,
        alpha: float = 1.0,
        delta: float = 0.15,
        max_tries: int = 3,
        tau_sim: float = 0.35,
        tau_det: float | None = None,
        injection_ratio: float = 1.0,
        placebo: bool = False,
    ):
        self.g = rng(seed, "misgen")
        self.bias_prob = bias_prob
        self.use_bias = use_bias
        self.use_minor = use_minor
        self.use_rewrite = use_rewrite
        self.use_feedback = use_feedback
        self.delta = delta
        self.max_tries = max_tries
        self.injection_ratio = injection_ratio
        self.placebo = placebo
        self.filter = QualityFilter(SimilarityEvaluator(background_corpus), detector, tau_sim, detector.threshold if tau_det is None else tau_det, enabled=use_filter)
        self.s_bias = StrategySampler(BIAS_TYPES, alpha, use_feedback)
        self.s_minor = StrategySampler(MINOR_TYPES, alpha, use_feedback)
        self.s_rewrite = StrategySampler(REWRITE_TYPES, alpha, use_feedback)
        self.history = StrategyHistory()
        self.stats = GenStats()
        self._today_bias: str | None = None

    # ---------------------------------------------------------------------
    def sample_bias(self, cr: float) -> BiasConfig:
        """Day-level bias for the agent context (Phi_Bias)."""
        if self.placebo or not self.use_bias or self.g.random() >= self.bias_prob:
            self._today_bias = None
            return BiasConfig()
        self._today_bias = self.s_bias.sample(self.g, cr=cr, delta=self.delta)
        return bias_config(self._today_bias)

    def generate_for_items(self, items: list[NewsItem], date: str) -> list[NewsItem]:
        """Produce synthetic variants for the given clean items (Algorithm 1 inner loop)."""
        out: list[NewsItem] = []
        n_target = int(round(self.injection_ratio * len(items)))
        if n_target <= 0:
            return out
        chosen = items if n_target >= len(items) else [items[i] for i in self.g.choice(len(items), n_target, replace=False)]
        for k, item in enumerate(chosen):
            if self.placebo:
                text = apply_placebo(item.text, self.g)
                out.append(self._mk(item, text, date, k, {"bias": None, "minor": "placebo", "rewrite": None}, "placebo"))
                continue
            accepted = None
            for _ in range(self.max_tries):
                self.stats.generated += 1
                if self.use_minor:
                    m = self.s_minor.sample(self.g)
                    cand = apply_minor(item.text, m, self.g)
                else:
                    m = "none"
                    cand = apply_placebo(item.text, self.g)  # w/o Minor: general rewriting only
                s, d, ok_s, ok_d = self.filter.evaluate(item.text, cand)
                self.stats.sim_scores.append(s)
                self.stats.det_scores.append(d)
                strat = {"bias": self._today_bias, "minor": m, "rewrite": None, "sim": round(s, 3), "det": round(d, 3)}
                if ok_s and ok_d:
                    accepted = (cand, strat)
                    break
                if self.use_rewrite and ok_s:
                    r = self.s_rewrite.sample(self.g)
                    cand2 = apply_rewrite(cand, r, self.g)
                    s2, d2, ok_s2, ok_d2 = self.filter.evaluate(item.text, cand2)
                    self.stats.det_scores.append(d2)
                    if ok_d2 and ok_s2:
                        accepted = (cand2, {**strat, "rewrite": r, "sim": round(s2, 3), "det": round(d2, 3)})
                        self.stats.accepted_after_rewrite += 1
                        break
            if accepted is None:
                self.stats.rejected += 1
                continue
            self.stats.accepted += 1
            out.append(self._mk(item, accepted[0], date, k, accepted[1], "synthetic"))
        return out

    def _mk(self, src: NewsItem, text: str, date: str, k: int, strat: dict, prov: str) -> NewsItem:
        return NewsItem(
            id=f"syn_{src.id}_{k}",
            date=date,
            text=text,
            provenance=prov,  # type: ignore[arg-type]
            parent_id=src.id,
            strategy=strat,
            true_sentiment=src.true_sentiment,
        )

    # ---------------------------------------------------------------------
    def update_history(self, decision_changed: bool, retrieved_synthetic: list[NewsItem]) -> None:
        """Paper Eq. 7: reinforce strategies that were retrieved on a flipped decision."""
        if not decision_changed or not self.use_feedback:
            return
        if self._today_bias:
            self.history.bias[self._today_bias] += 1
            self.s_bias.reinforce(self._today_bias)
        for it in retrieved_synthetic:
            st = it.strategy or {}
            m, r = st.get("minor"), st.get("rewrite")
            if m in self.history.minor:
                self.history.minor[m] += 1
                self.s_minor.reinforce(m)
            if r in self.history.rewrite:
                self.history.rewrite[r] += 1
                self.s_rewrite.reinforce(r)

    def summary(self) -> dict:
        st = self.stats
        return {
            "generated": st.generated, "accepted": st.accepted, "accepted_after_rewrite": st.accepted_after_rewrite,
            "rejected": st.rejected, "accept_rate": st.accepted / max(st.generated, 1),
            "mean_sim": float(np.mean(st.sim_scores)) if st.sim_scores else None,
            "mean_det": float(np.mean(st.det_scores)) if st.det_scores else None,
            "history": {"bias": self.history.bias, "minor": self.history.minor, "rewrite": self.history.rewrite},
            "final_probs": {
                "bias": dict(zip(BIAS_TYPES, self.s_bias.probabilities().round(3).tolist())),
                "minor": dict(zip(MINOR_TYPES, self.s_minor.probabilities().round(3).tolist())),
                "rewrite": dict(zip(REWRITE_TYPES, self.s_rewrite.probabilities().round(3).tolist())),
            },
        }


def build_detector_training_set(clean_texts: list[str], seed: int, per_type: int | None = None) -> tuple[list[str], list[str]]:
    """Create (clean, perturbed) lists by applying each Phi_Minor operator to clean text."""
    g = rng(seed, "detector_train")
    perturbed: list[str] = []
    n = per_type or max(len(clean_texts) // len(MINOR_TYPES), 20)
    for m in MINOR_TYPES:
        idx = g.choice(len(clean_texts), size=min(n, len(clean_texts)), replace=False)
        for i in idx:
            perturbed.append(apply_minor(clean_texts[i], m, g))
    return list(clean_texts), perturbed
