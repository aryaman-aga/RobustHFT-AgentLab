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

Randomness uses *keyed* streams: the day's bias draw uses rng(seed, "bias", date)
and each item's perturbation uses rng(seed, "item", date, item_id), instead of one
sequential generator. Two engines with the same seed therefore draw the same
uniforms for the same (day, item) no matter what happened earlier in their runs.
This gives common random numbers across defence arms: with feedback off they inject
identical documents, and with feedback on they differ only where the learned
strategy probabilities differ. A single sequential stream would desynchronise after
the first differing accept/reject and turn every later draw into fresh noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agent.sentiment import score_sentiment
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
        evidence_aware: str = "none",
    ):
        self.seed = seed  # all draws come from rng(seed, <purpose>, date[, item_id]); see module docstring
        self.bias_prob = bias_prob
        self.use_bias = use_bias
        self.use_minor = use_minor
        self.use_rewrite = use_rewrite
        self.use_feedback = use_feedback
        self.delta = delta
        self.max_tries = max_tries
        self.injection_ratio = injection_ratio
        self.placebo = placebo
        # Defense-aware attacker (our extension): instead of the first acceptable candidate, try all
        # max_tries and keep the one whose lexicon sentiment best agrees ("aligned") or disagrees
        # ("conflict") with the day's public market evidence s_mkt. "none" = the paper's attacker.
        self.evidence_aware = evidence_aware
        self.filter = QualityFilter(SimilarityEvaluator(background_corpus), detector, tau_sim, detector.threshold if tau_det is None else tau_det, enabled=use_filter)
        self.s_bias = StrategySampler(BIAS_TYPES, alpha, use_feedback)
        self.s_minor = StrategySampler(MINOR_TYPES, alpha, use_feedback)
        self.s_rewrite = StrategySampler(REWRITE_TYPES, alpha, use_feedback)
        self.history = StrategyHistory()
        self.stats = GenStats()
        self._today_bias: str | None = None

    # ---------------------------------------------------------------------
    def sample_bias(self, cr: float, date: str) -> BiasConfig:
        """Day-level bias for the agent context (Phi_Bias)."""
        g = rng(self.seed, "bias", date)
        if self.placebo or not self.use_bias or g.random() >= self.bias_prob:
            self._today_bias = None
            return BiasConfig()
        self._today_bias = self.s_bias.sample(g, cr=cr, delta=self.delta)
        return bias_config(self._today_bias)

    def generate_for_items(self, items: list[NewsItem], date: str, s_mkt: float | None = None) -> list[NewsItem]:
        """Produce synthetic variants for the given clean items (Algorithm 1 inner loop).

        ``s_mkt`` (public market evidence at this date, computed from data up to the date only) is
        used by the defense-aware attacker; the paper's attacker ignores it.
        """
        aware = self.evidence_aware != "none" and s_mkt is not None
        out: list[NewsItem] = []
        n_target = int(round(self.injection_ratio * len(items)))
        if n_target <= 0:
            return out
        if n_target >= len(items):
            chosen = items
        else:
            chosen = [items[i] for i in rng(self.seed, "select", date).choice(len(items), n_target, replace=False)]
        for k, item in enumerate(chosen):
            g = rng(self.seed, "item", date, item.id)  # this item's own stream, shared by all arms
            if self.placebo:
                text = apply_placebo(item.text, g)
                out.append(self._mk(item, text, date, k, {"bias": None, "minor": "placebo", "rewrite": None}, "placebo"))
                continue
            accepted = None
            cands: list[tuple[str, dict, bool]] = []  # (text, strategy, via_rewrite); aware mode keeps them all
            for _ in range(self.max_tries):
                self.stats.generated += 1
                if self.use_minor:
                    m = self.s_minor.sample(g)
                    cand = apply_minor(item.text, m, g)
                else:
                    m = "none"
                    cand = apply_placebo(item.text, g)  # w/o Minor: general rewriting only
                s, d, ok_s, ok_d = self.filter.evaluate(item.text, cand)
                self.stats.sim_scores.append(s)
                self.stats.det_scores.append(d)
                strat = {"bias": self._today_bias, "minor": m, "rewrite": None, "sim": round(s, 3), "det": round(d, 3)}
                if ok_s and ok_d:
                    cands.append((cand, strat, False))
                    if not aware:
                        break
                    continue
                if self.use_rewrite and ok_s:
                    r = self.s_rewrite.sample(g)
                    cand2 = apply_rewrite(cand, r, g)
                    s2, d2, ok_s2, ok_d2 = self.filter.evaluate(item.text, cand2)
                    self.stats.det_scores.append(d2)
                    if ok_d2 and ok_s2:
                        cands.append((cand2, {**strat, "rewrite": r, "sim": round(s2, 3), "det": round(d2, 3)}, True))
                        if not aware:
                            break
            if cands:
                text, strat, via_rewrite = self._choose(cands, s_mkt) if aware else cands[0]
                accepted = (text, strat)
                self.stats.accepted_after_rewrite += int(via_rewrite)
            if accepted is None:
                self.stats.rejected += 1
                continue
            self.stats.accepted += 1
            out.append(self._mk(item, accepted[0], date, k, accepted[1], "synthetic"))
        return out

    def _choose(self, cands: list[tuple[str, dict, bool]], s_mkt: float) -> tuple[str, dict, bool]:
        """Defense-aware pick: maximise (+/-) sign(s_mkt) * lexicon sentiment; first maximum wins."""
        direction = float(np.sign(s_mkt)) * (1.0 if self.evidence_aware == "aligned" else -1.0)
        align = [float(np.sign(s_mkt)) * score_sentiment(c[0]) for c in cands]
        i = int(np.argmax([direction * a for a in align])) if direction != 0 else 0
        text, strat, via = cands[i]
        return text, {**strat, "aware": self.evidence_aware, "s_mkt": round(float(s_mkt), 3), "align": round(align[i], 3)}, via

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
