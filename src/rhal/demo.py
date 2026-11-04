"""Interactive demo: one trading day, your own headlines, the agent with and without the defense layer.

Why this exists: the experiments report averages over thousands of decisions. The demo shows one
decision end to end, so a reader can see *why* grounding helps. It runs the experiments' own code:
the agent's news reader (the finance lexicon, or a local LLM in reader mode), ``SurrogatePolicy.decide``
(fusion, attacker bias, temporal grounding; paper Eqs. 2 and 4), ``EvidenceBuilder`` (market evidence
from prices up to the chosen day, with leak-free analogs) and ``RiskLayer``, on one day of the
synthetic market. As in the experiments, each verdict compares an attacked agent with an identical
clean twin that reads only the genuine news.

Two simplifications, both stated in the page: every headline gets the same retrieval weight (the
experiments weight by TF-IDF relevance and recency), and every agent starts flat with an empty memory,
so memory-based biases (anchoring, herding, confirmation) have nothing to act on and are not offered.

Safety: the demo generates no text. Headlines come from the user or from the hand-written presets
about the fictional company; in lexicon mode they stay in memory, in LLM mode the reader's answers are
cached in ``outputs/llm_cache/`` exactly as in the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import lexicon as L
from .agent.llm_reader import LLMTextScorer
from .agent.memory_agent import AgentMemory
from .agent.perception import Observation
from .agent.policy import PolicyOutput, SurrogatePolicy, _basic_market_read
from .agent.profile import Profile
from .agent.risk import PortfolioState, RiskLayer
from .agent.sentiment import score_sentiment, sentiment_hits
from .config import Config
from .evaluation.experiment import prepare_dataset, train_detector
from .features.windows import feature_columns
from .grounding import EvidenceBuilder
from .grounding.evidence import consistency_score
from .llm import build_client
from .misgen import DetectabilityModel, bias_config
from .schemas import MarketEvidence, NewsItem, RetrievedDoc

COMPANY = L.ENTITY["company"]
MAX_HEADLINES = 6  # per side: keeps the page readable and an LLM reader quick
MAX_CHARS = 400
BIASES = ("none", "overconfidence", "loss_aversion")  # the biases that act without any agent memory
READERS = ("lexicon", "llm")
RISK_MODES = ("off", "hold", "shrink")
CHART_DAYS = 60

# Hand-written examples about the fictional company. The wording avoids the first three words after
# "Nordwind", because the lexicon's negation window reads the "no" inside that name as a negator.
PRESETS: list[dict] = [
    {
        "id": "flip",
        "title": "Fake reversal against a rising market",
        "scenario": "rising",
        "genuine": [f"{COMPANY} said quarterly revenue jumped on strong cloud demand; analysts upgraded the stock."],
        "injected": [f"{COMPANY} said quarterly revenue plunged on weak cloud demand; analysts downgraded the stock."],
        "bias": "none",
        "note": "The attacker flips the direction of a genuine headline (MisGen's 'direction flip' edit).",
    },
    {
        "id": "aligned",
        "title": "Fake news that agrees with the market",
        "scenario": "falling",
        "genuine": [f"{COMPANY} said quarterly revenue was in line with guidance; the outlook is unchanged."],
        "injected": [
            f"{COMPANY} said quarterly revenue tumbled and warned of weak demand; analysts downgraded the stock."
        ],
        "bias": "none",
        "note": "The fake headline pushes where the market already points, so there is no conflict to spot.",
    },
    {
        "id": "limit",
        "title": "When the market gives no clear signal",
        "scenario": "flat",
        "genuine": [f"{COMPANY} said quarterly revenue jumped on strong cloud demand; analysts upgraded the stock."],
        "injected": [
            f"{COMPANY} said quarterly revenue plunged on weak cloud demand; analysts downgraded the stock.",
            f"Reports say the {COMPANY} chief executive warned of losses and a dividend cut.",
        ],
        "bias": "loss_aversion",
        "note": "Two fakes and a loss-aversion prompt on a flat market: with no clear evidence to check the "
        "news against, grounding cannot stop the attack. This is a limit of the defense.",
    },
    {
        "id": "record",
        "title": "Record high vs record low (local LLM)",
        "scenario": "rising",
        "genuine": [f"{COMPANY} sales touch record high"],
        "injected": [f"{COMPANY} sales touch record low"],
        "bias": "none",
        "reader": "llm",
        "note": "Free wording needs the local LLM reader (start it once with: make llm-up). The paper's word list "
        "counts 'record' as positive and does not know 'low', so it reads both headlines the same way.",
    },
]


class DemoError(ValueError):
    """A request the demo cannot serve; the message is shown to the user as is."""


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    t: int
    evidence: MarketEvidence
    basic_read: float  # the undefended agent's crude market read (last return and drawdown)
    chart: dict = field(default_factory=dict)


class Demo:
    """Serves scenarios and analyses for the demo page; one instance per server process."""

    def __init__(
        self,
        cfg: Config,
        seed: int | None = None,
        llm_model: str = "qwen2.5:7b",
        detector: DetectabilityModel | None = None,
    ):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.llm_model = llm_model
        self.ds = prepare_dataset(cfg, self.seed)
        n = len(self.ds.dates)
        self.t0, self.t1 = n - cfg.data.eval_days - 1, n - 1  # the experiments' evaluation window
        g = cfg.grounding
        # the defended agent's evidence model, exactly as Simulation builds it for the grounded level (micro on)
        self.evidence = EvidenceBuilder(
            self.ds.feats,
            ["z_" + c for c in feature_columns(True)],
            micro=True,
            window=g.window,
            horizon=g.horizon,
            top_k=g.top_k,
            analog_scale=float(g.analog_scale) if g.analog_scale != "auto" else 0.03,
        )
        self.scenarios = self._pick_scenarios()
        self._detector = detector  # None: load the experiments' saved model, or train one, on first use
        self._llm_reader: LLMTextScorer | None = None

    # -- market scenarios ------------------------------------------------------------------------------
    def _observation(self, t: int, docs: list[RetrievedDoc], evidence: MarketEvidence | None) -> Observation:
        row = self.ds.feats.iloc[t]

        def fv(k: str) -> float:
            v = row.get(k)
            return 0.0 if v is None or pd.isna(v) else float(v)

        return Observation(
            t=t,
            date=str(row["date"]),
            close=fv("close"),
            ret_1=fv("ret_1"),
            vol_5=fv("vol_5"),
            dd_20=fv("dd_20"),
            docs=docs,
            evidence=evidence,
        )

    def _pick_scenarios(self) -> dict[str, Scenario]:
        """Three days of the evaluation window: clearly rising, clearly falling and flat market evidence.
        Rising/falling days also need the crude read to agree, so both agents see the same direction."""
        rows = []
        for t in range(self.t0, self.t1):
            ev = self.evidence.build(t)
            rows.append((t, ev, _basic_market_read(self._observation(t, [], None))))
        rising = max((r for r in rows if r[2] > 0), key=lambda r: (r[1].market_signal, -r[0]))
        falling = min((r for r in rows if r[2] < 0), key=lambda r: (r[1].market_signal, r[0]))
        flat = min(rows, key=lambda r: (abs(r[1].market_signal), r[0]))
        out = {}
        for sid, name, desc, (t, ev, basic) in (
            ("rising", "Rising market", "Prices and similar past windows point up.", rising),
            ("falling", "Falling market", "Prices and similar past windows point down.", falling),
            ("flat", "Flat market", "The market evidence is close to neutral.", flat),
        ):
            out[sid] = Scenario(sid, name, desc, t, ev, basic, self._chart(t))
        return out

    def _chart(self, t: int) -> dict:
        lo = max(0, t - CHART_DAYS + 1)
        px = self.ds.prices.iloc[lo : t + 1]
        return {"dates": px["date"].astype(str).tolist(), "close": [round(float(x), 4) for x in px["close"]]}

    def scenario_payload(self) -> list[dict]:
        out = []
        for s in self.scenarios.values():
            ev = s.evidence
            out.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "date": ev.date,
                    "s_mkt": round(ev.market_signal, 3),
                    "basic_read": round(s.basic_read, 3),
                    "analog_mean_forward_return": ev.analog_mean_forward_return,
                    "analog_hit_rate": ev.analog_hit_rate,
                    "n_analogs": len(ev.analog_dates),
                    "momentum_5": _feat(ev, "z_mom_5"),
                    "drawdown_20": _feat(ev, "z_dd_20"),
                    "chart": s.chart,
                }
            )
        return out

    # -- readers and detector --------------------------------------------------------------------------
    def reader(self, name: str):
        if name == "lexicon":
            return score_sentiment
        if self._llm_reader is None:
            cfg = self.cfg.with_overrides({"llm.provider": "ollama", "llm.model": self.llm_model})
            self._llm_reader = LLMTextScorer(build_client(cfg.llm))
        return self._llm_reader

    def detector(self) -> DetectabilityModel:
        """The experiments' detectability classifier: the saved model if present, else trained (a few seconds)."""
        if self._detector is None:
            path = Path(self.cfg.output_dir) / "models" / "detector.pkl"
            self._detector = DetectabilityModel.load(path) if path.exists() else train_detector(self.cfg, self.cfg.seed)
        return self._detector

    # -- one agent, one day ----------------------------------------------------------------------------
    def _agent(self, s: Scenario, texts: list[tuple[str, str]], *, defended: bool, scorer, bias: str, risk: str):
        date = s.evidence.date
        docs = [
            RetrievedDoc(item=NewsItem(id=f"demo{i}", date=date, text=txt, provenance=prov), score=1.0, age_days=0)
            for i, (txt, prov) in enumerate(texts)
        ]
        obs = self._observation(s.t, docs, s.evidence if defended else None)
        a, g, r = self.cfg.agent, self.cfg.grounding, self.cfg.risk
        profile = Profile(symbol=self.cfg.data.symbol, bias=bias_config(None if bias == "none" else bias))
        out: PolicyOutput = SurrogatePolicy(a.action_threshold, scorer=scorer).decide(
            obs,
            profile,
            AgentMemory(k=a.memory_len),
            grounded=defended,
            kappa=g.strength,
            text_weight=a.text_weight,
            adaptive_trust=False,  # needs a trading history; off in the experiments too
        )
        layer = RiskLayer(
            enabled=defended and risk != "off",
            max_position=r.max_position,
            vol_target=r.vol_target,
            stop_loss=r.stop_loss,
            take_profit=r.take_profit,
            hold_on_conflict=r.hold_on_conflict,
            conflict_threshold=r.conflict_threshold,
            min_confidence=r.min_confidence,
            drawdown_throttle=r.drawdown_throttle,
            conflict_mode="shrink" if risk == "shrink" else "hold",
        )
        state = PortfolioState(cash=1.0, last_price=obs.close, peak_equity=1.0)  # flat: no open position
        final, target, overrides = layer.apply(out, state, obs.vol_5)
        return {
            "decision": out.action.value,
            "final": final.value,
            "position": round(float(target), 3),
            "confidence": round(out.confidence, 3),
            "s_text": round(out.text_signal, 3),
            "s_mkt": round(out.market_signal, 3),
            "consistency": round(out.consistency, 3),
            "s_fused": round(a.text_weight * out.text_signal + (1 - a.text_weight) * out.market_signal, 3),
            "s_final": round(out.raw_signal, 3),
            "threshold": round(out.threshold, 3),
            "overrides": overrides,
            "conflict": bool(defended and out.consistency < r.conflict_threshold),
        }

    # -- the request -----------------------------------------------------------------------------------
    def analyze(self, req: dict) -> dict:
        s = self.scenarios.get(str(req.get("scenario", "")))
        if s is None:
            raise DemoError(f"unknown market scenario; choose one of {', '.join(self.scenarios)}")
        genuine, injected = _headlines(req.get("genuine"), "genuine"), _headlines(req.get("injected"), "injected")
        if not genuine and not injected:
            raise DemoError("type at least one headline")
        reader, bias, risk = (str(req.get(k, d)) for k, d in (("reader", "lexicon"), ("bias", "none"), ("risk", "off")))
        for value, allowed, what in ((reader, READERS, "reader"), (bias, BIASES, "bias"), (risk, RISK_MODES, "risk")):
            if value not in allowed:
                raise DemoError(f"unknown {what} {value!r}; choose one of {', '.join(allowed)}")
        try:
            scorer = self.reader(reader)
            scores = {txt: float(scorer(txt)) for txt in genuine + injected}
        except (ConnectionError, OSError, RuntimeError) as e:  # the local LLM is not running or not pulled
            raise DemoError(f"The local LLM reader is not available: {e}") from e

        clean = [(t, "clean") for t in genuine]
        attacked = clean + [(t, "synthetic") for t in injected]
        pairs = {}
        for key, defended in (("none", False), ("defended", True)):
            twin = self._agent(s, clean, defended=defended, scorer=scorer, bias="none", risk=risk)
            hit = self._agent(s, attacked, defended=defended, scorer=scorer, bias=bias, risk=risk)
            pairs[key] = {"clean": twin, "attacked": hit, "flipped": twin["final"] != hit["final"]}

        det = self.detector()
        heads = []
        for role, texts in (("genuine", genuine), ("injected", injected)):
            for txt in texts:
                d = det.score(txt)
                heads.append(
                    {
                        "text": txt,
                        "role": role,
                        "score": round(scores[txt], 3),
                        "agreement": round(consistency_score(scores[txt], s.evidence.market_signal), 3),
                        "words": [
                            {"phrase": h.phrase, "weight": round(h.weight, 3), "notes": list(h.notes)}
                            for h in sentiment_hits(txt)
                        ]
                        if reader == "lexicon"
                        else [],
                        "detector": {"score": round(d, 3), "flagged": bool(d > det.threshold)},
                    }
                )
        a, g, r = self.cfg.agent, self.cfg.grounding, self.cfg.risk
        return {
            "scenario": s.id,
            "reader": reader,
            "bias": bias,
            "risk": risk,
            "headlines": heads,
            "pairs": pairs,
            "params": {
                "text_weight": a.text_weight,
                "kappa": g.strength,
                "threshold": a.action_threshold,
                "conflict_threshold": r.conflict_threshold,
                "detector_threshold": round(det.threshold, 3),
            },
        }


def _feat(ev: MarketEvidence, key: str) -> float | None:
    v = ev.features.get(key)
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def _headlines(value, what: str) -> list[str]:
    """Clean a list of headlines from the page: strings, trimmed, non-empty, bounded in count and length."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DemoError(f"{what} headlines must be a list of strings")
    out = [" ".join(v.split()) for v in value if v.strip()]
    if len(out) > MAX_HEADLINES:
        raise DemoError(f"at most {MAX_HEADLINES} {what} headlines")
    if any(len(v) > MAX_CHARS for v in out):
        raise DemoError(f"each headline must be at most {MAX_CHARS} characters")
    return out
