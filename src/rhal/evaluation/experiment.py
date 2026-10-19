"""Paired simulation runner.

Each run drives TWO agents in lockstep over the same market path:
  * shadow  -- reads only the clean pool, unbiased context,
  * treated -- reads clean + synthetic pool, attacker-controlled context.
They differ ONLY in what text they read. ASR is the per-step action disagreement;
the MisGen feedback loop (paper Eq. 7) is closed from the same pair.
With ``injection='none'`` the two agents are identical (a determinism check).

Two attack settings are compared across defences:
  * adaptive (``injection='misgen'``) -- MisGen runs live and its feedback loop
    adapts to the defended agent. It is seeded by (seed, injection type), never by
    arm name, so all adaptive arms of a seed draw the same random numbers (common
    random numbers) and arm labels cannot change results.
  * frozen (``injection='replay'``) -- an AttackTape recorded from one adaptive run
    (by default against the undefended agent) is replayed verbatim, so every defence
    faces bit-identical documents and bias contexts. Differences between frozen arms
    are caused by the defence alone (a transfer attack, as in adversarial-ML practice).

Matched placebo controls (built from the same tape, so exact for the frozen setting):
  * placebo         -- paraphrase EVERY clean item, no bias (classic noise floor);
  * placebo_matched -- paraphrase only the items the attack actually injected (same count);
  * placebo_bias    -- placebo_matched plus the attack's daily bias contexts.
All placebo variants share one paraphrase stream per (day, item), so a matched pool
is a subset of the classic placebo pool. This splits ASR into pool noise, count,
bias and meaning: meaning = ASR(frozen) - ASR(placebo_bias). Because the controls copy
one tape, ASR_net_matched is only reported for arms that face it (frozen__*, injected__none).

One-step counterfactual ASR (experiment.counterfactual_asr): after the first flip the
treated agent's portfolio and memory drift from the shadow's, so later flips can come
from that drift rather than from the text. Each day a third, throw-away agent decides
with the SHADOW's memory and portfolio but the treated pool and bias; ASR_cf compares
its action with the shadow's. It only reads copies, so it cannot change any other result.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..agent import LLMPolicy, Perceiver, PortfolioState, Profile, RiskLayer, RulePolicy, SurrogatePolicy, TradingAgent
from ..agent.llm_reader import LLMTextScorer
from ..backtest import Backtester
from ..config import Config
from ..data.synth import SynthConfig, generate_synthetic_dataset, simulate_news
from ..data import load_news, load_prices
from ..features import build_feature_frame
from ..features.windows import feature_columns
from ..grounding import EvidenceBuilder
from ..llm import build_client
from ..memory import NewsPool, Retriever
from ..misgen import DetectabilityModel, MisGenEngine, bias_config
from ..misgen.engine import build_detector_training_set
from ..schemas import NewsItem
from ..seeding import child_seed
from .metrics import compute_metrics

Injection = Literal["none", "misgen", "placebo", "replay", "placebo_matched", "placebo_bias"]

# Arm whose adaptive attack is recorded and replayed against every defence (frozen setting).
TAPE_SOURCE_ARM = "injected__none"
# Injection types that are driven by a recorded AttackTape instead of a live attacker.
TAPE_INJECTIONS = ("replay", "placebo_matched", "placebo_bias")


@dataclass
class RunSpec:
    name: str
    injection: Injection = "misgen"
    grounded: bool = False
    risk: bool = False
    micro: bool = False
    overrides: dict = field(default_factory=dict)


@dataclass
class AttackTape:
    """What the attacker did in one adaptive run, day by day, so it can be replayed exactly."""

    source: str  # arm the tape was recorded from
    docs: dict[str, list[NewsItem]] = field(default_factory=dict)  # date -> injected items
    bias: dict[str, str | None] = field(default_factory=dict)  # date -> bias name (None = no bias)


@dataclass
class RunResult:
    spec: RunSpec
    seed: int
    metrics: dict
    treated_steps: list[dict]
    shadow_steps: list[dict]
    misgen_summary: dict | None
    tape: AttackTape | None = None  # recorded for adaptive (injection='misgen') runs

    def to_row(self) -> dict:
        row = {"arm": self.spec.name, "seed": self.seed, "injection": self.spec.injection, "grounded": self.spec.grounded, "risk": self.spec.risk, "micro": self.spec.micro}
        row.update({k: v for k, v in self.metrics.items() if not isinstance(v, dict)})
        return row


# ---------------------------------------------------------------------------
# dataset preparation
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    prices: pd.DataFrame
    news: pd.DataFrame
    intraday: pd.DataFrame | None
    feats: pd.DataFrame
    dates: list[str]


def prepare_dataset(cfg: Config, seed: int) -> Dataset:
    if cfg.data.source == "synthetic":
        d = generate_synthetic_dataset(
            SynthConfig(n_days=cfg.data.n_days, news_per_day=cfg.data.news_per_day, intraday=cfg.data.intraday.enabled, bars_per_day=cfg.data.intraday.bars_per_day, symbol=cfg.data.symbol),
            child_seed(seed, "data"),
        )
        prices, news, intraday = d["prices"], d["news"], d.get("intraday")
    elif cfg.data.source == "real":  # one historical window per seed; template news generated per seed
        prices = real_window(load_prices(cfg.data.real_prices, repair=True), cfg.data.n_days, cfg.data.eval_days, seed - cfg.seed)
        news, intraday = simulate_news(prices, cfg.data.news_per_day, child_seed(seed, "data")), None
    else:
        prices, news, intraday = load_prices(cfg.data.csv_prices), load_news(cfg.data.csv_news), None
    f = cfg.features
    feats = build_feature_frame(
        prices, intraday, vol_window=f.vol_window, drawdown_window=f.drawdown_window, short_trend=f.short_trend, long_trend=f.long_trend,
        anomaly_window=f.anomaly_window, micro=True, rv_window=f.micro.rv_window, volume_window=f.micro.volume_window,
    )
    feats["close"] = prices["close"].to_numpy()
    return Dataset(prices, news, intraday, feats, prices["date"].tolist())


def real_window(full: pd.DataFrame, n_days: int, eval_days: int, index: int) -> pd.DataFrame:
    """The ``index``-th n_days window of a real history, counting back from the latest data.

    Consecutive windows are shifted by ``eval_days``, so their evaluation segments (the last
    ``eval_days`` rows) never overlap; indices wrap around after the last full window.
    """
    n_win = (len(full) - n_days) // eval_days + 1
    if n_win < 1:
        raise ValueError(f"real price history has {len(full)} rows, fewer than data.n_days={n_days}")
    end = len(full) - (index % n_win) * eval_days
    return full.iloc[end - n_days:end].reset_index(drop=True)


def n_real_windows(cfg: Config) -> int:
    """How many distinct evaluation windows the real price history offers."""
    return (len(load_prices(cfg.data.real_prices, repair=True)) - cfg.data.n_days) // cfg.data.eval_days + 1


def train_detector(cfg: Config, seed: int, target_fpr: float = 0.10) -> DetectabilityModel:
    """Train A_det on an independent synthetic corpus (never the evaluation corpus)."""
    d = generate_synthetic_dataset(SynthConfig(n_days=260, news_per_day=cfg.data.news_per_day), child_seed(seed, "detector_corpus"))
    clean, pert = build_detector_training_set(d["news"]["text"].tolist(), seed=child_seed(seed, "detector_train"))
    return DetectabilityModel().fit(clean, pert, target_fpr=target_fpr, seed=child_seed(seed, "detector_fit"))


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
class Simulation:
    def __init__(self, cfg: Config, dataset: Dataset, detector: DetectabilityModel, seed: int):
        self.cfg = cfg
        self.ds = dataset
        self.det = detector
        self.seed = seed
        n = len(dataset.dates)
        self.t0 = n - cfg.data.eval_days - 1
        self.t1 = n - 1  # decisions at t in [t0, t1), executed at t+1 <= n-1
        if self.t0 < 100:
            raise ValueError(f"not enough history: n_days={n}, eval_days={cfg.data.eval_days} (need >=100 warmup)")
        # Text models (retrieval TF-IDF, attacker similarity filter) are fit ONLY on news dated before
        # the evaluation window. Fitting on the whole corpus let future articles' vocabulary/IDF move
        # earlier decisions -- lookahead through text (tests/test_text_nolookahead.py).
        self.text_background = dataset.news.loc[dataset.news["date"].astype(str) < dataset.dates[self.t0], "text"].tolist()
        self._llm = None  # one LLM client per simulation, shared by every agent (built on first use)
        self._reader: LLMTextScorer | None = None  # reader mode: one text -> score function, shared
        self._filters: dict[float, object] = {}  # detection baseline: one filter per flag rate
        self.filter_thresholds: dict[float, float] = {}  # flag rate -> detector threshold (from pre-window news)
        self._det_scores: dict[str, float] = {}  # detector score per text (filters share it)
        self._attacker_ev: EvidenceBuilder | None = None  # defense-aware attacker's view of the market
        self.analog_scale = self._analog_scale(cfg)

    def _analog_scale(self, cfg: Config) -> float:
        """grounding.analog_scale, or for "auto" the median |H-day log return| over windows that end
        by the start of the evaluation window (uses no price after t0)."""
        if cfg.grounding.analog_scale != "auto":
            return float(cfg.grounding.analog_scale)
        h = cfg.grounding.horizon
        lc = np.log(self.ds.prices["close"].to_numpy(dtype=float)[: self.t0 + 1])
        return float(np.median(np.abs(lc[h:] - lc[:-h])))

    # -- baselines and the defense-aware attacker ----------------------------------
    def detect_filter(self, fpr: float):
        """Keep-predicate for retrieval: drop docs the detector flags. The threshold is the (1-fpr)
        quantile of detector scores on clean news dated before the evaluation window (no lookahead)."""
        if fpr not in self._filters:
            thr = float(np.quantile(self.det.predict_proba(self.text_background), 1.0 - fpr))
            self.filter_thresholds[fpr] = thr
            scores = self._det_scores

            def keep(item: NewsItem) -> bool:
                if item.text not in scores:
                    scores[item.text] = self.det.score(item.text)
                return scores[item.text] <= thr

            self._filters[fpr] = keep
        return self._filters[fpr]

    def attacker_market_signal(self, t: int) -> float:
        """Public market evidence at day t (prices up to t only) as the defense-aware attacker sees it.
        One fixed builder for every arm, so the attacker's view does not depend on the defense."""
        if self._attacker_ev is None:
            c = self.cfg.grounding
            self._attacker_ev = EvidenceBuilder(self.ds.feats, ["z_" + k for k in feature_columns(True)], micro=True,
                                                window=c.window, horizon=c.horizon, top_k=c.top_k, analog_scale=self.analog_scale)
        return self._attacker_ev.build(t).market_signal

    # -- LLM plumbing --------------------------------------------------------
    def llm_client(self):
        if self._llm is None:
            self._llm = build_client(self.cfg.llm)
        return self._llm

    def reader(self) -> LLMTextScorer:
        """Shared reader-mode scorer; a pure text -> score map, so sharing it cannot couple agents."""
        if self._reader is None:
            self._reader = LLMTextScorer(self.llm_client())
        return self._reader

    def llm_stats(self) -> dict | None:
        if self._reader is not None:
            return self._reader.stats()
        return self._llm.stats() if self._llm is not None else None

    # -- builders ----------------------------------------------------------
    def _agent(self, cfg: Config, spec: RunSpec) -> TradingAgent:
        zc = ["z_" + c for c in feature_columns(spec.micro)]
        doc_filter = self.detect_filter(cfg.defense.detect_fpr) if cfg.defense.detect_filter else None
        retr = Retriever(cfg.retrieval.method, cfg.retrieval.top_k, cfg.retrieval.recency_days, cfg.retrieval.recency_decay, background=self.text_background,
                         doc_filter=doc_filter)
        perc = Perceiver(self.ds.feats, retr, self.ds.dates)
        ev = EvidenceBuilder(self.ds.feats, zc, micro=spec.micro, window=cfg.grounding.window, horizon=cfg.grounding.horizon, top_k=cfg.grounding.top_k,
                             analog_scale=self.analog_scale) if spec.grounded else None
        if cfg.agent.mode == "rule":
            policy = RulePolicy(cfg.agent.action_threshold)
        elif cfg.agent.mode == "llm":  # decider mode: the LLM picks the action
            policy = LLMPolicy(self.llm_client(), SurrogatePolicy(cfg.agent.action_threshold))
        else:  # surrogate policy; reader mode swaps the lexicon for an LLM text scorer
            policy = SurrogatePolicy(cfg.agent.action_threshold, scorer=self.reader() if cfg.agent.text_scorer == "llm" else None,
                                     text_agg=cfg.agent.text_agg, trim_frac=cfg.agent.trim_frac)
        r = cfg.risk
        risk = RiskLayer(enabled=spec.risk, max_position=r.max_position, vol_target=r.vol_target, stop_loss=r.stop_loss, take_profit=r.take_profit,
                         hold_on_conflict=r.hold_on_conflict, conflict_threshold=r.conflict_threshold, min_confidence=r.min_confidence,
                         drawdown_throttle=r.drawdown_throttle, conflict_mode=r.conflict_mode)
        return TradingAgent(perceiver=perc, evidence=ev, policy=policy, risk=risk, profile=Profile(symbol=cfg.data.symbol), memory_len=cfg.agent.memory_len,
                            grounded=spec.grounded, kappa=cfg.grounding.strength, text_weight=cfg.agent.text_weight, adaptive_trust=cfg.agent.adaptive_trust)

    def _misgen(self, cfg: Config, spec: RunSpec) -> MisGenEngine | None:
        if spec.injection in ("none", "replay"):
            return None
        m = cfg.misgen
        matched = spec.injection in ("placebo_matched", "placebo_bias")
        # keyed by injection type, NOT spec.name: renaming an arm must never change its results.
        # Every placebo variant uses the "placebo" key, so the same (day, item) gets the same paraphrase.
        key = "misgen" if spec.injection == "misgen" else "placebo"
        return MisGenEngine(
            seed=child_seed(self.seed, "misgen", key), background_corpus=self.text_background, detector=self.det,
            bias_prob=m.bias_prob, use_bias=m.use_bias, use_minor=m.use_minor, use_rewrite=m.use_rewrite, use_filter=m.use_filter,
            use_feedback=m.use_feedback, alpha=m.alpha, delta=m.delta, max_tries=m.max_tries, tau_sim=m.sim_threshold, tau_det=m.det_threshold,
            injection_ratio=1.0 if matched else m.injection_ratio,  # matched: paraphrase exactly the tape's items
            placebo=(spec.injection != "misgen"),
            evidence_aware=m.evidence_aware if spec.injection == "misgen" else "none",
        )

    # -- main loop ---------------------------------------------------------
    def run(self, spec: RunSpec, tape: AttackTape | None = None) -> RunResult:
        """Run one paired arm. ``tape`` is required for, and only allowed with, the TAPE_INJECTIONS."""
        if (spec.injection in TAPE_INJECTIONS) != (tape is not None):
            raise ValueError(f"arm {spec.name!r}: pass a tape exactly when injection is one of {TAPE_INJECTIONS}")
        cfg = self.cfg.with_overrides(spec.overrides) if spec.overrides else self.cfg
        bt = Backtester(fee_bps=cfg.backtest.fee_bps, slippage_bps=cfg.backtest.slippage_bps)
        shadow_pool = NewsPool.from_frame(self.ds.news)
        treated_pool = NewsPool.from_frame(self.ds.news)
        shadow, treated = self._agent(cfg, spec), self._agent(cfg, spec)
        counterfactual = self._agent(cfg, spec) if cfg.experiment.counterfactual_asr else None
        s_state = PortfolioState(cash=cfg.backtest.initial_cash)
        t_state = PortfolioState(cash=cfg.backtest.initial_cash)
        misgen = self._misgen(cfg, spec)
        recorded = AttackTape(source=spec.name) if spec.injection == "misgen" else None
        close = self.ds.prices["close"].to_numpy(dtype=float)
        open_ = self.ds.prices["open"].to_numpy(dtype=float)
        s_steps: list[dict] = []
        t_steps: list[dict] = []
        for t in range(self.t0, self.t1):
            date = self.ds.dates[t]
            bias_name = None
            if spec.injection in ("misgen", "placebo"):  # live MisGen: adaptive attack or classic placebo
                cr = t_state.equity / cfg.backtest.initial_cash if t_state.last_price > 0 else 1.0
                bias = misgen.sample_bias(cr, date)
                bias_name = bias.name
                treated.set_bias(bias)
                today_clean = [it for it in shadow_pool.items() if it.date == date]
                s_mkt = self.attacker_market_signal(t) if spec.injection == "misgen" and cfg.misgen.evidence_aware != "none" else None
                injected = misgen.generate_for_items(today_clean, date, s_mkt=s_mkt)
                treated_pool.extend(injected)
                if recorded is not None:
                    recorded.docs[date], recorded.bias[date] = injected, bias_name
            elif spec.injection == "replay":  # frozen attack: replay exactly what the source run injected
                bias_name = tape.bias.get(date)
                treated.set_bias(bias_config(bias_name))
                treated_pool.extend(tape.docs.get(date, []))
            elif spec.injection in ("placebo_matched", "placebo_bias"):  # paraphrase only the items the tape attacked
                parents = list(dict.fromkeys(doc.parent_id for doc in tape.docs.get(date, [])))  # unique, in tape order
                treated_pool.extend(misgen.generate_for_items([shadow_pool.get(i) for i in parents], date))
                bias_name = tape.bias.get(date) if spec.injection == "placebo_bias" else None
                treated.set_bias(bias_config(bias_name))
            for st in (s_state, t_state):
                bt.mark(st, close[t])
            eq_s0, eq_t0 = s_state.equity, t_state.equity
            if counterfactual is not None:  # snapshot the shadow BEFORE it records today's step
                counterfactual.memory = copy.deepcopy(shadow.memory)
                counterfactual.set_bias(treated.profile.bias)
                cf_state = copy.deepcopy(s_state)
            d_s = shadow.step(t, shadow_pool, s_state)
            d_t = treated.step(t, treated_pool, t_state)
            d_cf = counterfactual.step(t, treated_pool, cf_state) if counterfactual is not None else None
            exec_px = open_[t + 1] if cfg.backtest.execution == "next_open" else close[t]
            f_s = bt.execute(s_state, d_s.target_position, exec_px)
            f_t = bt.execute(t_state, d_t.target_position, exec_px)
            bt.mark(s_state, close[t + 1])
            bt.mark(t_state, close[t + 1])
            asset_ret = float(np.log(close[t + 1] / close[t]))
            shadow.observe_return(asset_ret)
            treated.observe_return(asset_ret)
            vis = treated_pool.visible(date, cfg.retrieval.recency_days, self.ds.dates)
            vis_syn = sum(1 for it, _ in vis if it.provenance != "clean") / max(len(vis), 1)
            s_steps.append(self._rec(d_s, f_s, s_state, eq_s0, exec_px, 0.0, None, []))
            t_steps.append(self._rec(d_t, f_t, t_state, eq_t0, exec_px, vis_syn, bias_name, [treated_pool.get(i).strategy for i, p in zip(d_t.retrieved_ids, d_t.retrieved_provenance) if p != "clean"]))
            t_steps[-1]["cf_action"] = d_cf.action.value if d_cf is not None else None
            if spec.injection == "misgen":  # only the adaptive attacker learns from feedback
                retrieved_syn = [treated_pool.get(i) for i, p in zip(d_t.retrieved_ids, d_t.retrieved_provenance) if p != "clean"]
                misgen.update_history(d_t.action != d_s.action, retrieved_syn)
        metrics = compute_metrics(t_steps, s_steps, cfg.backtest.initial_cash)
        return RunResult(spec, self.seed, metrics, t_steps, s_steps, misgen.summary() if misgen else None, recorded)

    @staticmethod
    def _rec(d, f, state: PortfolioState, eq0: float, exec_px: float, vis_syn: float, bias: str | None, strategies: list) -> dict:
        return {
            "date": d.date, "action": d.action.value, "confidence": d.confidence, "raw_signal": d.raw_signal, "text_signal": d.text_signal,
            "market_signal": d.market_signal, "consistency": d.consistency, "target_position": d.target_position, "executed_price": exec_px,
            "fill_price": f.price, "shares_delta": f.shares_delta, "turnover": f.turnover, "cost_paid": f.fee_paid,
            "position": state.shares, "cash": state.cash, "equity": state.equity, "daily_return": state.equity / eq0 - 1.0 if eq0 > 0 else 0.0,
            "retrieved_ids": d.retrieved_ids, "retrieved_provenance": d.retrieved_provenance, "retrieved_strategies": strategies,
            "visible_synthetic_frac": vis_syn, "bias": bias, "risk_overrides": d.risk_overrides, "rationale": d.rationale,
            "parse_failed": d.parse_failed, "filter_visible": d.filter_visible, "filter_dropped": d.filter_dropped,
        }


# ---------------------------------------------------------------------------
# standard arms
# ---------------------------------------------------------------------------
def defence_levels(micro_default: bool = True) -> list[tuple[str, bool, bool, bool]]:
    """(name, grounded, risk, micro) for the three defence levels of the main experiment."""
    return [("none", False, False, False), ("grounded", True, False, micro_default), ("grounded_risk", True, True, micro_default)]


def standard_arms(micro_default: bool = True) -> list[RunSpec]:
    """Three defence levels x three injection modes (clean / misgen / placebo)."""
    arms: list[RunSpec] = []
    for dname, grounded, risk, micro in defence_levels(micro_default):
        for inj in ("none", "misgen", "placebo"):
            label = {"none": "clean", "misgen": "injected", "placebo": "placebo"}[inj]
            arms.append(RunSpec(name=f"{label}__{dname}", injection=inj, grounded=grounded, risk=risk, micro=micro))
    return arms


def frozen_arms(micro_default: bool = True) -> list[RunSpec]:
    """Each defence level against the tape recorded from TAPE_SOURCE_ARM (frozen attack).

    frozen__none replays the tape against the agent it was recorded from, so it must
    reproduce injected__none exactly; run_pipeline checks this on every seed.
    """
    return [RunSpec(name=f"frozen__{d}", injection="replay", grounded=g, risk=r, micro=m) for d, g, r, m in defence_levels(micro_default)]


def matched_placebo_arms(micro_default: bool = True) -> list[RunSpec]:
    """Count-matched placebo, with and without the attack's bias, per defence level (tape of TAPE_SOURCE_ARM)."""
    return [RunSpec(name=f"{inj}__{d}", injection=inj, grounded=g, risk=r, micro=m)
            for d, g, r, m in defence_levels(micro_default) for inj in ("placebo_matched", "placebo_bias")]


TREATED_PREFIXES = ("injected__", "frozen__")  # arms whose placebo partners are placebo*__<suffix>

# derived column -> (metric, partner prefix): column = metric(treated) - metric(partner arm of the same seed)
NET_COLUMNS = {
    "ASR_net": ("ASR", "placebo__"),  # classic net ASR (Eq. 9 minus the paraphrase noise floor)
    "signal_deviation_net": ("signal_deviation", "placebo__"),
    "ASR_active_net": ("ASR_active", "placebo__"),
    "ASR_cf_net": ("ASR_cf", "placebo__"),
    "ASR_net_matched": ("ASR", "placebo_bias__"),  # meaning only: same items, count and bias as the attack
    "ASR_cf_net_matched": ("ASR_cf", "placebo_bias__"),
}
# Partners built from the TAPE_SOURCE_ARM tape: they are matched controls only for arms that face that tape.
TAPE_PARTNER_PREFIXES = ("placebo_matched__", "placebo_bias__")


def faces_tape(arm: str) -> bool:
    """True if the arm's attack is exactly the recorded tape: every frozen arm and the arm the tape came from.

    An adaptive attack on a defended agent (injected__grounded, ...) learns its own strategy mix, so it
    injects other items and bias types than the tape; a tape-matched placebo is not its matched control.
    """
    return arm.startswith("frozen__") or arm == TAPE_SOURCE_ARM


def attach_net_asr(rows: list[dict]) -> list[dict]:
    """Add the NET_COLUMNS per (seed, defence) onto injected and frozen rows; NaN elsewhere or if a partner is missing.

    The tape-matched columns (``*_net_matched``) are only filled for arms that face the tape (see ``faces_tape``).
    """
    by = {(r["seed"], r["arm"]): r for r in rows}
    for r in rows:
        prefix = next((p for p in TREATED_PREFIXES if r["arm"].startswith(p)), None)
        for col, (metric, partner_prefix) in NET_COLUMNS.items():
            partner = None
            if prefix is not None and (partner_prefix not in TAPE_PARTNER_PREFIXES or faces_tape(r["arm"])):
                partner = by.get((r["seed"], partner_prefix + r["arm"][len(prefix):]))
            if partner is not None and metric in r and metric in partner:
                r[col] = r[metric] - partner[metric]
            else:
                r.setdefault(col, float("nan"))
    return rows
