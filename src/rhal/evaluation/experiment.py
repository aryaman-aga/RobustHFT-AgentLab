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
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..agent import LLMPolicy, Perceiver, PortfolioState, Profile, RiskLayer, RulePolicy, SurrogatePolicy, TradingAgent
from ..backtest import Backtester
from ..config import Config
from ..data.synth import SynthConfig, generate_synthetic_dataset
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

Injection = Literal["none", "misgen", "placebo", "replay"]

# Arm whose adaptive attack is recorded and replayed against every defence (frozen setting).
TAPE_SOURCE_ARM = "injected__none"


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
    else:
        prices, news, intraday = load_prices(cfg.data.csv_prices), load_news(cfg.data.csv_news), None
    f = cfg.features
    feats = build_feature_frame(
        prices, intraday, vol_window=f.vol_window, drawdown_window=f.drawdown_window, short_trend=f.short_trend, long_trend=f.long_trend,
        anomaly_window=f.anomaly_window, micro=True, rv_window=f.micro.rv_window, volume_window=f.micro.volume_window,
    )
    feats["close"] = prices["close"].to_numpy()
    return Dataset(prices, news, intraday, feats, prices["date"].tolist())


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

    # -- builders ----------------------------------------------------------
    def _agent(self, cfg: Config, spec: RunSpec) -> TradingAgent:
        zc = ["z_" + c for c in feature_columns(spec.micro)]
        retr = Retriever(cfg.retrieval.method, cfg.retrieval.top_k, cfg.retrieval.recency_days, cfg.retrieval.recency_decay, background=self.ds.news["text"].tolist())
        perc = Perceiver(self.ds.feats, retr, self.ds.dates)
        ev = EvidenceBuilder(self.ds.feats, zc, micro=spec.micro, window=cfg.grounding.window, horizon=cfg.grounding.horizon, top_k=cfg.grounding.top_k) if spec.grounded else None
        if cfg.agent.mode == "rule":
            policy = RulePolicy(cfg.agent.action_threshold)
        elif cfg.agent.mode == "llm":
            policy = LLMPolicy(build_client(cfg.llm), SurrogatePolicy(cfg.agent.action_threshold))
        else:
            policy = SurrogatePolicy(cfg.agent.action_threshold)
        r = cfg.risk
        risk = RiskLayer(enabled=spec.risk, max_position=r.max_position, vol_target=r.vol_target, stop_loss=r.stop_loss, take_profit=r.take_profit,
                         hold_on_conflict=r.hold_on_conflict, conflict_threshold=r.conflict_threshold, min_confidence=r.min_confidence,
                         drawdown_throttle=r.drawdown_throttle)
        return TradingAgent(perceiver=perc, evidence=ev, policy=policy, risk=risk, profile=Profile(symbol=cfg.data.symbol), memory_len=cfg.agent.memory_len,
                            grounded=spec.grounded, kappa=cfg.grounding.strength, text_weight=cfg.agent.text_weight, adaptive_trust=cfg.agent.adaptive_trust)

    def _misgen(self, cfg: Config, spec: RunSpec) -> MisGenEngine | None:
        if spec.injection in ("none", "replay"):
            return None
        m = cfg.misgen
        return MisGenEngine(
            # keyed by injection type, NOT spec.name: renaming an arm must never change its results
            seed=child_seed(self.seed, "misgen", spec.injection), background_corpus=self.ds.news["text"].tolist(), detector=self.det,
            bias_prob=m.bias_prob, use_bias=m.use_bias, use_minor=m.use_minor, use_rewrite=m.use_rewrite, use_filter=m.use_filter,
            use_feedback=m.use_feedback, alpha=m.alpha, delta=m.delta, max_tries=m.max_tries, tau_sim=m.sim_threshold, tau_det=m.det_threshold,
            injection_ratio=m.injection_ratio, placebo=(spec.injection == "placebo"),
        )

    # -- main loop ---------------------------------------------------------
    def run(self, spec: RunSpec, tape: AttackTape | None = None) -> RunResult:
        """Run one paired arm. ``tape`` is required for, and only allowed with, injection='replay'."""
        if (spec.injection == "replay") != (tape is not None):
            raise ValueError(f"arm {spec.name!r}: pass a tape exactly when injection='replay'")
        cfg = self.cfg.with_overrides(spec.overrides) if spec.overrides else self.cfg
        bt = Backtester(fee_bps=cfg.backtest.fee_bps, slippage_bps=cfg.backtest.slippage_bps)
        shadow_pool = NewsPool.from_frame(self.ds.news)
        treated_pool = NewsPool.from_frame(self.ds.news)
        shadow, treated = self._agent(cfg, spec), self._agent(cfg, spec)
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
            if misgen is not None:
                cr = t_state.equity / cfg.backtest.initial_cash if t_state.last_price > 0 else 1.0
                bias = misgen.sample_bias(cr, date)
                bias_name = bias.name
                treated.set_bias(bias)
                today_clean = [it for it in shadow_pool.items() if it.date == date]
                injected = misgen.generate_for_items(today_clean, date)
                treated_pool.extend(injected)
                if recorded is not None:
                    recorded.docs[date], recorded.bias[date] = injected, bias_name
            elif tape is not None:  # frozen attack: replay exactly what the source run injected
                bias_name = tape.bias.get(date)
                treated.set_bias(bias_config(bias_name))
                treated_pool.extend(tape.docs.get(date, []))
            for st in (s_state, t_state):
                bt.mark(st, close[t])
            eq_s0, eq_t0 = s_state.equity, t_state.equity
            d_s = shadow.step(t, shadow_pool, s_state)
            d_t = treated.step(t, treated_pool, t_state)
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
            if misgen is not None:
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


TREATED_PREFIXES = ("injected__", "frozen__")  # arms whose placebo partner is placebo__<suffix>


def attach_net_asr(rows: list[dict]) -> list[dict]:
    """Add ASR_net = ASR(treated) - ASR(placebo) per (seed, defence) onto injected and frozen rows."""
    by = {(r["seed"], r["arm"]): r for r in rows}
    for r in rows:
        prefix = next((p for p in TREATED_PREFIXES if r["arm"].startswith(p)), None)
        if prefix is not None:
            plc = by.get((r["seed"], "placebo__" + r["arm"][len(prefix):]))
            r["ASR_net"] = r["ASR"] - plc["ASR"] if plc else float("nan")
            r["signal_deviation_net"] = r["signal_deviation"] - plc["signal_deviation"] if plc else float("nan")
            r["ASR_active_net"] = r["ASR_active"] - plc["ASR_active"] if plc else float("nan")
        else:
            r.setdefault("ASR_net", float("nan"))
            r.setdefault("signal_deviation_net", float("nan"))
            r.setdefault("ASR_active_net", float("nan"))
    return rows
