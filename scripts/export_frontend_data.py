#!/usr/bin/env python
"""Export the data behind the animated frontend (frontend/index.html).

Writes frontend/data.js as ``window.RHAL_DATA = {...}`` (a script, not JSON, so the
page also works when opened straight from disk):
  * replay  -- one deterministic seed replayed day by day for every arm the
               "with vs without" scenarios need. The grounded arms replay the attack
               recorded against the undefended agent, so those panels face identical
               documents and bias contexts.
  * results -- headline paired comparisons from the latest report run in outputs/.

SAFETY: only actions, signals, provenance labels and perturbation TYPES are exported,
never document text -- generated misinformation is never published (CLAUDE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from _common import ROOT, base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.experiment import RunSpec, Simulation, prepare_dataset, train_detector
from rhal.misgen import DetectabilityModel

# replay arm -> how to run it (tape=True: replay the undefended agent's recorded attack)
REPLAY_ARMS = {
    "clean": (RunSpec("clean__none", "none"), False),
    "attack": (RunSpec("injected__none", "misgen"), False),
    "placebo": (RunSpec("placebo__none", "placebo"), False),
    "grounded": (RunSpec("frozen__grounded", "replay", grounded=True, micro=True), True),
    "grounded_risk": (RunSpec("frozen__grounded_risk", "replay", grounded=True, risk=True, micro=True), True),
}
SHORT = {"clean": "c", "synthetic": "s", "placebo": "p"}


def _docs(step: dict) -> list[dict]:
    """Retrieved docs in rank order as {k: c|s|p, m: minor type, r: rewrite style} -- labels only."""
    strategies = iter(step.get("retrieved_strategies") or [])
    out = []
    for prov in step["retrieved_provenance"]:
        doc = {"k": SHORT.get(prov, "c")}
        if prov == "synthetic":
            st = next(strategies, None) or {}
            doc["m"], doc["r"] = st.get("minor"), st.get("rewrite")
        elif prov == "placebo":
            next(strategies, None)
        out.append(doc)
    return out


def _steps(treated: list[dict], shadow: list[dict], close: dict[str, float], cash: float) -> list[dict]:
    rows = []
    for a, b in zip(treated, shadow):
        rows.append({
            "d": a["date"], "px": round(close[a["date"]], 4),
            "a": a["action"], "s": b["action"],
            "sig": round(a["raw_signal"], 4), "ts": round(a["text_signal"], 4), "ms": round(a["market_signal"], 4), "c": round(a["consistency"], 4),
            "eq": round(a["equity"] / cash, 5), "seq": round(b["equity"] / cash, 5),
            "docs": _docs(a), "bias": a.get("bias"), "ov": [o.split("(")[0] for o in a["risk_overrides"]],
        })
    return rows


def replay(cfg, seed: int) -> dict:
    ds = prepare_dataset(cfg, seed)
    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    det = DetectabilityModel.load(det_path) if det_path.exists() else train_detector(cfg, cfg.seed)
    sim = Simulation(cfg, ds, det, seed)
    close = dict(zip(ds.dates, ds.prices["close"].astype(float)))
    source = sim.run(REPLAY_ARMS["attack"][0])
    arms = {}
    for key, (spec, needs_tape) in REPLAY_ARMS.items():
        r = source if key == "attack" else sim.run(spec, tape=source.tape if needs_tape else None)
        arms[key] = _steps(r.treated_steps, r.shadow_steps, close, cfg.backtest.initial_cash)
    return {"seed": seed, "threshold": cfg.agent.action_threshold, "days": len(arms["clean"]), "arms": arms}


def _cmp(st: pd.DataFrame | None, key: str) -> dict | None:
    if st is None or key not in st.index:
        return None
    r = st.loc[key]
    return {"mean": round(100 * r["mean_diff"], 1), "lo": round(100 * r["ci_lo"], 1), "hi": round(100 * r["ci_hi"], 1),
            "p": float(r["p_holm"]), "sig": bool(r["significant"])}


def results(root: Path) -> dict | None:
    summary = root / "pipeline" / "pipeline_summary.csv"
    if not summary.exists():
        return None
    agg = pd.read_csv(summary).set_index("arm")

    def stats(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path).set_index("key") if path.exists() else None

    main, controls = stats(root / "pipeline" / "pipeline_stats.csv"), stats(root / "pipeline" / "pipeline_stats_controls.csv")
    defense, kappa = stats(root / "ablation/defense/ablation_defense_stats.csv"), stats(root / "ablation/grounding_strength/ablation_grounding_strength_stats.csv")

    def mean(arm: str, col: str) -> float | None:
        return round(100 * float(agg.loc[arm, f"{col}_mean"]), 1) if arm in agg.index else None

    abl_seeds = None
    abl_summary = root / "ablation/defense/ablation_defense_summary.csv"
    if abl_summary.exists():
        abl_seeds = int(pd.read_csv(abl_summary)["n_seeds"].max())
    return {
        "n_seeds": int(agg["n_seeds"].max()), "ablation_seeds": abl_seeds,
        "asr": {"naive": mean("injected__none", "ASR"), "placebo": mean("placebo__none", "ASR"), "grounded": mean("injected__grounded", "ASR"),
                "grounded_risk": mean("injected__grounded_risk", "ASR")},
        "net": {d: _cmp(main, f"net_{d}") for d in ("none", "grounded", "grounded_risk")},
        "decomposition": {"count": _cmp(controls, "dec_count_none"), "bias": _cmp(controls, "dec_bias_none"), "meaning": _cmp(controls, "dec_meaning_none")},
        "gain": {"grounding_adaptive": _cmp(main, "gain_adaptive_grounded_net"), "grounding_frozen": _cmp(main, "gain_frozen_grounded_net"),
                 "risk_on_top": _cmp(main, "gain_adaptive_risk_net"), "cf_grounding": _cmp(controls, "cf_gain_grounded")},
        "cost": {"grounding": _cmp(main, "cost_grounded"), "clean_cr_none": mean("clean__none", "CR"), "clean_cr_grounded": mean("clean__grounded", "CR")},
        "drift": _cmp(controls, "cf_gap_none"),
        "microstructure": _cmp(defense, "abl_def_adaptive_micro"),
        "kappa": _cmp(kappa, "abl_kappa_adaptive_0.3_0.7"),
    }


def main() -> None:
    p = base_parser("Export data for the animated frontend")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    data = {"replay": replay(cfg, cfg.seed), "results": results(Path(cfg.output_dir))}
    out = ROOT / "frontend" / "data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("// AUTO-GENERATED by scripts/export_frontend_data.py -- do not edit. Labels only, no document text.\n"
                   "window.RHAL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {data['replay']['days']} days x {len(data['replay']['arms'])} arms; "
          f"results from {data['results']['n_seeds'] if data['results'] else 'no'} seeds")


if __name__ == "__main__":
    main()
