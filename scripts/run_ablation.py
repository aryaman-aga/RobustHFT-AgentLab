#!/usr/bin/env python
"""Ablation matrix runner.

Groups (configs/ablation.yaml):
  attack   -- paper Table 2 style: remove one MisGen component; vulnerable (undefended) agent.
  defense  -- our extension: which defensive component buys robustness, at what return cost.
  retrieval / grounding_strength -- sensitivity sweeps.
Outputs under <output_dir>/ablation/<group>/.
"""
from __future__ import annotations

import time
from pathlib import Path

from _common import ROOT, base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.experiment import RunSpec, Simulation, attach_net_asr, prepare_dataset, train_detector
from rhal.evaluation.report import plot_bars, to_markdown, write_tables
from rhal.misgen import DetectabilityModel
from rhal.utils.io import ensure_dir, load_yaml
from rhal.utils.logging import get_logger

log = get_logger("ablation")


def specs_for(group: str, name: str, ov: dict, cfg) -> list[RunSpec]:
    """Translate a dotted-override dict into paired RunSpecs for the group."""
    g = ov.get("grounding.enabled", cfg.grounding.enabled)
    r = ov.get("risk.enabled", cfg.risk.enabled)
    m = ov.get("features.micro.enabled", cfg.features.micro.enabled)
    if group == "attack":  # undefended agent, as in the paper's ablation
        return [RunSpec(f"injected__{name}", "misgen", False, False, False, ov), RunSpec(f"placebo__{name}", "placebo", False, False, False, ov)]
    if group in ("defense", "grounding_strength"):
        if group == "grounding_strength":
            g, r, m = True, True, True
        return [RunSpec(f"injected__{name}", "misgen", g, r, m, ov), RunSpec(f"placebo__{name}", "placebo", g, r, m, ov), RunSpec(f"clean__{name}", "none", g, r, m, ov)]
    if group == "retrieval":
        return [RunSpec(f"injected__{name}", "misgen", False, False, False, ov), RunSpec(f"placebo__{name}", "placebo", False, False, False, ov)]
    raise ValueError(group)


def main() -> None:
    p = base_parser("Run ablations")
    p.add_argument("--ablation", default=str(ROOT / "configs" / "ablation.yaml"))
    p.add_argument("--group", default="all", help="attack | defense | retrieval | grounding_strength | all")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    n_seeds = a.seeds or cfg.experiment.n_seeds
    abl = load_yaml(a.ablation)["ablations"]
    groups = list(abl) if a.group == "all" else [a.group]
    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    det = DetectabilityModel.load(det_path) if det_path.exists() else train_detector(cfg, cfg.seed)
    if not det_path.exists():
        det.save(det_path)
    datasets = {}
    for group in groups:
        out = ensure_dir(Path(cfg.output_dir) / "ablation" / group)
        rows: list[dict] = []
        t0 = time.time()
        for i in range(n_seeds):
            seed = cfg.seed + i
            if seed not in datasets:
                datasets[seed] = prepare_dataset(cfg, seed)
            sim = Simulation(cfg, datasets[seed], det, seed)
            for name, ov in abl[group].items():
                for spec in specs_for(group, name, dict(ov or {}), cfg):
                    r = sim.run(spec)
                    rows.append(r.to_row())
                    m = r.metrics
                    log.info(f"[{group}] seed={seed} {spec.name:28s} MER={m['MER']:.3f} ASR={m['ASR']:.3f} CR={m['CR']:+.3f}")
        rows = attach_net_asr(rows)
        agg = write_tables(rows, out, f"ablation_{group}")
        plot_bars(agg, "ASR", out / "fig_asr.png")
        print(f"\n=== Ablation group: {group} ({time.time()-t0:.1f}s) ===")
        print(to_markdown(agg))


if __name__ == "__main__":
    main()
