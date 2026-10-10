#!/usr/bin/env python
"""Main experiment: clean vs injected vs placebo under three defence levels, over n seeds.

Per seed it also records the adaptive attack against the undefended agent
(injected__none) and replays it against every defence level (frozen__<level>), so
defences can be compared on bit-identical attacks as well as under an adaptive one.

Outputs (under <output_dir>/pipeline/):
  pipeline_runs.csv, pipeline_summary.csv, pipeline_summary.md,
  steps_seed0_<arm>.json, misgen_seed0_<arm>.json,
  fig_equity.png, fig_asr.png, fig_mer.png, fig_frontier.png, fig_exposure.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from _common import base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.experiment import TAPE_SOURCE_ARM, Simulation, attach_net_asr, frozen_arms, prepare_dataset, standard_arms, train_detector
from rhal.evaluation.report import plot_bars, plot_equity, plot_exposure_by_type, plot_frontier, to_markdown, write_tables
from rhal.misgen import DetectabilityModel
from rhal.utils.io import ensure_dir, save_json
from rhal.utils.logging import get_logger

log = get_logger("pipeline")


def main() -> None:
    p = base_parser("Run the main RobustHFT-AgentLab experiment")
    p.add_argument("--no-micro", action="store_true", help="disable HFT/microstructure features in grounded arms")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    n_seeds = a.seeds or cfg.experiment.n_seeds
    out = ensure_dir(Path(cfg.output_dir) / "pipeline")
    log.info(f"config={a.config} seeds={n_seeds} agent={cfg.agent.mode} llm={cfg.llm.provider} misgen={cfg.misgen.mode}")
    log.info("SAFETY: offline simulation only; all generated text is synthetic and labelled.")

    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    if det_path.exists():
        det = DetectabilityModel.load(det_path)
        log.info(f"loaded detector {det_path} (thr={det.threshold:.3f})")
    else:
        det = train_detector(cfg, cfg.seed)
        det.save(det_path)
        log.info(f"trained detector -> {det_path} (acc={det.meta['train_acc']:.3f}, thr={det.threshold:.3f})")

    # standard arms first: the frozen arms replay the tape recorded by TAPE_SOURCE_ARM
    arms = standard_arms(micro_default=not a.no_micro) + frozen_arms(micro_default=not a.no_micro)
    rows: list[dict] = []
    seed0_steps: dict[str, list[dict]] = {}
    attack_metrics: list[dict] = []  # adaptive-arm metrics for the exposure-by-type figure
    t_all = time.time()
    for i in range(n_seeds):
        seed = cfg.seed + i
        ds = prepare_dataset(cfg, seed)
        sim = Simulation(cfg, ds, det, seed)
        by_name = {}
        for spec in arms:
            r = sim.run(spec, tape=by_name[TAPE_SOURCE_ARM].tape if spec.injection == "replay" else None)
            by_name[spec.name] = r
            if spec.injection == "misgen":
                attack_metrics.append(r.metrics)
            rows.append(r.to_row())
            if i == 0:
                seed0_steps[spec.name] = r.treated_steps
                save_json(r.treated_steps, out / f"steps_seed0_{spec.name}.json")
                if r.misgen_summary:
                    save_json(r.misgen_summary, out / f"misgen_seed0_{spec.name}.json")
            m = r.metrics
            log.info(f"seed={seed} {spec.name:24s} MER={m['MER']:.3f} lift={m['MER_lift']:.2f} ASR={m['ASR']:.3f} CR={m['CR']:+.3f} DD={m['max_drawdown']:+.3f}")
        # replaying a tape against the very agent it was recorded from must change nothing
        if "frozen__none" in by_name and by_name["frozen__none"].metrics != by_name[TAPE_SOURCE_ARM].metrics:
            raise RuntimeError(f"seed={seed}: frozen__none does not reproduce {TAPE_SOURCE_ARM}; the replay path is broken")
    rows = attach_net_asr(rows)
    agg = write_tables(rows, out, "pipeline")
    log.info(f"{len(rows)} runs in {time.time()-t_all:.1f}s")

    plot_equity({k: v for k, v in seed0_steps.items() if k.startswith(("clean__none", "injected__"))}, cfg.backtest.initial_cash, out / "fig_equity.png")
    plot_bars(agg, "ASR", out / "fig_asr.png")
    plot_bars(agg, "MER", out / "fig_mer.png")
    plot_frontier(agg, out / "fig_frontier.png")
    plot_exposure_by_type(attack_metrics, out / "fig_exposure.png")
    print("\n=== Summary (mean ± std over seeds; % where applicable) ===")
    print(to_markdown(agg))
    print(f"\nartifacts in {out}/")


if __name__ == "__main__":
    main()
