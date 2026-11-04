#!/usr/bin/env python
"""Main experiment: clean vs injected vs placebo under three defence levels, over n seeds.

Per seed it also records the adaptive attack against the undefended agent
(injected__none) and replays it against every defence level (frozen__<level>), so
defences can be compared on bit-identical attacks as well as under an adaptive one.
The same tape drives the matched placebo controls (placebo_matched__<level>,
placebo_bias__<level>) that split ASR into pool noise, count, bias and meaning.

LLM runs (agent.mode=llm or agent.text_scorer=llm) usually select a subset of arms with
--arms/--levels and write llm_stats.json (model, digest, cache hits/misses, parse failures).

Outputs (under <output_dir>/pipeline/):
  pipeline_runs.csv, pipeline_summary.csv, pipeline_summary.md,
  steps_seed0_<arm>.json, misgen_seed0_<arm>.json, llm_stats.json (LLM runs),
  fig_equity.png, fig_asr.png, fig_mer.png, fig_frontier.png, fig_exposure.png
"""

from __future__ import annotations

import time
from pathlib import Path

from _common import base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.experiment import (
    TAPE_INJECTIONS,
    TAPE_SOURCE_ARM,
    Simulation,
    attach_net_asr,
    frozen_arms,
    matched_placebo_arms,
    prepare_dataset,
    standard_arms,
    train_detector,
)
from rhal.evaluation.report import (
    plot_bars,
    plot_equity,
    plot_exposure_by_type,
    plot_frontier,
    to_markdown,
    write_tables,
)
from rhal.misgen import DetectabilityModel
from rhal.utils.io import ensure_dir, save_json
from rhal.utils.logging import get_logger

log = get_logger("pipeline")


def main() -> None:
    p = base_parser("Run the main RobustHFT-AgentLab experiment")
    p.add_argument("--no-micro", action="store_true", help="disable HFT/microstructure features in grounded arms")
    p.add_argument(
        "--arms", default=None, help="comma list of arm kinds to run, e.g. clean,injected,placebo,frozen (default: all)"
    )
    p.add_argument("--levels", default=None, help="comma list of defense levels, e.g. none,grounded (default: all)")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    n_seeds = a.seeds or cfg.experiment.n_seeds
    out = ensure_dir(Path(cfg.output_dir) / "pipeline")
    log.info(
        f"config={a.config} seeds={n_seeds} agent={cfg.agent.mode} reader={cfg.agent.text_scorer} llm={cfg.llm.provider}:{cfg.llm.model}"
    )
    log.info("SAFETY: offline simulation only; all generated text is synthetic and labelled.")

    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    if det_path.exists():
        det = DetectabilityModel.load(det_path)
        log.info(f"loaded detector {det_path} (thr={det.threshold:.3f})")
    else:
        det = train_detector(cfg, cfg.seed)
        det.save(det_path)
        log.info(f"trained detector -> {det_path} (acc={det.meta['train_acc']:.3f}, thr={det.threshold:.3f})")

    # standard arms first: the tape-driven arms replay the tape recorded by TAPE_SOURCE_ARM
    micro = not a.no_micro
    arms = select_arms(
        standard_arms(micro_default=micro)
        + frozen_arms(micro_default=micro)
        + matched_placebo_arms(micro_default=micro),
        a.arms,
        a.levels,
    )
    log.info(f"{len(arms)} arms per seed: {', '.join(s.name for s in arms)}")
    uses_llm = cfg.agent.mode == "llm" or cfg.agent.text_scorer == "llm"
    llm_totals: dict = {}
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
            r = sim.run(spec, tape=by_name[TAPE_SOURCE_ARM].tape if spec.injection in TAPE_INJECTIONS else None)
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
            log.info(
                f"seed={seed} {spec.name:24s} MER={m['MER']:.3f} lift={m['MER_lift']:.2f} ASR={m['ASR']:.3f} CR={m['CR']:+.3f} DD={m['max_drawdown']:+.3f}"
            )
        if uses_llm:
            accumulate_llm_stats(llm_totals, sim.llm_stats())
            save_json(llm_totals, out / "llm_stats.json")
            log.info(f"llm so far: {llm_totals}")
        # replaying a tape against the very agent it was recorded from must change nothing
        if "frozen__none" in by_name and by_name["frozen__none"].metrics != by_name[TAPE_SOURCE_ARM].metrics:
            raise RuntimeError(
                f"seed={seed}: frozen__none does not reproduce {TAPE_SOURCE_ARM}; the replay path is broken"
            )
    rows = attach_net_asr(rows)
    agg = write_tables(rows, out, "pipeline")
    log.info(f"{len(rows)} runs in {time.time() - t_all:.1f}s")

    plot_equity(
        {k: v for k, v in seed0_steps.items() if k.startswith(("clean__none", "injected__"))},
        cfg.backtest.initial_cash,
        out / "fig_equity.png",
    )
    plot_bars(agg, "ASR", out / "fig_asr.png")
    plot_bars(agg, "MER", out / "fig_mer.png")
    plot_frontier(agg, out / "fig_frontier.png")
    plot_exposure_by_type(attack_metrics, out / "fig_exposure.png")
    print("\n=== Summary (mean ± std over seeds; % where applicable) ===")
    print(to_markdown(agg))
    print(f"\nartifacts in {out}/")


def select_arms(arms: list, kinds: str | None, levels: str | None) -> list:
    """Keep arms whose kind (before '__') and level (after '__') are listed; validate tape needs."""
    keep_k = {k.strip() for k in kinds.split(",")} if kinds else None
    keep_l = {v.strip() for v in levels.split(",")} if levels else None
    out = [
        s
        for s in arms
        if (keep_k is None or s.name.split("__")[0] in keep_k) and (keep_l is None or s.name.split("__")[1] in keep_l)
    ]
    if not out:
        raise SystemExit(f"--arms {kinds!r} --levels {levels!r} selects no arm")
    if any(s.injection in TAPE_INJECTIONS for s in out) and TAPE_SOURCE_ARM not in {s.name for s in out}:
        raise SystemExit(f"tape-driven arms replay {TAPE_SOURCE_ARM}: include arm kind 'injected' and level 'none'")
    return out


def accumulate_llm_stats(tot: dict, st: dict | None) -> None:
    """Sum per-seed counters (cache hits/misses, texts scored, parse failures); keep identity fields."""
    for k, v in (st or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            tot[k] = tot.get(k, 0) + v
        else:
            tot[k] = v


if __name__ == "__main__":
    main()
