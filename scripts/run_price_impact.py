#!/usr/bin/env python
"""Price-impact stress test: grounding when the misinformation also moves the market.

Per seed, the adaptive attack against the undefended agent is recorded (injected__none, the
same run as in the pipeline). For each gamma the tape defines a moved price path
(rhal/evaluation/impact.py); on that one path we run the clean arm, the three placebo
controls and the frozen replay of the tape under every defense level. Both twins trade the
moved prices; only the treated agent reads the fake text. gamma = 0 reproduces the
pipeline's frozen arms exactly.

Arms are named <kind>__<level>@g<gamma>. Per seed and gamma we also add a derived row
gain@g<gamma> (metric gain_net = net ASR of frozen__none - net ASR of frozen__grounded),
so the change in grounding's gain across gammas is a paired comparison.
Outputs under <output_dir>/price_impact/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.experiment import (
    TAPE_INJECTIONS,
    TAPE_SOURCE_ARM,
    RunSpec,
    Simulation,
    attach_net_asr,
    defence_levels,
    prepare_dataset,
    train_detector,
)
from rhal.evaluation.impact import impacted_dataset, sentiment_shifts
from rhal.evaluation.report import write_tables
from rhal.misgen import DetectabilityModel
from rhal.utils.io import ensure_dir, save_json
from rhal.utils.logging import get_logger

log = get_logger("price_impact")
GAMMAS = (0.0, 1.0, 2.0)  # daily sigmas of price move per unit of summed sentiment shift
KINDS = (
    ("clean", "none"),
    ("placebo", "placebo"),
    ("placebo_matched", "placebo_matched"),
    ("placebo_bias", "placebo_bias"),
    ("frozen", "replay"),
)


def main() -> None:
    p = base_parser("Price-impact stress test (frozen attack)")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    n_seeds = a.seeds or cfg.experiment.n_seeds
    out = ensure_dir(Path(cfg.output_dir) / "price_impact")
    det_path = Path(cfg.output_dir) / "models" / "detector.pkl"
    det = DetectabilityModel.load(det_path) if det_path.exists() else train_detector(cfg, cfg.seed)
    rows: list[dict] = []
    shifts: list[float] = []  # |summed sentiment shift| per attack day: how hard the stress test pushes
    t_all = time.time()
    for i in range(n_seeds):
        seed = cfg.seed + i
        ds = prepare_dataset(cfg, seed)
        tape = Simulation(cfg, ds, det, seed).run(RunSpec(TAPE_SOURCE_ARM, "misgen")).tape
        shifts += [abs(v) for v in sentiment_shifts(ds, tape).values()]
        for g in GAMMAS:
            sim = Simulation(cfg, impacted_dataset(ds, tape, g, cfg), det, seed)
            for level, grounded, risk, micro in defence_levels():
                for kind, inj in KINDS:
                    r = sim.run(
                        RunSpec(f"{kind}__{level}@g{g}", inj, grounded, risk, micro),
                        tape=tape if inj in TAPE_INJECTIONS else None,
                    )
                    rows.append(r.to_row())
        log.info(f"seed={seed} done ({time.time() - t_all:.0f}s)")
    rows = attach_net_asr(rows)
    df = pd.DataFrame(rows)
    net = df.set_index(["seed", "arm"])["ASR_net"]
    for seed in sorted(df["seed"].unique()):
        for g in GAMMAS:
            gain = net[(seed, f"frozen__none@g{g}")] - net[(seed, f"frozen__grounded@g{g}")]
            rows.append({"arm": f"gain@g{g}", "seed": seed, "gain_net": gain})
    write_tables(rows, out, "price_impact")  # the runs CSV includes the derived gain rows
    save_json(
        {
            "gammas": list(GAMMAS),
            "attack_days": len(shifts),
            "abs_shift_median": float(np.median(shifts)),
            "abs_shift_p90": float(np.quantile(shifts, 0.9)),
        },
        out / "impact_meta.json",
    )
    log.info(f"{len(rows)} rows in {time.time() - t_all:.1f}s -> {out}")


if __name__ == "__main__":
    main()
