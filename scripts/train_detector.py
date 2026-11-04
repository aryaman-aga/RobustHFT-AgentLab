#!/usr/bin/env python
"""Train the detectability proxy A_det and save it to outputs/models/detector.pkl.

The training corpus is an independent synthetic corpus (never the evaluation
corpus). Positives are Phi_Minor perturbations only; the threshold is
calibrated to a target false-positive rate on held-out clean text.
"""

from __future__ import annotations

import json
from pathlib import Path

from _common import base_parser, parse_overrides

from rhal.config import load_config
from rhal.data.synth import SynthConfig, generate_synthetic_dataset
from rhal.evaluation.experiment import train_detector
from rhal.misgen.operators import MINOR_TYPES, apply_minor, apply_rewrite
from rhal.seeding import rng


def main() -> None:
    p = base_parser("Train detectability proxy")
    p.add_argument("--target-fpr", type=float, default=0.10)
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    det = train_detector(cfg, cfg.seed, target_fpr=a.target_fpr)
    out = Path(a.out or cfg.output_dir) / "models"
    det.save(out / "detector.pkl")
    # per-type evaluation on a fresh corpus (reported in the paper's appendix)
    d = generate_synthetic_dataset(SynthConfig(n_days=150, news_per_day=cfg.data.news_per_day), 9999)
    texts = d["news"]["text"].tolist()[:250]
    evals = {"clean_fpr": float((det.predict_proba(texts) > det.threshold).mean())}
    for m in MINOR_TYPES:
        g = rng(11, m)
        cands = [apply_minor(t, m, g) for t in texts]
        evals[m] = {
            "minor": float((det.predict_proba(cands) > det.threshold).mean()),
            "news_style": float(
                (
                    det.predict_proba([apply_rewrite(c, "news_style", rng(12, i)) for i, c in enumerate(cands)])
                    > det.threshold
                ).mean()
            ),
            "academic_style": float(
                (
                    det.predict_proba([apply_rewrite(c, "academic_style", rng(13, i)) for i, c in enumerate(cands)])
                    > det.threshold
                ).mean()
            ),
        }
    meta = {**det.meta, "per_type_detection_rate": evals}
    (out / "detector_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / 'detector.pkl'}")
    print(
        f"train_acc={det.meta['train_acc']:.3f}  threshold={det.threshold:.3f}  holdout_clean_fpr={det.meta['holdout_clean_fpr_at_threshold']:.3f}"
    )
    print("detection rate by perturbation type (minor / after news rewrite / after academic rewrite):")
    for m in MINOR_TYPES:
        e = evals[m]
        print(f"  {m:10s} {e['minor']:.2f} / {e['news_style']:.2f} / {e['academic_style']:.2f}")


if __name__ == "__main__":
    main()
