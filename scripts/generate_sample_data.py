#!/usr/bin/env python
"""Write a small synthetic dataset to data/sample/ (prices.csv, news.csv, intraday.csv).

All data are synthetic. News text is template-generated and labelled clean; it
is *not* real news. Run:  python scripts/generate_sample_data.py
"""
from __future__ import annotations

from _common import ROOT, base_parser, parse_overrides

from rhal.config import load_config
from rhal.data import load_news, load_prices
from rhal.data.synth import SynthConfig, generate_synthetic_dataset
from rhal.seeding import child_seed
from rhal.utils.io import save_csv


def main() -> None:
    p = base_parser("Generate synthetic sample data")
    p.add_argument("--intraday", action="store_true", help="also write intraday.csv (larger)")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    d = generate_synthetic_dataset(
        SynthConfig(n_days=cfg.data.n_days, news_per_day=cfg.data.news_per_day, intraday=a.intraday, bars_per_day=cfg.data.intraday.bars_per_day, symbol=cfg.data.symbol),
        child_seed(cfg.seed, "data"),
    )
    out = ROOT / "data" / "sample"
    save_csv(d["prices"].drop(columns=["regime"]), out / "prices.csv")
    save_csv(d["news"], out / "news.csv")
    if a.intraday:
        save_csv(d["intraday"], out / "intraday.csv")
    # round-trip validation
    pr, nw = load_prices(out / "prices.csv"), load_news(out / "news.csv")
    print(f"wrote {out/'prices.csv'} ({len(pr)} rows) and {out/'news.csv'} ({len(nw)} rows, all provenance=clean)")
    print(f"date range {pr['date'].iloc[0]} .. {pr['date'].iloc[-1]}; close {pr['close'].iloc[0]:.2f} -> {pr['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
