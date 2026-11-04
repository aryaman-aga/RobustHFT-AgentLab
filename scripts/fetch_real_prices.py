#!/usr/bin/env python
"""Download a real daily OHLCV history ONCE into data/real/ (git-ignored).

yfinance is the project's only allowed network path (CLAUDE.md); everything downstream reads
the saved CSV offline. Only prices are downloaded: experiments on real prices pair them with
template news about the fictional company (data.source=real), so no text about a real asset is
ever generated.

Usage:
  python scripts/fetch_real_prices.py                        # BTC-USD, full history -> data/real/btc_usd.csv
  python scripts/fetch_real_prices.py --symbol SPY --out data/real/spy.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from _common import ROOT

from rhal.data.loaders import download_yfinance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--start", default="2014-01-01", help="first date to request (Yahoo starts BTC-USD in Sep 2014)")
    p.add_argument(
        "--end", default=None, help="exclusive end date (default: today, so today's incomplete bar is left out)"
    )
    p.add_argument("--out", default=str(ROOT / "data" / "real" / "btc_usd.csv"))
    a = p.parse_args()
    df = download_yfinance(
        a.symbol, a.start, a.end or date.today().isoformat()
    )  # validated with repair=True (tiny vendor inconsistencies fixed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {
        "symbol": a.symbol,
        "rows": len(df),
        "first": df["date"].iloc[0],
        "last": df["date"].iloc[-1],
        "downloaded": date.today().isoformat(),
        "source": "Yahoo Finance via yfinance (auto_adjust=True)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {meta}")


if __name__ == "__main__":
    main()
