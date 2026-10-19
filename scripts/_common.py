"""Shared CLI helpers for scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--seeds", type=int, default=None, help="override experiment.n_seeds")
    p.add_argument("--out", default=None, help="override output_dir")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set grounding.strength=0.5",
    )
    return p


def parse_overrides(pairs: list[str]) -> dict:
    import yaml

    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
