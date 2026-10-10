"""Tables and figures from run results (matplotlib Agg; no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils.io import ensure_dir, save_csv  # noqa: E402

KEY_COLS = ["MER", "MER_lift", "ASR", "ASR_net", "ASR_active", "ASR_active_net", "ASR_given_exposed", "exposure_rate", "signal_deviation", "CR", "CR_clean_paired", "max_drawdown", "sharpe", "turnover", "hold_rate"]


def aggregate(rows: list[dict], by: str = "arm") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = [c for c in KEY_COLS if c in df.columns]
    g = df.groupby(by)[cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=0).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    out["n_seeds"] = g.size()
    order = [c for pair in zip(mean.columns, std.columns) for c in pair] + ["n_seeds"]
    return out[order].reset_index()


def to_markdown(agg: pd.DataFrame, cols: list[str] | None = None, pct: tuple[str, ...] = ("MER", "ASR", "ASR_net", "ASR_active", "ASR_active_net", "ASR_given_exposed", "exposure_rate", "CR", "CR_clean_paired", "max_drawdown", "hold_rate")) -> str:
    cols = cols or [c for c in KEY_COLS if f"{c}_mean" in agg.columns]
    hdr = "| arm | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [hdr, sep]
    for _, r in agg.iterrows():
        cells = []
        for c in cols:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            cells.append(f"{100*m:.2f} ± {100*s:.2f}" if c in pct else f"{m:.3f} ± {s:.3f}")
        lines.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_equity(results: dict[str, list[dict]], initial_cash: float, path: Path, title: str = "Equity curves (seed 0)") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, steps in results.items():
        eq = [initial_cash] + [s["equity"] for s in steps]
        ax.plot(range(len(eq)), np.array(eq) / initial_cash, label=name, lw=1.4)
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("trading day")
    ax.set_ylabel("equity / initial")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bars(agg: pd.DataFrame, metric: str, path: Path, ylabel: str | None = None, pct: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    m = agg[f"{metric}_mean"] * (100 if pct else 1)
    s = agg[f"{metric}_std"] * (100 if pct else 1)
    ax.bar(agg["arm"], m, yerr=s, capsize=3, color="#4C72B0")
    ax.set_ylabel(ylabel or (f"{metric} (%)" if pct else metric))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_frontier(agg: pd.DataFrame, path: Path) -> None:
    """Robustness-return frontier: ASR (x) vs paired clean CR (y) per defence arm."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    inj = agg[agg["arm"].str.startswith("injected")]
    for _, r in inj.iterrows():
        ax.errorbar(100 * r["ASR_mean"], 100 * r["CR_clean_paired_mean"], xerr=100 * r["ASR_std"], yerr=100 * r["CR_clean_paired_std"], fmt="o", capsize=3)
        ax.annotate(r["arm"].replace("injected__", ""), (100 * r["ASR_mean"], 100 * r["CR_clean_paired_mean"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("ASR under attack (%)  ← more robust")
    ax.set_ylabel("clean cumulative return (%)")
    ax.set_title("Robustness–return frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_exposure_by_type(results: list, path: Path) -> None:
    exp: dict[str, int] = {}
    flp: dict[str, int] = {}
    for r in results:
        for k, v in r.metrics.get("exposure_by_type", {}).items():
            exp[k] = exp.get(k, 0) + v
        for k, v in r.metrics.get("flips_by_type", {}).items():
            flp[k] = flp.get(k, 0) + v
    if not exp:
        return
    keys = sorted(exp, key=lambda k: -exp[k])
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, [exp[k] for k in keys], 0.4, label="retrieved")
    ax.bar(x + 0.2, [flp.get(k, 0) for k in keys], 0.4, label="retrieved on flipped step")
    ax.set_xticks(x, keys, rotation=30, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Exposure and decision flips by perturbation type")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_tables(rows: list[dict], out_dir: Path, prefix: str) -> pd.DataFrame:
    ensure_dir(out_dir)
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / f"{prefix}_runs.csv")
    agg = aggregate(rows)
    save_csv(agg, out_dir / f"{prefix}_summary.csv")
    (out_dir / f"{prefix}_summary.md").write_text(to_markdown(agg) + "\n", encoding="utf-8")
    return agg
