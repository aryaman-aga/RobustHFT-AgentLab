"""Tables and figures from run results (matplotlib Agg; no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils.io import ensure_dir, save_csv  # noqa: E402
from .stats import bootstrap_ci  # noqa: E402

KEY_COLS = ["MER", "MER_lift", "ASR", "ASR_net", "ASR_active", "ASR_active_net", "ASR_given_exposed", "exposure_rate", "signal_deviation", "CR", "CR_clean_paired", "max_drawdown", "sharpe", "turnover", "hold_rate"]


def aggregate(rows: list[dict], by: str = "arm") -> pd.DataFrame:
    """Per-arm mean, std and 95% bootstrap CI of the mean (over seeds) for every KEY_COL."""
    df = pd.DataFrame(rows)
    cols = [c for c in KEY_COLS if c in df.columns]
    grp = df.groupby(by)
    g = grp[cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=0).add_suffix("_std")
    ci = {c: grp[c].agg(lambda s: bootstrap_ci(s.to_numpy())) for c in cols}
    lo = pd.DataFrame({f"{c}_ci_lo": ci[c].map(lambda t: t[0]) for c in cols})
    hi = pd.DataFrame({f"{c}_ci_hi": ci[c].map(lambda t: t[1]) for c in cols})
    out = pd.concat([mean, std, lo, hi], axis=1)
    out["n_seeds"] = g.size()
    order = [f"{c}_{s}" for c in cols for s in ("mean", "std", "ci_lo", "ci_hi")] + ["n_seeds"]
    return out[order].reset_index()


def _ci_err(agg: pd.DataFrame, metric: str, scale: float) -> np.ndarray | pd.Series:
    """Asymmetric error bars (2 x n) from the CI columns, falling back to +/- std for old summaries."""
    m = agg[f"{metric}_mean"]
    if f"{metric}_ci_lo" in agg.columns:
        return np.vstack([(m - agg[f"{metric}_ci_lo"]).clip(lower=0), (agg[f"{metric}_ci_hi"] - m).clip(lower=0)]) * scale
    return agg[f"{metric}_std"] * scale


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
    """Bar per arm with 95% bootstrap-CI error bars."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    scale = 100 if pct else 1
    ax.bar(agg["arm"], agg[f"{metric}_mean"] * scale, yerr=_ci_err(agg, metric, scale), capsize=3, color="#4C72B0")
    ax.set_ylabel(ylabel or (f"{metric} (%)" if pct else metric))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_frontier(agg: pd.DataFrame, path: Path) -> None:
    """Robustness-return frontier: ASR (x) vs paired clean CR (y) per defence level, 95% CIs.

    Filled circles: adaptive attacker (labelled); hollow squares: frozen attack (same documents for
    every defence), joined to the adaptive point of the same defence by a dotted line.
    frozen__none is skipped: it replays the attack against its own source agent, so it equals injected__none.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    inj = agg[agg["arm"].str.startswith("injected__")]
    frz = agg[agg["arm"].str.startswith("frozen__") & (agg["arm"] != "frozen__none")]
    for sub, style, label in ((inj, dict(fmt="o"), "adaptive attack"), (frz, dict(fmt="s", mfc="none"), "frozen attack")):
        if not sub.empty:
            ax.errorbar(100 * sub["ASR_mean"], 100 * sub["CR_clean_paired_mean"], xerr=_ci_err(sub, "ASR", 100), yerr=_ci_err(sub, "CR_clean_paired", 100),
                        capsize=3, label=label, **style)
    pos = {r["arm"]: (100 * r["ASR_mean"], 100 * r["CR_clean_paired_mean"]) for _, r in pd.concat([inj, frz]).iterrows()}
    for _, r in inj.iterrows():
        ax.annotate(r["arm"].replace("injected__", ""), pos[r["arm"]], textcoords="offset points", xytext=(5, 5), fontsize=8)
    for _, r in frz.iterrows():
        partner = r["arm"].replace("frozen__", "injected__")
        if partner in pos:
            (x0, y0), (x1, y1) = pos[partner], pos[r["arm"]]
            ax.plot([x0, x1], [y0, y1], ls=":", color="grey", lw=1)
    ax.set_xlabel("ASR under attack (%)  ← more robust")
    ax.set_ylabel("clean cumulative return (%)")
    ax.set_title("Robustness–return frontier")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_exposure_by_type(metrics: list[dict], path: Path) -> None:
    """Sum exposure and flips by Phi_Minor type over the given runs' metrics dicts."""
    exp: dict[str, int] = {}
    flp: dict[str, int] = {}
    for m in metrics:
        for k, v in m.get("exposure_by_type", {}).items():
            exp[k] = exp.get(k, 0) + v
        for k, v in m.get("flips_by_type", {}).items():
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
