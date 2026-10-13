#!/usr/bin/env python
"""Publication figures and compact tables for paper/main.tex.

Reads the per-seed run CSVs in outputs/ and recomputes every paired comparison with the
same definitions as scripts/make_report_tables.py, so the figures, the tables and the
\\val{} macros in the prose always agree.

Writes (vector PDF, embedded TrueType fonts, IEEE column 3.5 in / page 7.16 in):
  paper/figures/fig_decomposition.pdf  ASR split into pool noise, count, bias prompts, meaning
  paper/figures/fig_forest.pdf         paired effects with 95% CIs (defences, attack components)
  paper/figures/fig_frontier.pdf       robustness-return frontier (net ASR vs clean return)
  paper/figures/fig_replay.pdf         seed-level case study (decisions, flips) under one recorded attack
  paper/tables/tab_main.tex            compact main-results table
  paper/tables/figmacros.tex           numbers quoted in the prose that only these figures compute
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import ROOT, base_parser, parse_overrides  # noqa: E402
from make_report_tables import ablation_comparisons, control_comparisons, main_comparisons  # noqa: E402

from rhal.config import load_config  # noqa: E402
from rhal.evaluation.stats import compare  # noqa: E402

# validated dataviz palette (light mode); see the dataviz reference palette
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BUY, SELL, HOLD = "#2a78d6", "#e34948", "#898781"
FLIP = "#ec835a"  # status "serious": always paired with a label
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL, PAGE = 3.5, 7.16  # IEEE column and text widths (in)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150,
})


def stats_for(root: Path) -> pd.DataFrame:
    """Every paired comparison (main, controls, ablations), recomputed from the run CSVs."""
    frames = []
    pipe = pd.read_csv(root / "pipeline" / "pipeline_runs.csv")
    frames += [compare(pipe, main_comparisons()), compare(pipe, control_comparisons())]
    for group in ("attack", "defense", "retrieval", "grounding_strength"):
        f = root / "ablation" / group / f"ablation_{group}_runs.csv"
        if f.exists():
            frames.append(compare(pd.read_csv(f), ablation_comparisons(group)))
    return pd.concat(frames, ignore_index=True).set_index("key")


def pp(st: pd.DataFrame, key: str) -> tuple[float, float, float, bool]:
    r = st.loc[key]
    return 100 * r["mean_diff"], 100 * r["ci_lo"], 100 * r["ci_hi"], bool(r["significant"])


def save(fig, name: str) -> None:
    out = ROOT / "paper" / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def fig_decomposition(summ: pd.DataFrame, st: pd.DataFrame) -> None:
    levels = [("none", "No defence"), ("grounded", "Grounding"), ("grounded_risk", "Grounding + risk")]
    parts = [("Retrieval noise (placebo)", BLUE), ("Article count", ORANGE), ("Bias prompts", AQUA), ("Misinformation content", YELLOW)]
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    for row, (d, name) in enumerate(levels):
        y = len(levels) - 1 - row
        vals = [100 * summ.loc[f"placebo__{d}", "ASR_mean"], pp(st, f"dec_count_{d}")[0], pp(st, f"dec_bias_{d}")[0], pp(st, f"dec_meaning_{d}")[0]]
        left = 0.0
        for (label, color), v in zip(parts, vals):
            w = max(v, 0.0)
            ax.barh(y, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.8, label=label if row == 0 else None)
            if w >= 2.6:
                ax.text(left + w / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if color in (BLUE, ORANGE) else INK)
            left += v  # signed: a tiny negative count component overlaps the placebo bar, so the bar ends at the ASR
        ax.text(left + 0.6, y, f"{100 * summ.loc[f'frozen__{d}', 'ASR_mean']:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(levels)), [n for _, n in reversed(levels)])
    ax.set_xlim(0, 46)
    ax.set_xlabel("Days with a flipped decision, ASR (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig_decomposition.pdf")


def _forest(ax, rows: list[tuple[str, tuple]], title: str, xlabel: str) -> None:
    ys = np.arange(len(rows))[::-1]
    for y, (label, (m, lo, hi, sig)) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=INK if sig else MUTED, linewidth=1.0, solid_capstyle="butt")
        ax.plot(m, y, marker="o", markersize=3.6, markerfacecolor=INK if sig else "white", markeredgecolor=INK if sig else MUTED, markeredgewidth=0.9)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel(xlabel)


def fig_forest(st: pd.DataFrame) -> None:
    has = lambda k: k in st.index  # noqa: E731
    dfn = [("Grounding", "grounding"), ("Microstructure proxies", "micro"), ("Risk layer alone", "risk_alone"),
           ("Risk layer on top of grounding", "risk_on_top"), ("Hold-on-conflict rule", "hold")]
    defence = []
    for name, k in dfn:
        for setting, tag in (("adaptive", "adapt."), ("frozen", "frozen")):
            key = f"abl_def_{setting}_{k}"
            if has(key):
                defence.append((f"{name} ({tag})", pp(st, key)))
    for lo, hi in (("0.3", "0.7"), ("0.7", "1.0")):
        key = f"abl_kappa_adaptive_{lo}_{hi}"
        if has(key):
            defence.append((rf"$\kappa$ {lo}$\to${hi} (adapt.)", pp(st, key)))
    comps = [("Minor edits", "wo_minor"), ("Detector filtering", "wo_filtering"), ("Bias prompts", "wo_bias"), ("Strategy feedback", "wo_feedback"), ("Style rewriting", "wo_rewrite")]
    attack_net = [(n, pp(st, f"abl_attack_{v}_net")) for n, v in comps if has(f"abl_attack_{v}_net")]
    attack_mer = [(n, pp(st, f"abl_attack_{v}_mer")) for n, v in comps if has(f"abl_attack_{v}_mer")]
    fig = plt.figure(figsize=(PAGE, 2.75))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.95, hspace=0.95)
    _forest(fig.add_subplot(gs[:, 0]), defence, "(a) Defence components", "Reduction in net ASR (pp)  →  more robust")
    _forest(fig.add_subplot(gs[0, 1]), attack_net, "(b) Attack components: decision impact", "Change in net ASR when present (pp)")
    _forest(fig.add_subplot(gs[1, 1]), attack_mer, "(c) Attack components: exposure", "Change in MER when present (pp)")
    save(fig, "fig_forest.pdf")


# label offsets (dx, dy, ha) in data units, chosen so no label crosses a marker, a whisker or an axis
LABEL_AT = {"none": (-0.4, 0.7, "right"), "risk_only": (-0.4, 0.6, "right"), "grounding_only": (0.4, 1.15, "left"),
            "grounding_micro": (-0.35, -0.95, "right"), "full_defense": (0.45, -0.55, "left")}


def fig_frontier(root: Path) -> None:
    dfs = root / "ablation" / "defense" / "ablation_defense_summary.csv"
    if not dfs.exists():
        return
    d = pd.read_csv(dfs).set_index("arm")
    pts = [("none", "No defence"), ("risk_only", "Risk only"), ("grounding_only", "Grounding (price)"),
           ("grounding_micro", "Grounding\n+ micro"), ("full_defense", "Full defence")]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for v, label in pts:
        inj, cln = d.loc[f"injected__{v}"], d.loc[f"clean__{v}"]
        x, y = 100 * inj["ASR_net_mean"], 100 * cln["CR_mean"]
        xe = [[x - 100 * inj["ASR_net_ci_lo"]], [100 * inj["ASR_net_ci_hi"] - x]]
        ye = [[y - 100 * cln["CR_ci_lo"]], [100 * cln["CR_ci_hi"] - y]]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=BLUE, markersize=3.6, elinewidth=0.8, capsize=1.6, markeredgecolor="white", markeredgewidth=0.5)
        dx, dy, ha = LABEL_AT.get(v, (0.45, 0.25, "left"))
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=6.8, ha=ha, va="center", color=INK)
    lo = min(100 * d.loc[f"injected__{v}", "ASR_net_ci_lo"] for v, _ in pts)
    ax.set_xlim(lo - 3.2, None)  # room for the left-hand labels
    ax.set_xlabel("Net ASR under attack (%)  ← more robust")
    ax.set_ylabel("Clean-market return (%)")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_frontier.pdf")


def fig_replay(cfg) -> dict:
    from export_frontend_data import replay  # same deterministic replay as the frontend

    rp = replay(cfg, cfg.seed)
    panels = [("attack", "(a) No defence"), ("grounded", "(b) Temporal grounding")]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 1.95), sharey=True)
    for ax, (arm, title) in zip(axes, panels):
        steps = rp["arms"][arm]
        x = np.arange(1, len(steps) + 1)
        px = np.array([s["px"] for s in steps])
        flips = [i for i, s in enumerate(steps) if s["a"] != s["s"]]
        for i in flips:
            ax.axvspan(x[i] - 0.45, x[i] + 0.45, color=FLIP, alpha=0.22, linewidth=0)
        ax.plot(x, px, color=INK2, linewidth=1.0, zorder=2)
        span = px.max() - px.min()
        for i, s in enumerate(steps):
            for who, off, kw in (("s", 0.17, dict(markerfacecolor="white", markeredgewidth=0.7)), ("a", 0.08, {})):
                act = s[who]
                if act == "Hold":
                    continue
                up = act == "Buy"
                ax.plot(x[i], px[i] + (1 if up else -1) * off * span, marker="^" if up else "v", markersize=3.4 if who == "a" else 3.0,
                        color=BUY if up else SELL, markeredgecolor=BUY if up else SELL, linestyle="none", zorder=3, **kw)
        ax.set_title(f"{title}: {len(flips)} of {len(steps)} decisions flipped", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel("Trading day")
        ax.set_xlim(0, len(steps) + 1)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Price")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="^", color=BUY, linestyle="none", markersize=3.6, label="agent buys"),
               Line2D([], [], marker="v", color=SELL, linestyle="none", markersize=3.6, label="agent sells"),
               Line2D([], [], marker="^", color=BUY, markerfacecolor="white", linestyle="none", markersize=3.4, label="clean twin (outline)"),
               Patch(color=FLIP, alpha=0.4, label="flip: decisions differ")]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig_replay.pdf")
    flips = {arm: sum(s["a"] != s["s"] for s in rp["arms"][arm]) for arm, _ in panels}
    return {"ReplaySeed": str(rp["seed"]), "ReplayDays": str(rp["days"]), "ReplayFlipsNone": str(flips["attack"]), "ReplayFlipsGrounded": str(flips["grounded"])}


def tex_num(x: float, plus: bool = False) -> str:
    """One decimal with a typeset minus sign (and an optional explicit plus)."""
    return ("$-$" if x < 0 else "+" if plus and x > 0 else "") + f"{abs(x):.1f}"


def tab_main(summ: pd.DataFrame, st: pd.DataFrame) -> None:
    def ci(key: str) -> str:
        m, lo, hi, _ = pp(st, key)
        return f"{tex_num(m)} [{tex_num(lo)}, {tex_num(hi)}]"
    rows = [("No defence", "none"), ("Grounding", "grounded"), ("Grounding + risk", "grounded_risk")]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Main results over \val{NseedsRun} paired seeds (\%). Net ASR $=$ ASR $-$ ASR of the paraphrase placebo, with 95\% bootstrap CIs. "
             r"\emph{Adaptive}: MisGen's feedback loop runs against the defended agent. \emph{Frozen}: the attack recorded against the undefended agent is replayed verbatim. "
             r"Clean return is the paired clean agent's cumulative return over the 60-day window; drawdown is the attacked agent's maximum drawdown.}",
             r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Defence & MER & ASR & Placebo ASR & Net ASR (adaptive) & Net ASR (frozen) & Clean return & Drawdown \\", r"\midrule"]
    for name, d in rows:
        # frozen__none replays the tape against the agent it was recorded from, so it equals the adaptive cell exactly
        frozen = ci("net_none") if d == "none" else ci(f"net_frozen_{d}")
        lines.append(f"{name} & {100 * summ.loc[f'injected__{d}', 'MER_mean']:.1f} & {100 * summ.loc[f'injected__{d}', 'ASR_mean']:.1f} & "
                     f"{100 * summ.loc[f'placebo__{d}', 'ASR_mean']:.1f} & {ci(f'net_{d}')} & {frozen} & "
                     f"{tex_num(100 * summ.loc[f'clean__{d}', 'CR_mean'], plus=True)} & {tex_num(100 * summ.loc[f'injected__{d}', 'max_drawdown_mean'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = ROOT / "paper" / "tables" / "tab_main.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    p = base_parser("Publication figures for paper/main.tex")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    if a.out:
        cfg = cfg.with_overrides({"output_dir": a.out})
    root = Path(cfg.output_dir)
    summ = pd.read_csv(root / "pipeline" / "pipeline_summary.csv").set_index("arm")
    st = stats_for(root)
    fig_decomposition(summ, st)
    fig_forest(st)
    fig_frontier(root)
    macros = fig_replay(cfg)
    tab_main(summ, st)
    macros["DecCountMaxAbs"] = f"{max(abs(pp(st, f'dec_count_{d}')[0]) for d in ('none', 'grounded', 'grounded_risk')):.1f}"
    macros["ShareNoiseOfNaive"] = f"{100 * summ.loc['placebo__none', 'ASR_mean'] / summ.loc['injected__none', 'ASR_mean']:.0f}"
    out = ROOT / "paper" / "tables" / "figmacros.tex"
    out.write_text("% AUTO-GENERATED by scripts/make_paper_figures.py -- do not edit\n"
                   + "".join(f"\\expandafter\\def\\csname {k}\\endcsname{{{v}}}\n" for k, v in macros.items()))
    print(f"wrote {out.relative_to(ROOT)}: {macros}")


if __name__ == "__main__":
    main()
