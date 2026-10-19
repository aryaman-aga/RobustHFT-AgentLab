#!/usr/bin/env python
"""Collect outputs/ tables into a single report.md and LaTeX tables for the paper.

Reads:  outputs/pipeline/pipeline_summary.csv, outputs/ablation/*/ablation_*_summary.csv,
        outputs/models/detector_meta.json
Writes: outputs/report.md, paper/tables/*.tex (incl. numbers.tex, the macros behind every number in the prose)
Paper figures are drawn by scripts/make_paper_figures.py; paper/ is self-contained, so it compiles from a
fresh clone, where outputs/ is empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from _common import ROOT, base_parser, parse_overrides

from rhal.config import load_config
from rhal.evaluation.stats import Comparison, compare
from rhal.utils.io import ensure_dir, save_csv

PCT = {
    "MER",
    "ASR",
    "ASR_net",
    "ASR_active",
    "ASR_active_net",
    "ASR_net_matched",
    "ASR_cf",
    "ASR_cf_net",
    "ASR_cf_net_matched",
    "ASR_given_exposed",
    "exposure_rate",
    "CR",
    "CR_clean_paired",
    "CR_delta_vs_clean",
    "max_drawdown",
    "hold_rate",
    "filter_recall",
    "filter_clean_drop",
    "parse_fail_rate",
    "position_gap",
}
# Generated tables shrink to the available width only when they would overflow it (needs \\usepackage{adjustbox}).
FIT_BEGIN, FIT_END = "\\begin{adjustbox}{max width=\\linewidth}", "\\end{adjustbox}"
WIDE_COLS = 6  # arm tables with this many metric columns become two-column table*


# ---------------------------------------------------------------------------
# Paired comparisons (mean over seeds of metric(A) - metric(B)). Holm is applied per family.
# Sign convention for defence/component rows: positive = the thing named second lowers the metric.
# ---------------------------------------------------------------------------
def main_comparisons() -> list[Comparison]:
    C, out = Comparison, []
    fam = "Misinformation effect beyond placebo"
    for d in ("none", "grounded", "grounded_risk"):
        out.append(C(f"net_{d}", fam, f"net ASR, adaptive, {d}", "ASR", f"injected__{d}", f"placebo__{d}"))
    for d in ("grounded", "grounded_risk"):
        out.append(C(f"net_frozen_{d}", fam, f"net ASR, frozen, {d}", "ASR", f"frozen__{d}", f"placebo__{d}"))
    for d in ("none", "grounded", "grounded_risk"):
        out.append(
            C(f"actnet_{d}", fam, f"net active ASR, adaptive, {d}", "ASR_active", f"injected__{d}", f"placebo__{d}")
        )
    for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
        fam = f"Defense gain, {setting} attack"
        for d in ("grounded", "grounded_risk"):
            out.append(C(f"gain_{setting}_{d}_asr", fam, f"ASR: none $-$ {d}", "ASR", f"{pre}__none", f"{pre}__{d}"))
            out.append(
                C(f"gain_{setting}_{d}_net", fam, f"net ASR: none $-$ {d}", "ASR_net", f"{pre}__none", f"{pre}__{d}")
            )
            out.append(
                C(
                    f"gain_{setting}_{d}_actnet",
                    fam,
                    f"net active ASR: none $-$ {d}",
                    "ASR_active_net",
                    f"{pre}__none",
                    f"{pre}__{d}",
                )
            )
        out.append(
            C(
                f"gain_{setting}_risk_net",
                fam,
                "net ASR: grounded $-$ grounded_risk",
                "ASR_net",
                f"{pre}__grounded",
                f"{pre}__grounded_risk",
            )
        )
        out.append(
            C(
                f"gain_{setting}_risk_actnet",
                fam,
                "net active ASR: grounded $-$ grounded_risk",
                "ASR_active_net",
                f"{pre}__grounded",
                f"{pre}__grounded_risk",
            )
        )
    fam = "Placebo (retrieval-noise) sensitivity"
    for d in ("grounded", "grounded_risk"):
        out.append(C(f"placebo_{d}", fam, f"placebo ASR: none $-$ {d}", "ASR", "placebo__none", f"placebo__{d}"))
    fam = "Costs"
    for d in ("grounded", "grounded_risk"):
        out.append(C(f"cost_{d}", fam, f"clean CR: none $-$ {d}", "CR", "clean__none", f"clean__{d}"))
    out.append(
        C(
            "dd_risk",
            fam,
            "attacked max DD: grounded_risk $-$ grounded",
            "max_drawdown",
            "injected__grounded_risk",
            "injected__grounded",
        )
    )
    return out


LEVELS = ("none", "grounded", "grounded_risk")


def control_comparisons() -> list[Comparison]:
    """Matched placebo decomposition and one-step counterfactual ASR (separate families, so the
    Holm-adjusted p-values of main_comparisons() do not change when these are added)."""
    C, out = Comparison, []
    fam = "ASR decomposition on the frozen tape"  # placebo -> +count matching -> +bias -> attack
    for d in LEVELS:
        out.append(
            C(
                f"dec_count_{d}",
                fam,
                f"count: matched $-$ placebo, {d}",
                "ASR",
                f"placebo_matched__{d}",
                f"placebo__{d}",
            )
        )
        out.append(
            C(
                f"dec_bias_{d}",
                fam,
                f"bias: bias-matched $-$ matched, {d}",
                "ASR",
                f"placebo_bias__{d}",
                f"placebo_matched__{d}",
            )
        )
        out.append(
            C(
                f"dec_meaning_{d}",
                fam,
                f"content: attack $-$ bias-matched, {d}",
                "ASR",
                f"frozen__{d}",
                f"placebo_bias__{d}",
            )
        )
    fam = "One-step counterfactual ASR (cf-ASR)"
    for d in LEVELS:  # positive gap = sequential ASR is inflated by the treated agent's drifted state
        out.append(
            C(
                f"cf_gap_{d}",
                fam,
                f"ASR $-$ cf-ASR, adaptive, {d}",
                "ASR",
                f"injected__{d}",
                f"injected__{d}",
                metric_b="ASR_cf",
            )
        )
    for d in LEVELS:
        out.append(C(f"cf_net_{d}", fam, f"net cf-ASR, adaptive, {d}", "ASR_cf", f"injected__{d}", f"placebo__{d}"))
    for d in LEVELS:  # the cleanest single-day meaning effect: same items, count, bias and state as the clean agent
        out.append(
            C(
                f"cf_meaning_{d}",
                fam,
                f"matched net cf-ASR, frozen, {d}",
                "ASR_cf",
                f"frozen__{d}",
                f"placebo_bias__{d}",
            )
        )
    fam = "Defense gain on matched net cf-ASR (frozen)"
    for d in ("grounded", "grounded_risk"):
        out.append(
            C(
                f"cf_gain_{d}",
                fam,
                f"matched net cf-ASR: none $-$ {d}",
                "ASR_cf_net_matched",
                "frozen__none",
                f"frozen__{d}",
            )
        )
    return out


# Replications of the main experiment with another reader/agent or on real prices. Each run lives in
# its own directory under the output root and gets its own families, so nothing above moves.
REPLICATIONS = [  # (key/macro prefix, directory under the output root, label)
    ("llmr", "llm/reader_qwen7b", "LLM reader"),
    ("llmd", "llm/decider_qwen7b", "LLM decider"),
    ("real", "real_btc", "BTC-USD prices"),
]


def replication_comparisons(prefix: str, label: str) -> list[Comparison]:
    """The headline comparisons of the main experiment, re-keyed ``<prefix>_<key>`` in new families.
    Comparisons whose arms a run does not have (e.g. matched placebos in the decider run) are skipped."""
    C, out = Comparison, []
    fam = f"{label}: misinformation effect and defense gain"
    for d in LEVELS:
        out.append(C(f"{prefix}_net_{d}", fam, f"net ASR, adaptive, {d}", "ASR", f"injected__{d}", f"placebo__{d}"))
    for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
        for d in ("grounded", "grounded_risk"):
            out.append(
                C(
                    f"{prefix}_gain_{setting}_{d}_net",
                    fam,
                    f"net ASR ({setting}): none $-$ {d}",
                    "ASR_net",
                    f"{pre}__none",
                    f"{pre}__{d}",
                )
            )
        out.append(
            C(
                f"{prefix}_gain_{setting}_risk_net",
                fam,
                f"net ASR ({setting}): grounded $-$ grounded_risk",
                "ASR_net",
                f"{pre}__grounded",
                f"{pre}__grounded_risk",
            )
        )
    out.append(
        C(
            f"{prefix}_placebo_grounded",
            fam,
            "placebo ASR: none $-$ grounded",
            "ASR",
            "placebo__none",
            "placebo__grounded",
        )
    )
    out.append(C(f"{prefix}_cost_grounded", fam, "clean CR: none $-$ grounded", "CR", "clean__none", "clean__grounded"))
    fam = f"{label}: ASR decomposition on the frozen tape"
    for d in ("none", "grounded"):
        out.append(
            C(
                f"{prefix}_dec_bias_{d}",
                fam,
                f"bias: bias-matched $-$ matched, {d}",
                "ASR",
                f"placebo_bias__{d}",
                f"placebo_matched__{d}",
            )
        )
        out.append(
            C(
                f"{prefix}_dec_meaning_{d}",
                fam,
                f"content: attack $-$ bias-matched, {d}",
                "ASR",
                f"frozen__{d}",
                f"placebo_bias__{d}",
            )
        )
    return out


IMPACT_GAMMAS = ("0.0", "1.0", "2.0")  # must match scripts/run_price_impact.py


def impact_comparisons() -> list[Comparison]:
    """Price-impact stress test (frozen attack): grounding's gain and the content effect per gamma,
    and the paired drop in grounding's gain relative to gamma = 0 (new families)."""
    C, out = Comparison, []
    fam = "Price impact: defense gain and content effect (frozen attack)"
    for g in IMPACT_GAMMAS:
        for d in ("grounded", "grounded_risk"):
            out.append(
                C(
                    f"imp_gain_{d}_{g}",
                    fam,
                    f"net ASR: none $-$ {d}, $\\gamma={g}$",
                    "ASR_net",
                    f"frozen__none@g{g}",
                    f"frozen__{d}@g{g}",
                )
            )
        for d in ("none", "grounded"):
            out.append(
                C(
                    f"imp_meaning_{d}_{g}",
                    fam,
                    f"content: attack $-$ bias-matched, {d}, $\\gamma={g}$",
                    "ASR",
                    f"frozen__{d}@g{g}",
                    f"placebo_bias__{d}@g{g}",
                )
            )
    fam = "Price impact: change in grounding's gain"
    for g in IMPACT_GAMMAS[1:]:
        out.append(
            C(f"imp_gain_drop_{g}", fam, f"gain: $\\gamma=0$ $-$ $\\gamma={g}$", "gain_net", "gain@g0.0", f"gain@g{g}")
        )
    return out


def decomposition_tables(full: pd.DataFrame, st: pd.DataFrame) -> tuple[str, str]:
    """Markdown + LaTeX: mean ASR of each control per defence level, and the meaning share with its CI."""
    mean = full.set_index("arm")["ASR_mean"]
    by_key = st.set_index("key") if not st.empty else pd.DataFrame()
    md = [
        "| defence | placebo | count-matched | + bias | frozen attack | meaning (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{ASR (\\%) of the controls built from the frozen attack tape, mean over seeds. Each column adds one ingredient of the attack: "
        "paraphrases of every item (classic placebo), paraphrases of only the attacked items, plus the attack's bias contexts, and finally the "
        "misinformation itself. Meaning $=$ frozen $-$ (count-matched $+$ bias), with its 95\\% bootstrap CI.}",
        "\\label{tab:decomposition}",
        "\\small",
        FIT_BEGIN,
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Defense & Placebo & Matched & $+$Bias & Attack & Meaning \\\\",
        "\\midrule",
    ]
    for d in LEVELS:
        arms = [f"placebo__{d}", f"placebo_matched__{d}", f"placebo_bias__{d}", f"frozen__{d}"]
        if not all(a in mean.index for a in arms) or f"dec_meaning_{d}" not in by_key.index:
            continue
        v = [100 * mean[a] for a in arms]
        r = by_key.loc[f"dec_meaning_{d}"]
        ci = f"{100 * r['mean_diff']:+.1f} [{100 * r['ci_lo']:+.1f}, {100 * r['ci_hi']:+.1f}]"
        md.append(f"| {d} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {ci} |")
        tex.append(d.replace("_", "\\_") + " & " + " & ".join(f"{x:.1f}" for x in v) + f" & {ci} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", FIT_END, "\\end{table}"]
    return "\n".join(md), "\n".join(tex)


def ablation_comparisons(group: str) -> list[Comparison]:
    C, out = Comparison, []
    if group == "attack":  # positive = removing the component weakens the attack
        for v in ("wo_bias", "wo_feedback", "wo_filtering", "wo_minor", "wo_rewrite"):
            out.append(
                C(
                    f"abl_attack_{v}_net",
                    "Attack components (net ASR)",
                    f"net ASR: full $-$ {v}",
                    "ASR_net",
                    "injected__full",
                    f"injected__{v}",
                )
            )
            out.append(
                C(
                    f"abl_attack_{v}_mer",
                    "Attack components (MER)",
                    f"MER: full $-$ {v}",
                    "MER",
                    "injected__full",
                    f"injected__{v}",
                )
            )
    elif group == "defense":  # positive = the second variant has lower net ASR
        pairs = [
            ("micro", "grounding_only", "grounding_micro"),
            ("grounding", "none", "grounding_only"),
            ("risk_alone", "none", "risk_only"),
            ("risk_on_top", "grounding_micro", "full_defense"),
            ("hold", "wo_hold_on_conflict", "full_defense"),
        ]
        for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
            for key, a, b in pairs:
                out.append(
                    C(
                        f"abl_def_{setting}_{key}",
                        f"Defense components, {setting} attack (net ASR)",
                        f"{a} $-$ {b}",
                        "ASR_net",
                        f"{pre}__{a}",
                        f"{pre}__{b}",
                    )
                )
    elif group == "grounding_strength":  # positive = larger kappa lowers net ASR / costs clean return
        for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
            for lo, hi in (("0.3", "0.7"), ("0.7", "1.0")):
                out.append(
                    C(
                        f"abl_kappa_{setting}_{lo}_{hi}",
                        f"Grounding strength, {setting} attack (net ASR)",
                        f"$\\kappa$ {lo} $-$ {hi}",
                        "ASR_net",
                        f"{pre}__kappa_{lo}",
                        f"{pre}__kappa_{hi}",
                    )
                )
        for lo, hi in (("0.3", "0.7"), ("0.7", "1.0")):
            out.append(
                C(
                    f"abl_kappa_cost_{lo}_{hi}",
                    "Grounding strength (clean CR)",
                    f"$\\kappa$ {lo} $-$ {hi}",
                    "CR",
                    f"clean__kappa_{lo}",
                    f"clean__kappa_{hi}",
                )
            )
    elif group == "baselines":  # positive = the second-named variant has the lower value
        gains = [
            ("grounding", "none", "grounding"),
            ("text_free", "none", "text_free"),
            ("detect_10", "none", "detect_10"),
            ("detect_25", "none", "detect_25"),
            ("median", "none", "median"),
            ("trimmed", "none", "trimmed"),
            ("detect_10_vs_grounding", "detect_10", "grounding"),
            ("detect_25_vs_grounding", "detect_25", "grounding"),
            ("median_vs_grounding", "median", "grounding"),
            ("trimmed_vs_grounding", "trimmed", "grounding"),
        ]
        for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
            for key, a, b in gains:
                out.append(
                    C(
                        f"base_{setting}_{key}",
                        f"Baseline defenses, {setting} attack (net ASR)",
                        f"{a} $-$ {b}",
                        "ASR_net",
                        f"{pre}__{a}",
                        f"{pre}__{b}",
                    )
                )
            out.append(
                C(
                    f"base_{setting}_detect_nofilter",
                    f"Detection vs an unfiltered attacker, {setting} (net ASR)",
                    "none_nofilter $-$ detect_25_nofilter",
                    "ASR_net",
                    f"{pre}__none_nofilter",
                    f"{pre}__detect_25_nofilter",
                )
            )
        for v in ("grounding", "text_free", "detect_10", "detect_25", "median", "trimmed"):
            out.append(
                C(f"base_cost_{v}", "Baseline defenses (clean CR)", f"none $-$ {v}", "CR", "clean__none", f"clean__{v}")
            )
    elif group == "aware_attack":  # a - b; for "x_aligned - x" a positive value means the aware attacker is stronger
        pairs = [
            ("gain_std", "none", "grounding"),
            ("gain_aligned", "none_aligned", "grounding_aligned"),
            ("aligned_vs_std_none", "none_aligned", "none"),
            ("aligned_vs_std_grounding", "grounding_aligned", "grounding"),
            ("aligned_vs_std_full", "full_aligned", "full"),
            ("conflict_vs_std_full", "full_conflict", "full"),
        ]
        for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
            for key, a, b in pairs:
                out.append(
                    C(
                        f"aware_{setting}_{key}",
                        f"Defense-aware attacker, {setting} (net ASR)",
                        f"{a} $-$ {b}",
                        "ASR_net",
                        f"{pre}__{a}",
                        f"{pre}__{b}",
                    )
                )
    elif group == "conflict_handling":  # a - b: positive = the second-named rule leaves the lower value
        for setting, pre in (("adaptive", "injected"), ("frozen", "frozen")):
            for atk in ("", "_conflict"):
                tag = "conflict-seeking attacker" if atk else "standard attacker"
                for key, a, b in (("shrink", f"hold{atk}", f"shrink{atk}"), ("nohold", f"hold{atk}", f"nohold{atk}")):
                    for metric, mname in (("ASR_net", "net ASR"), ("position_gap", "position gap")):
                        out.append(
                            C(
                                f"conf_{setting}{atk}_{key}_{metric.lower()}",
                                f"Conflict rule, {setting}, {tag} ({mname})",
                                f"{a} $-$ {b}",
                                metric,
                                f"{pre}__{a}",
                                f"{pre}__{b}",
                            )
                        )
            # how much stronger the conflict-seeking attacker is under each rule (positive = stronger)
            for rule in ("hold", "shrink", "nohold"):
                out.append(
                    C(
                        f"conf_{setting}_attacker_{rule}",
                        f"Conflict-seeking vs standard attacker, {setting} (net ASR)",
                        f"{rule}_conflict $-$ {rule}",
                        "ASR_net",
                        f"{pre}__{rule}_conflict",
                        f"{pre}__{rule}",
                    )
                )
    elif group == "retrieval":  # is net ASR an artefact of top-k? (expect no difference)
        for k in ("k2", "k8"):
            out.append(
                C(
                    f"abl_ret_{k}_net",
                    "Retrieval breadth (net ASR)",
                    f"net ASR: {k} $-$ k4",
                    "ASR_net",
                    f"injected__{k}",
                    "injected__k4",
                )
            )
            out.append(
                C(
                    f"abl_ret_{k}_placebo",
                    "Retrieval breadth (placebo ASR)",
                    f"placebo ASR: {k} $-$ k4",
                    "ASR",
                    f"placebo__{k}",
                    "placebo__k4",
                )
            )
    return out


def _fmt_p(p: float) -> str:
    return "--" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def stats_markdown(st: pd.DataFrame) -> str:
    out = ["| family | comparison | Δ (pp) | 95% CI | p (Holm) | n |", "|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        sig = " *" if r["significant"] else ""
        out.append(
            f"| {r['family']} | {r['label'].replace('$-$', '−').replace(chr(92) + 'kappa', 'κ').replace('$', '')} | {100 * r['mean_diff']:+.1f} | [{100 * r['ci_lo']:+.1f}, {100 * r['ci_hi']:+.1f}] | {_fmt_p(r['p_holm'])}{sig} | {r['n']} |"
        )
    return "\n".join(out)


# arm identifiers in comparison labels -> words, for the paper's appendix tables (order matters)
PRETTY = [
    (r"\bplacebo_matched\b", "count-matched placebo"),
    (r"\bplacebo_bias\b", "bias-matched placebo"),
    (r"\bgrounded_risk\b", "grounding + risk"),
    (r"\bgrounding_micro\b", "grounding + micro"),
    (r"\bgrounding_only\b", "price-only grounding"),
    (r"\bfull_defense\b", "full defense"),
    (r"\brisk_only\b", "risk only"),
    (r"\bwo_hold_on_conflict\b", "w/o hold-on-conflict"),
    (r"\bwo_minor\b", "w/o minor edits"),
    (r"\bwo_filtering\b", "w/o filtering"),
    (r"\bwo_bias\b", "w/o bias prompts"),
    (r"\bwo_feedback\b", "w/o feedback"),
    (r"\bwo_rewrite\b", "w/o rewriting"),
    (r"\bfull\b(?! defense)", "full attack"),
    (r"\bnone\b", "no defense"),
    (r"\bgrounded\b", "grounding"),
    (r"\bfrozen \$-\$", "frozen attack $-$"),
    (r"\bk([248])\b", r"$k{=}\1$"),
    (r"\$\\kappa\$ ([0-9.]+) \$-\$ ([0-9.]+)", r"$\\kappa{=}\1$ $-$ $\\kappa{=}\2$"),
]


# operator names (misgen.operators.MINOR_TYPES) -> the names used in the paper's text
EDIT_TYPE_NAMES = {"flipping": "direction flip", "sentiment": "sentiment shift", "other": "vague attribution"}


def pretty_label(label: str) -> str:
    for pat, rep in PRETTY:
        label = re.sub(pat, rep, label)
    return label


def _tex_signed(x: float) -> str:
    """Signed one-decimal number with a typeset minus sign (a hyphen is too short in a table)."""
    return f"{x:+.1f}".replace("-", "$-$")


def stats_latex(st: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        FIT_BEGIN,
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Comparison & $\\Delta$ (pp) & 95\\% CI & $p_{\\mathrm{Holm}}$ \\\\",
    ]
    for fam, block in st.groupby("family", sort=False):
        lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\textit{{{fam}}}}} \\\\"]
        for _, r in block.iterrows():
            p = _fmt_p(r["p_holm"]).replace("<", "$<$")
            p = f"\\textbf{{{p}}}" if r["significant"] else p
            label = pretty_label(r["label"]).replace(
                "_", "\\_"
            )  # (no backslashes inside f-string braces before Python 3.12)
            lines.append(
                f"{label} & {_tex_signed(100 * r['mean_diff'])} & [{_tex_signed(100 * r['ci_lo'])}, {_tex_signed(100 * r['ci_hi'])}] & {p} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", FIT_END, "\\end{table}"]
    return "\n".join(lines)


def latex_table(agg: pd.DataFrame, cols: list[str], caption: str, label: str, arm_map=None) -> str:
    cols = [c for c in cols if f"{c}_mean" in agg.columns]
    head = " & ".join(["Arm"] + [c.replace("_", "\\_") for c in cols]) + " \\\\"
    env = "table*" if len(cols) >= WIDE_COLS else "table"  # wide tables span both IEEE columns
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        FIT_BEGIN,
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        head,
        "\\midrule",
    ]
    for _, r in agg.iterrows():
        cells = []
        for c in cols:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            if pd.isna(m):
                cells.append("--")
            elif c in PCT:
                cells.append(f"{100 * m:.1f}$\\pm${100 * s:.1f}")
            else:
                cells.append(f"{m:.2f}$\\pm${s:.2f}")
        name = (arm_map(r["arm"]) if arm_map else r["arm"]).replace("_", "\\_")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", FIT_END, f"\\end{{{env}}}"]
    return "\n".join(lines)


def md_table(agg: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if f"{c}_mean" in agg.columns]
    out = ["| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for _, r in agg.iterrows():
        cells = []
        for c in cols:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            cells.append(
                "--" if pd.isna(m) else (f"{100 * m:.1f} ± {100 * s:.1f}" if c in PCT else f"{m:.2f} ± {s:.2f}")
            )
        out.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


_DIGITS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    ".": "p",
}


def _macro_name(*parts: str) -> str:
    out = ""
    for part in parts:
        for tok in part.replace("-", "_").split("_"):
            tok = "".join(_DIGITS.get(ch, ch) for ch in tok)
            out += tok[:1].upper() + tok[1:]
    return out


def emit_numbers(agg_by_prefix: dict[str, pd.DataFrame], extra: dict[str, str], path: Path) -> None:
    r"""Write LaTeX \csname macros so prose numbers in main.tex always match outputs."""
    lines = ["% AUTO-GENERATED by scripts/make_report_tables.py -- do not edit"]

    def put(name: str, val) -> None:
        lines.append(f"\\expandafter\\def\\csname {name}\\endcsname{{{val}}}")

    for prefix, agg in agg_by_prefix.items():
        for _, r in agg.iterrows():
            arm = r["arm"]
            for metric in [
                "MER",
                "MER_lift",
                "ASR",
                "ASR_net",
                "ASR_active",
                "ASR_active_net",
                "ASR_net_matched",
                "ASR_cf",
                "ASR_cf_net",
                "ASR_cf_net_matched",
                "CR",
                "CR_clean_paired",
                "max_drawdown",
                "signal_deviation",
                "hold_rate",
            ]:
                if f"{metric}_mean" not in agg.columns or pd.isna(r[f"{metric}_mean"]):
                    continue
                m, sd = r[f"{metric}_mean"], r[f"{metric}_std"]
                fmt = (f"{100 * m:.1f}", f"{100 * sd:.1f}") if metric in PCT else (f"{m:.2f}", f"{sd:.2f}")
                put(_macro_name(prefix, arm, metric), fmt[0])
                put(_macro_name(prefix, arm, metric, "sd"), fmt[1])
    for k, v in extra.items():
        put(k, v)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = base_parser("Build report tables")
    a = p.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    root = Path(a.out or cfg.output_dir)
    tex_dir = ensure_dir(ROOT / "paper" / "tables")
    aggs: dict[str, pd.DataFrame] = {}
    extra: dict[str, str] = {
        "Nseeds": str(cfg.experiment.n_seeds),
        "NevalDays": str(cfg.data.eval_days),
        "NewsPerDay": str(cfg.data.news_per_day),
        "NDays": str(cfg.data.n_days),
        "Kappa": f"{cfg.grounding.strength:.1f}",
        "TopK": str(cfg.retrieval.top_k),
        "FeeBps": f"{cfg.backtest.fee_bps:.0f}",
        "SlipBps": f"{cfg.backtest.slippage_bps:.0f}",
    }
    md = [
        "# RobustHFT-AgentLab -- experiment report",
        "",
        "Arm tables: mean ± std over seeds; percentages where applicable. Paired comparisons: mean per-seed difference (percentage points), "
        "95% percentile-bootstrap CI over seeds, two-sided sign-flip permutation p-value with Holm correction within each family (* = p_Holm < 0.05). "
        "Offline synthetic simulation; not investment advice.",
        "",
    ]

    def add_stats(st: pd.DataFrame) -> None:
        """Expose every comparison to the paper as \\val{Cmp<Key>}, ...Lo, ...Hi, ...P (Holm)."""
        for _, r in st.iterrows():
            extra[_macro_name("Cmp", r["key"])] = f"{100 * r['mean_diff']:.1f}"
            extra[_macro_name("Cmp", r["key"], "lo")] = f"{100 * r['ci_lo']:.1f}"
            extra[_macro_name("Cmp", r["key"], "hi")] = f"{100 * r['ci_hi']:.1f}"
            extra[_macro_name("Cmp", r["key"], "p")] = _fmt_p(r["p_holm"]).replace("<", "$<$")
            # the same comparison read the other way round (B - A), so prose can say "adds X pp" without a minus sign
            extra[_macro_name("Cmp", r["key"], "neg")] = f"{-100 * r['mean_diff']:.1f}"
            extra[_macro_name("Cmp", r["key"], "neg", "lo")] = f"{-100 * r['ci_hi']:.1f}"
            extra[_macro_name("Cmp", r["key"], "neg", "hi")] = f"{-100 * r['ci_lo']:.1f}"

    pipe = root / "pipeline" / "pipeline_summary.csv"
    if pipe.exists():
        full = pd.read_csv(pipe)
        order = [
            "clean__none",
            "placebo__none",
            "injected__none",
            "clean__grounded",
            "placebo__grounded",
            "injected__grounded",
            "frozen__grounded",
            "clean__grounded_risk",
            "placebo__grounded_risk",
            "injected__grounded_risk",
            "frozen__grounded_risk",
        ]
        agg = full.set_index("arm").reindex([o for o in order if o in set(full["arm"])]).reset_index()
        aggs["M"] = full  # macros for every arm (incl. matched placebos); the main table shows the ordered subset
        extra["NseedsRun"] = str(int(agg["n_seeds"].iloc[0]))
        cols_attack = [
            "MER",
            "MER_lift",
            "ASR",
            "ASR_net",
            "ASR_net_matched",
            "ASR_cf",
            "ASR_cf_net",
            "ASR_active",
            "ASR_active_net",
            "signal_deviation",
            "CR",
            "CR_clean_paired",
            "max_drawdown",
            "hold_rate",
        ]
        md += [
            "## Main results (clean / placebo / injected × defence level)",
            "",
            "`injected__*` face an adaptive attacker; `frozen__*` replay the attack recorded against the undefended agent of the same seed "
            "(`frozen__none` ≡ `injected__none`, so it is omitted).",
            "",
            md_table(agg, cols_attack),
            "",
        ]
        (tex_dir / "main_results.tex").write_text(
            latex_table(
                agg,
                ["MER", "MER_lift", "ASR", "ASR_net", "ASR_active_net", "CR", "max_drawdown"],
                "Main results on synthetic data (mean$\\pm$std over seeds; \\%). Net ASR $=$ ASR $-$ ASR$_{\\text{placebo}}$. "
                "\\texttt{frozen} rows replay the attack recorded against the undefended agent, so every defence faces identical documents.",
                "tab:main",
            )
        )
        runs_csv = root / "pipeline" / "pipeline_runs.csv"
        if runs_csv.exists():
            st = compare(pd.read_csv(runs_csv), main_comparisons())
            if not st.empty:
                save_csv(st, root / "pipeline" / "pipeline_stats.csv")
                add_stats(st)
                md += ["## Paired comparisons: main experiment", "", stats_markdown(st), ""]
                (tex_dir / "stats_main.tex").write_text(
                    stats_latex(
                        st,
                        "Paired comparisons over seeds (main experiment): mean difference in percentage points, 95\\% bootstrap CI, "
                        "sign-flip permutation $p$ with Holm correction per family (bold: $p_{\\mathrm{Holm}}<0.05$).",
                        "tab:stats_main",
                    )
                )
            st_c = compare(pd.read_csv(runs_csv), control_comparisons())
            if not st_c.empty:
                save_csv(st_c, root / "pipeline" / "pipeline_stats_controls.csv")
                add_stats(st_c)
                dec_md, dec_tex = decomposition_tables(full, st_c)
                meaning = (
                    st_c.set_index("key").loc["dec_meaning_none", "mean_diff"]
                    if "dec_meaning_none" in set(st_c["key"])
                    else None
                )
                if meaning:
                    naive = float(full.set_index("arm").loc["injected__none", "ASR_mean"])
                    extra["RatioNaiveToMeaning"] = f"{naive / meaning:.1f}"  # naive ASR / content effect
                    extra["ShareMeaningOfNaive"] = f"{100 * meaning / naive:.0f}"  # content effect as % of naive ASR
                md += [
                    "## Matched controls: ASR decomposition on the frozen tape",
                    "",
                    "Each column adds one ingredient of the attack to the treated pool (same seed, same tape). "
                    "Meaning = frozen attack − (count-matched placebo + bias).",
                    "",
                    dec_md,
                    "",
                    "## Paired comparisons: matched controls and one-step counterfactual ASR",
                    "",
                    stats_markdown(st_c),
                    "",
                ]
                (tex_dir / "asr_decomposition.tex").write_text(dec_tex)
                (tex_dir / "stats_controls.tex").write_text(
                    stats_latex(
                        st_c,
                        "Matched placebo decomposition and one-step counterfactual ASR (cf-ASR: the treated agent's text and bias "
                        "applied to the clean agent's state each day); conventions as in Table~\\ref{tab:stats_main}.",
                        "tab:stats_controls",
                    )
                )
        # defence robustness gain table
        inj = agg[agg["arm"].str.startswith("injected__")].copy()
        inj["defence"] = inj["arm"].str.replace("injected__", "")
        base_asr = (
            float(inj.loc[inj["defence"] == "none", "ASR_mean"].iloc[0])
            if (inj["defence"] == "none").any()
            else float("nan")
        )
        base_net = (
            float(inj.loc[inj["defence"] == "none", "ASR_net_mean"].iloc[0])
            if (inj["defence"] == "none").any()
            else float("nan")
        )
        inj["robustness_gain_ASR"] = base_asr - inj["ASR_mean"]
        inj["robustness_gain_net"] = base_net - inj["ASR_net_mean"]
        for _, r in inj.iterrows():
            extra[_macro_name("Gain", r["defence"], "ASR")] = f"{100 * r['robustness_gain_ASR']:.1f}"
            extra[_macro_name("Gain", r["defence"], "ASRnet")] = f"{100 * r['robustness_gain_net']:.1f}"
        md += [
            "## Robustness gain of defences (Δ ASR vs undefended, percentage points)",
            "",
            "| defence | ASR | net ASR | active net ASR | gain (ASR) | gain (net) | clean CR | attacked DD |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in inj.iterrows():
            act = r.get("ASR_active_net_mean", float("nan"))
            md.append(
                f"| {r['defence']} | {100 * r['ASR_mean']:.1f} | {100 * r['ASR_net_mean']:.1f} | {'--' if pd.isna(act) else f'{100 * act:.1f}'} | {100 * r['robustness_gain_ASR']:+.1f} | {100 * r['robustness_gain_net']:+.1f} | {100 * r['CR_clean_paired_mean']:+.1f} | {100 * r['max_drawdown_mean']:+.1f} |"
            )
        md.append("")

    abl_stats: list[pd.DataFrame] = []
    ext_stats: list[pd.DataFrame] = []  # baselines + defense-aware attacker: their own appendix table
    for group, cols in [
        ("attack", ["MER", "MER_lift", "ASR", "ASR_net", "CR"]),
        ("defense", ["ASR", "ASR_net", "ASR_active_net", "CR", "CR_clean_paired", "max_drawdown", "hold_rate"]),
        ("retrieval", ["MER", "ASR", "ASR_net", "CR"]),
        ("grounding_strength", ["ASR", "ASR_net", "CR", "CR_clean_paired"]),
        ("baselines", ["MER", "ASR", "ASR_net", "CR", "CR_clean_paired", "filter_recall", "filter_clean_drop"]),
        ("aware_attack", ["MER", "ASR", "ASR_net", "CR_clean_paired"]),
        ("conflict_handling", ["ASR", "ASR_net", "position_gap", "CR_clean_paired", "hold_rate"]),
    ]:
        f = root / "ablation" / group / f"ablation_{group}_summary.csv"
        if f.exists():
            agg = pd.read_csv(f)
            aggs[_macro_name("Abl", group)] = agg
            md += [f"## Ablation: {group}", "", md_table(agg, cols), ""]
            (tex_dir / f"ablation_{group}.tex").write_text(
                latex_table(
                    agg, cols, f"Ablation ({group.replace('_', ' ')}), mean$\\pm$std over seeds.", f"tab:abl_{group}"
                )
            )
            runs_csv = root / "ablation" / group / f"ablation_{group}_runs.csv"
            if runs_csv.exists():
                st = compare(pd.read_csv(runs_csv), ablation_comparisons(group))
                if not st.empty:
                    save_csv(st, root / "ablation" / group / f"ablation_{group}_stats.csv")
                    add_stats(st)
                    (ext_stats if group in ("baselines", "aware_attack", "conflict_handling") else abl_stats).append(st)
                    md += [f"### Paired comparisons: {group}", "", stats_markdown(st), ""]
    if ext_stats:
        (tex_dir / "stats_extensions.tex").write_text(
            stats_latex(
                pd.concat(ext_stats, ignore_index=True),
                "Paired comparisons over seeds for the baseline defenses and the defense-aware attacker; conventions as in Table~\\ref{tab:stats_main}. "
                "In \\emph{x\\_aligned $-$ x} rows a positive $\\Delta$ means the aware attacker is stronger.",
                "tab:stats_extensions",
            )
        )
    if abl_stats:
        (tex_dir / "stats_ablation.tex").write_text(
            stats_latex(
                pd.concat(abl_stats, ignore_index=True),
                "Paired comparisons over seeds (ablations); conventions as in Table~\\ref{tab:stats_main}. "
                "Positive $\\Delta$: the second-named variant has the lower value.",
                "tab:stats_ablation",
            )
        )

    meta = root / "models" / "detector_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        extra.update(
            {
                "DetAcc": f"{m['train_acc']:.2f}",
                "DetThr": f"{m['threshold']:.2f}",
                "DetFPR": f"{100 * m['holdout_clean_fpr_at_threshold']:.0f}",
                "DetNtrain": str(m["n_clean_train"] + m["n_perturbed"]),
            }
        )
        if "per_type_detection_rate" in m:
            for k, v in m["per_type_detection_rate"].items():
                if isinstance(v, dict):
                    extra[_macro_name("Det", k, "minor")] = f"{100 * v['minor']:.0f}"
                    extra[_macro_name("Det", k, "news")] = f"{100 * v['news_style']:.0f}"
                    extra[_macro_name("Det", k, "acad")] = f"{100 * v['academic_style']:.0f}"
        md += [
            "## Detectability proxy",
            "",
            f"train acc {m['train_acc']:.3f}; threshold {m['threshold']:.3f} at target FPR {m['target_fpr']}; held-out clean FPR {m['holdout_clean_fpr_at_threshold']:.3f}",
            "",
        ]
        if "per_type_detection_rate" in m:
            md += ["| type | minor | after news rewrite | after academic rewrite |", "|---|---|---|---|"]
            for k, v in m["per_type_detection_rate"].items():
                if isinstance(v, dict):
                    md.append(f"| {k} | {v['minor']:.2f} | {v['news_style']:.2f} | {v['academic_style']:.2f} |")
            md.append("")
            rows = [(k, v) for k, v in m["per_type_detection_rate"].items() if isinstance(v, dict)]
            tex = [
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{Detection rate of the proxy detector by edit type, for the minor edit alone and after each style rewrite (threshold set at {100 * m['target_fpr']:.0f}\\% false positives on clean text).}}",
                "\\label{tab:detector}",
                "\\footnotesize",
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Edit type & Minor edit & + news rewrite & + academic rewrite \\\\",
                "\\midrule",
            ]
            tex += [
                f"{EDIT_TYPE_NAMES.get(k, k.replace('_', ' '))} & {v['minor']:.2f} & {v['news_style']:.2f} & {v['academic_style']:.2f} \\\\"
                for k, v in rows
            ]
            tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
            (tex_dir / "detector.tex").write_text("\n".join(tex))

    imp_csv = root / "price_impact" / "price_impact_runs.csv"
    imp_meta = root / "price_impact" / "impact_meta.json"
    if imp_meta.exists():
        im = json.loads(imp_meta.read_text())
        extra["ImpShiftMedian"] = f"{im['abs_shift_median']:.2f}"
        extra["ImpShiftPNinety"] = f"{im['abs_shift_p90']:.2f}"
    if imp_csv.exists():
        st = compare(pd.read_csv(imp_csv), impact_comparisons())
        if not st.empty:
            save_csv(st, root / "price_impact" / "price_impact_stats.csv")
            add_stats(st)
            md += ["## Price-impact stress test (frozen attack)", "", stats_markdown(st), ""]
            (tex_dir / "stats_impact.tex").write_text(
                stats_latex(
                    st,
                    "Price-impact stress test: the frozen attack also moves the shared price path by "
                    "$\\gamma$ daily standard deviations per unit of sentiment shift; conventions as in Table~\\ref{tab:stats_main}.",
                    "tab:stats_impact",
                )
            )
    rep_stats: list[pd.DataFrame] = []
    for prefix, rel, label in REPLICATIONS:
        rdir = root / rel / "pipeline"
        if not (rdir / "pipeline_runs.csv").exists():
            continue
        rsum = pd.read_csv(rdir / "pipeline_summary.csv")
        aggs[_macro_name(prefix)] = rsum  # e.g. \val{LlmrInjectedNoneASR}
        extra[_macro_name("Nseeds", prefix)] = str(int(rsum["n_seeds"].max()))
        llm = rdir / "llm_stats.json"
        if llm.exists():
            ls = json.loads(llm.read_text())
            extra[_macro_name(prefix, "model")] = str(ls.get("model", "")).replace("_", "\\_")
            extra[_macro_name(prefix, "parse", "failures")] = str(ls.get("parse_failures", 0))
            if "texts_scored" in ls:
                extra[_macro_name(prefix, "texts", "scored")] = str(ls["texts_scored"])
            if "cache_misses" in ls:
                extra[_macro_name(prefix, "llm", "calls")] = str(ls["cache_misses"])
        st = compare(pd.read_csv(rdir / "pipeline_runs.csv"), replication_comparisons(prefix, label))
        if not st.empty:
            save_csv(st, rdir / "pipeline_stats_replication.csv")
            add_stats(st)
            rep_stats.append(st)
            md += [f"## Replication: {label} ({rel})", "", stats_markdown(st), ""]
    if rep_stats:
        (tex_dir / "stats_replications.tex").write_text(
            stats_latex(
                pd.concat(rep_stats, ignore_index=True),
                "Paired comparisons over seeds for the replications with a local LLM and on real BTC-USD prices; "
                "conventions as in Table~\\ref{tab:stats_main}.",
                "tab:stats_replications",
            )
        )
    emit_numbers(aggs, extra, tex_dir / "numbers.tex")
    (root / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {root / 'report.md'} and LaTeX tables to {tex_dir}/")


if __name__ == "__main__":
    main()
