"""Paired statistics over seeds.

Why: with 5 seeds and mean +/- std, effects such as "grounding lowers net ASR from
13.7 to 6.0" (std ~7) cannot be told apart from noise. Every arm of a seed shares the
market path, the clean news and -- with common-random-number seeding -- the
attacker's random draws, so comparisons are *paired by seed*. For "A vs B on metric X"
we analyse the per-seed differences d_s = X(A, s) - X(B, s) and report:

  * the mean difference with a 95% percentile-bootstrap CI (resampling seeds);
  * a two-sided sign-flip permutation p-value. Under H0 "A and B are exchangeable"
    each d_s is symmetric around 0, so randomly flipping signs generates the null
    distribution of the mean. Exact enumeration for n <= 16 seeds, Monte Carlo above;
  * Holm-Bonferroni adjusted p-values within a family of related comparisons, so
    testing many ablation rows does not manufacture significance.

All resampling goes through rhal.seeding.rng, so the statistics are deterministic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..seeding import rng

EXACT_MAX_N = 16  # 2^16 sign patterns is still cheap to enumerate


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def bootstrap_ci(x, level: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap CI for the mean of ``x`` (NaNs dropped)."""
    a = _clean(x)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), float(a[0])
    idx = rng(seed, "bootstrap", len(a)).integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def sign_flip_pvalue(d, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired permutation test of H0: mean(d) = 0 (NaNs dropped)."""
    a = _clean(d)
    n = len(a)
    if n == 0:
        return float("nan")
    obs = abs(a.mean())
    tol = 1e-12 * max(1.0, obs)  # count ties as extreme despite float rounding
    if n <= EXACT_MAX_N:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        null = np.abs((signs * a).mean(axis=1))
        return float(np.mean(null >= obs - tol))
    signs = rng(seed, "signflip", n).choice((-1.0, 1.0), size=(n_perm, n))
    null = np.abs((signs * a).mean(axis=1))
    return float((1 + np.sum(null >= obs - tol)) / (1 + n_perm))  # +1: the observed labelling counts


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment; NaNs are passed through and not counted."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, (m - rank) * p[i])  # enforce monotone adjusted p-values
        out[i] = min(1.0, running)
    return out


def paired_diffs(runs: pd.DataFrame, metric: str, arm_a: str, arm_b: str) -> pd.Series:
    """Per-seed ``metric(arm_a) - metric(arm_b)``; seeds missing either arm are dropped."""
    a = runs.loc[runs["arm"] == arm_a].set_index("seed")[metric]
    b = runs.loc[runs["arm"] == arm_b].set_index("seed")[metric]
    return (a - b).dropna()


@dataclass(frozen=True)
class Comparison:
    """One paired test: mean over seeds of metric(arm_a) - metric(arm_b)."""

    key: str  # short id, used for LaTeX macro names
    family: str  # Holm correction is applied within a family
    label: str  # human-readable claim being tested
    metric: str
    arm_a: str
    arm_b: str


def compare(runs: pd.DataFrame, comparisons: list[Comparison], level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Evaluate comparisons on per-seed run rows; comparisons whose arms are absent are skipped."""
    out = []
    for c in comparisons:
        if c.metric not in runs.columns:
            continue
        d = paired_diffs(runs, c.metric, c.arm_a, c.arm_b)
        if d.empty:
            continue
        lo, hi = bootstrap_ci(d.to_numpy(), level=level, seed=seed)
        out.append({"key": c.key, "family": c.family, "label": c.label, "metric": c.metric, "arm_a": c.arm_a, "arm_b": c.arm_b,
                    "n": len(d), "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p": sign_flip_pvalue(d.to_numpy(), seed=seed)})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    for _, idx in df.groupby("family").groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].to_numpy())
    df["significant"] = df["p_holm"] < (1.0 - level)
    return df
