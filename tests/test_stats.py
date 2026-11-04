"""Hand-computed cases for the paired statistics (evaluation/stats.py)."""

import numpy as np
import pandas as pd

from rhal.evaluation.stats import Comparison, bootstrap_ci, compare, holm, paired_diffs, sign_flip_pvalue


def test_bootstrap_ci_degenerate_and_ordering():
    assert bootstrap_ci([0.2] * 10) == (0.2, 0.2)  # no spread -> point interval
    assert bootstrap_ci([0.5]) == (0.5, 0.5)
    lo, hi = bootstrap_ci(np.arange(30) / 30)
    assert lo < np.mean(np.arange(30) / 30) < hi
    assert bootstrap_ci(np.arange(30) / 30) == (lo, hi)  # deterministic (rhal.seeding.rng)


def test_sign_flip_exact_by_hand():
    # d = (1, 1, 1): of the 8 sign patterns only +++ and --- reach |mean| = 1  -> p = 2/8
    assert sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25
    # d = (1, -1): observed mean 0, every pattern is at least as extreme -> p = 1
    assert sign_flip_pvalue([1.0, -1.0]) == 1.0
    # d = (3, 1): patterns give |mean| 2, 1, 1, 2 -> p = 2/4
    assert sign_flip_pvalue([3.0, 1.0]) == 0.5


def test_sign_flip_monte_carlo_detects_large_effect():
    d = np.full(30, 0.1) + np.linspace(-0.01, 0.01, 30)  # all positive, n > exact limit
    p = sign_flip_pvalue(d)
    assert p == 1 / 20_001  # no random flip beats the observed all-positive mean
    assert sign_flip_pvalue(np.concatenate([d, -d])) > 0.5  # symmetric around 0 -> no effect


def test_holm_by_hand():
    # sorted p: 0.01*3 = 0.03, 0.03*2 = 0.06, max(0.06, 0.04*1) = 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = holm([0.5, np.nan, 0.6])  # NaN not counted in m = 2
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [1.0, 1.0])


def test_paired_diffs_aligns_on_seed():
    runs = pd.DataFrame({"arm": ["a", "a", "a", "b", "b"], "seed": [1, 2, 3, 2, 1], "ASR": [0.5, 0.4, 0.9, 0.1, 0.2]})
    d = paired_diffs(runs, "ASR", "a", "b")
    assert d.to_dict() == {1: 0.5 - 0.2, 2: 0.4 - 0.1}  # seed 3 has no partner -> dropped


def test_compare_end_to_end():
    seeds = range(20)
    runs = pd.DataFrame(
        [
            {"arm": arm, "seed": s, "ASR": base + 0.01 * (s % 3)}
            for s in seeds
            for arm, base in (("inj", 0.30), ("plc", 0.20), ("other", 0.30))
        ]
    )
    comps = [
        Comparison("net", "f", "inj vs plc", "ASR", "inj", "plc"),
        Comparison("null", "f", "inj vs other", "ASR", "inj", "other"),
        Comparison("missing", "f", "absent arm", "ASR", "inj", "nope"),
    ]
    st = compare(runs, comps).set_index("key")
    assert "missing" not in st.index  # comparisons without data are skipped, not crashed on
    assert np.isclose(st.loc["net", "mean_diff"], 0.10) and st.loc["net", "ci_lo"] <= 0.10 <= st.loc["net", "ci_hi"]
    assert st.loc["net", "significant"] and not st.loc["null", "significant"]
    assert st.loc["null", "mean_diff"] == 0.0 and st.loc["null", "p"] == 1.0
    assert (st["p_holm"] >= st["p"]).all() and st.loc["net", "n"] == 20


def test_paired_diffs_with_second_metric():
    runs = pd.DataFrame({"arm": ["a", "a"], "seed": [1, 2], "ASR": [0.5, 0.4], "ASR_cf": [0.3, 0.4]})
    d = paired_diffs(runs, "ASR", "a", "a", metric_b="ASR_cf")  # within-arm ASR - ASR_cf
    assert np.allclose(d.to_numpy(), [0.2, 0.0])
    st = compare(runs, [Comparison("gap", "f", "ASR - cf", "ASR", "a", "a", metric_b="ASR_cf")])
    assert np.isclose(st.loc[0, "mean_diff"], 0.1) and st.loc[0, "metric_b"] == "ASR_cf"
