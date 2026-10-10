import numpy as np

from rhal.misgen.sampler import StrategySampler


def test_uniform_without_history():
    s = StrategySampler(["a", "b", "c"], alpha=1.0)
    assert np.allclose(s.probabilities(), [1 / 3] * 3)


def test_polya_urn_formula():
    s = StrategySampler(["a", "b"], alpha=2.0)
    s.reinforce("a", 3)
    # w_a = 1 + 2*3 = 7, w_b = 1 -> P(a) = 7/8
    assert np.allclose(s.probabilities(), [7 / 8, 1 / 8])


def test_feedback_off_ignores_history():
    s = StrategySampler(["a", "b"], alpha=2.0, use_feedback=False)
    s.reinforce("a", 10)
    assert np.allclose(s.probabilities(), [0.5, 0.5])


def test_return_dependent_bias_adjustment():
    s = StrategySampler(["overconfidence", "loss_aversion", "herding"], alpha=1.0)
    p_up = s.probabilities(cr=1.2, delta=0.3)
    p_dn = s.probabilities(cr=0.8, delta=0.3)
    assert p_up[0] > p_up[1] and p_dn[1] > p_dn[0]
    assert abs(p_up.sum() - 1) < 1e-12 and abs(p_dn.sum() - 1) < 1e-12


def test_sampling_matches_probabilities():
    s = StrategySampler(["a", "b"], alpha=1.0)
    s.reinforce("a", 9)  # P(a) = 10/11
    g = np.random.default_rng(0)
    draws = [s.sample(g) for _ in range(3000)]
    assert abs(draws.count("a") / 3000 - 10 / 11) < 0.03
