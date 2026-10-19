"""Real-price mode (data.source=real): validation repair, per-seed windows, news, analog scale.

Uses a synthetic stand-in CSV, so the tests need no network and no downloaded data.
"""
import numpy as np
import pandas as pd
import pytest

from rhal import lexicon as L
from rhal.data.synth import SynthConfig, generate_synthetic_dataset
from rhal.data.validate import validate_prices
from rhal.evaluation.experiment import Dataset, RunSpec, Simulation, prepare_dataset, real_window


@pytest.fixture(scope="module")
def real_csv(tmp_path_factory):
    prices = generate_synthetic_dataset(SynthConfig(n_days=700, news_per_day=1), 3)["prices"]
    path = tmp_path_factory.mktemp("real") / "prices.csv"
    prices[["date", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def real_cfg(small_cfg, real_csv):
    return small_cfg.with_overrides({"data.source": "real", "data.real_prices": real_csv, "data.n_days": 180, "data.eval_days": 12})


def test_repair_fixes_float_noise_and_rejects_real_errors():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "open": [100.0, 101.0, 102.0],
                       "high": [101.0, 100.9999, 103.0], "low": [99.0, 100.5, 101.0], "close": [100.5, 101.0, 102.5], "volume": [1e6, 1e6, 1e6]})
    with pytest.raises(ValueError):
        validate_prices(df)  # strict mode: high 100.9999 < open 101
    fixed = validate_prices(df, repair=True)
    assert fixed.loc[1, "high"] == 101.0
    bad = df.copy()
    bad.loc[1, "high"] = 95.0  # 6% below the open: a data error, not float noise
    with pytest.raises(ValueError, match="inconsistent"):
        validate_prices(bad, repair=True)
    zero = df.copy()
    zero.loc[2, "volume"] = 0.0
    with pytest.raises(ValueError, match="volume"):
        validate_prices(zero, repair=True)


def test_real_windows_are_distinct_and_deterministic():
    full = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=500).strftime("%Y-%m-%d"), "close": np.arange(500.0)})
    w = [real_window(full, 200, 50, i) for i in range(8)]
    n_win = (500 - 200) // 50 + 1
    ends = [x["date"].iloc[-1] for x in w[:n_win]]
    assert len(set(ends)) == n_win and all(len(x) == 200 for x in w)
    assert w[0]["date"].iloc[-1] == full["date"].iloc[-1]  # index 0 is the latest window
    assert w[n_win]["date"].tolist() == w[0]["date"].tolist()  # indices wrap around
    tails = [set(x["date"].iloc[-50:]) for x in w[:n_win]]  # evaluation segments never overlap
    assert all(not (tails[i] & tails[j]) for i in range(n_win) for j in range(i + 1, n_win))


def test_real_mode_dataset_per_seed(real_cfg):
    a, a2, b = prepare_dataset(real_cfg, 42), prepare_dataset(real_cfg, 42), prepare_dataset(real_cfg, 43)
    assert a.dates == a2.dates and a.news["text"].tolist() == a2.news["text"].tolist()  # deterministic
    assert a.dates[-1] != b.dates[-1]  # a different window for the next seed
    # the template news is about the fictional company, never the real asset
    assert a.news["text"].str.contains(L.ENTITY["company"]).mean() > 0.5


def test_auto_analog_scale_uses_only_pre_window_prices(real_cfg, detector):
    cfg = real_cfg.with_overrides({"grounding.analog_scale": "auto"})
    ds = prepare_dataset(cfg, 42)
    sim = Simulation(cfg, ds, detector, 42)
    prices = ds.prices.copy()
    prices.loc[sim.t0 + 1:, ["open", "high", "low", "close"]] *= 3.0  # change every price after t0
    mutated = Dataset(prices, ds.news, ds.intraday, ds.feats, ds.dates)
    assert Simulation(cfg, mutated, detector, 42).analog_scale == sim.analog_scale
    assert 0.0 < sim.analog_scale < 0.5


def test_real_mode_runs_end_to_end(real_cfg, detector):
    cfg = real_cfg.with_overrides({"grounding.analog_scale": "auto"})
    r = Simulation(cfg, prepare_dataset(cfg, 42), detector, 42).run(RunSpec("injected__grounded", "misgen", True, False, True))
    assert 0.0 <= r.metrics["ASR"] <= 1.0 and r.metrics["MER"] > 0
