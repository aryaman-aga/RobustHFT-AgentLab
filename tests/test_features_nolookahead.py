"""The flagship test: mutate the FUTURE and assert the PAST is byte-identical."""

import numpy as np

from rhal.features import build_feature_frame
from rhal.features.windows import feature_columns, market_signal
from rhal.grounding import EvidenceBuilder


def _frame(prices, intraday):
    f = build_feature_frame(prices, intraday, micro=True)
    f["close"] = prices["close"].to_numpy()
    return f


def test_features_do_not_use_future(synth):
    p, i = synth["prices"], synth["intraday"]
    cut = 100
    p2 = p.copy()
    p2.loc[cut:, ["open", "high", "low", "close"]] *= 5.0
    p2.loc[cut:, "volume"] *= 50.0
    f1, f2 = _frame(p, i), _frame(p2, i)
    cols = [c for c in f1.columns if c != "date"]
    a, b = f1.loc[: cut - 1, cols].to_numpy(float), f2.loc[: cut - 1, cols].to_numpy(float)
    assert np.array_equal(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))
    # and the future rows DO differ (the mutation reached the features)
    assert not np.allclose(
        np.nan_to_num(f1.loc[cut + 25 :, cols].to_numpy(float)), np.nan_to_num(f2.loc[cut + 25 :, cols].to_numpy(float))
    )


def test_no_full_sample_normalisation(synth):
    """Trailing z-score must not depend on the sample mean over the whole series."""
    p, i = synth["prices"], synth["intraday"]
    f1 = _frame(p, i)
    f2 = _frame(p.iloc[:120].reset_index(drop=True), i[i["date"].isin(p["date"].iloc[:120])])
    zc = ["z_" + c for c in feature_columns(True)]
    a, b = f1.loc[:119, zc].to_numpy(float), f2.loc[:119, zc].to_numpy(float)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_evidence_and_market_signal_do_not_use_future(synth):
    p, i = synth["prices"], synth["intraday"]
    cut = 110
    p2 = p.copy()
    p2.loc[cut:, ["open", "high", "low", "close"]] *= 0.2
    zc = ["z_" + c for c in feature_columns(True)]
    e1 = EvidenceBuilder(_frame(p, i), zc, micro=True, window=30, horizon=5, top_k=5)
    e2 = EvidenceBuilder(_frame(p2, i), zc, micro=True, window=30, horizon=5, top_k=5)
    for t in (cut - 1, cut - 3, cut - 10):
        a, b = e1.build(t), e2.build(t)
        assert a.market_signal == b.market_signal
        assert a.analog_dates == b.analog_dates
        assert a.analog_mean_forward_return == b.analog_mean_forward_return


def test_market_signal_bounded_and_monotone():
    z = {
        f"z_{k}": 0.0
        for k in ["mom_5", "trend_div", "mom_20", "anomaly", "ofi_proxy", "volume_shock", "spread_proxy", "rv_gk"]
    }
    assert market_signal(z) == 0.0
    up = dict(z, z_mom_5=2.0, z_trend_div=2.0)
    dn = dict(z, z_mom_5=-2.0, z_trend_div=-2.0)
    assert -1 < market_signal(dn) < 0 < market_signal(up) < 1
    assert market_signal(dict(z, z_ofi_proxy=1.0), micro=False) == 0.0  # micro ignored when disabled
