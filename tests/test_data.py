import pandas as pd
import pytest

from rhal.data.synth import SynthConfig, generate_synthetic_dataset
from rhal.data.validate import validate_news, validate_prices


def test_synthetic_deterministic():
    a = generate_synthetic_dataset(SynthConfig(n_days=60, news_per_day=2), 3)
    b = generate_synthetic_dataset(SynthConfig(n_days=60, news_per_day=2), 3)
    assert a["prices"].equals(b["prices"]) and a["news"].equals(b["news"])
    c = generate_synthetic_dataset(SynthConfig(n_days=60, news_per_day=2), 4)
    assert not a["prices"].equals(c["prices"])


def test_ohlc_consistency(synth):
    p = synth["prices"]
    assert (p["high"] >= p[["open", "close"]].max(axis=1)).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1)).all()
    assert (p[["open", "high", "low", "close"]] > 0).all().all()


def test_intraday_matches_daily(synth):
    i, p = synth["intraday"], synth["prices"]
    last = i.groupby("date")["close"].last()
    assert (abs(last.values - p.set_index("date").loc[last.index, "close"].values) < 1e-9).all()
    assert (i["buy_volume"] + i["sell_volume"] == i["volume"]).all()


def test_news_is_informative_and_labelled(synth):
    n, p = synth["news"], synth["prices"]
    assert set(n["provenance"]) == {"clean"}
    assert n["id"].is_unique
    import numpy as np

    r = np.diff(np.log(p["close"].to_numpy()))
    m = n.groupby("date")["true_sentiment"].mean().reindex(p["date"].iloc[:-1]).to_numpy()
    assert np.corrcoef(m, r)[0, 1] > 0.3  # news carries genuine signal about the next-day return
    assert n["text"].str.contains("Nordwind").mean() > 0.5


def test_validators_reject_bad_frames():
    bad = pd.DataFrame(
        {"date": ["2024-01-01"], "open": [1.0], "high": [0.5], "low": [0.9], "close": [1.0], "volume": [1]}
    )
    with pytest.raises(ValueError):
        validate_prices(bad)
    with pytest.raises(ValueError):
        validate_prices(pd.DataFrame({"date": ["2024-01-01"], "open": [1.0]}))
    with pytest.raises(ValueError):
        validate_news(
            pd.DataFrame({"id": ["a", "a"], "date": ["2024-01-01"] * 2, "text": ["long enough text here"] * 2})
        )
    ok = validate_news(pd.DataFrame({"id": ["a"], "date": ["2024-01-01"], "text": ["long enough text here"]}))
    assert ok.loc[0, "provenance"] == "clean"
