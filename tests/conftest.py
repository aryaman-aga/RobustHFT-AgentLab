from __future__ import annotations

import pytest

from rhal.config import Config, load_config
from rhal.data.synth import SynthConfig, generate_synthetic_dataset
from rhal.evaluation.experiment import Dataset, prepare_dataset, train_detector


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    """Fast config: 180 days history, 12 eval days, 3 news/day."""
    return load_config(
        "configs/default.yaml",
        {"data.n_days": 180, "data.eval_days": 12, "data.news_per_day": 3, "experiment.n_seeds": 1},
    )


@pytest.fixture(scope="session")
def small_dataset(small_cfg) -> Dataset:
    return prepare_dataset(small_cfg, 42)


@pytest.fixture(scope="session")
def detector(small_cfg):
    return train_detector(small_cfg, 42)


@pytest.fixture(scope="session")
def synth():
    return generate_synthetic_dataset(SynthConfig(n_days=150, news_per_day=3, intraday=True, bars_per_day=12), 7)
