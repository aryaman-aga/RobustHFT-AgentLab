import os

from rhal.config import Config, load_config


def test_defaults_and_overrides():
    cfg = load_config("configs/default.yaml", {"grounding.enabled": False, "misgen.alpha": 2.5, "risk.stop_loss": 0.03})
    assert cfg.grounding.enabled is False
    assert cfg.misgen.alpha == 2.5
    assert cfg.risk.stop_loss == 0.03


def test_llm_keys_never_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 1\nllm:\n  provider: anthropic\n  api_key: SHOULD_BE_IGNORED\n")
    os.environ.pop("RHAL_LLM_PROVIDER", None)
    cfg = load_config(p)
    assert cfg.llm.api_key != "SHOULD_BE_IGNORED"
    assert cfg.llm.provider in ("mock", "anthropic", "openai")


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, Config)
    assert cfg.seed == 42


def test_with_overrides_is_pure():
    a = Config()
    b = a.with_overrides({"seed": 7})
    assert a.seed == 42 and b.seed == 7
