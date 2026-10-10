# CLAUDE.md — working notes for AI assistants on this repo

## What this is
RobustHFT-AgentLab (`rhal`): an offline, deterministic research framework (B.Tech project) that red-teams a simulated LLM trading agent with synthetic financial misinformation and measures how temporal grounding, microstructure features and a risk layer improve robustness. Base paper: AutoRedTrader (arXiv:2605.09185). Extensions and all implementation simplifications are documented in `docs/ARCHITECTURE.md` §"Provenance of ideas".

## Hard constraints — never violate
- **Offline only.** No broker, no order routing, no live-trading code, ever. `yfinance` is the only network path and is optional/read-only.
- **Synthetic text stays synthetic.** Anything produced by `rhal.misgen` must carry `provenance="synthetic"|"placebo"` and the default `safety_label`. Never write generated misinformation to a location that looks like real news; never post it anywhere.
- **No lookahead.** Any feature/evidence/decision at index `t` may only depend on rows `<= t`. Analogs must satisfy `end_idx + horizon <= t`. Normalisation is trailing, never full-sample. `tests/test_features_nolookahead.py` and `test_pipeline_smoke.py::test_decision_does_not_depend_on_future` enforce this by *mutating the future* — keep those tests passing and add a similar mutation test for any new feature.
- **Determinism.** All randomness goes through `rhal.seeding.rng(master, *tags)`. Never call `np.random.*` global state or `random`. Two runs with the same config+seed must produce identical `metrics`.
- **Keys only from `.env`.** `Config` deliberately drops any `llm:` block from YAML.
- **No GPU, no training of large models.** The only trained component is the sklearn `DetectabilityModel`.

## Environment
- Use the venv: `.venv/bin/python` (Python 3.14, pandas 3.0, numpy 2.5, scikit-learn 1.9). `make install` creates it. Never install globally.
- Run tests with `.venv/bin/python -m pytest tests/ -q` (~40 s). Run everything with `make all`.
- Machine note: keep concurrency low (RAM-limited). Scripts are single-process by design; don't parallelise seeds with multiprocessing without asking.

## Layout and where things live
- One knob = one place: `configs/default.yaml` ↔ `rhal/config.py` (pydantic). Ablations are dotted overrides in `configs/ablation.yaml`; `Config.with_overrides` applies them. If you add a config field, add it to **both** the YAML and the pydantic model.
- Finance vocabulary lives only in `rhal/lexicon.py` (used by the data generator, sentiment scorer, operators and detector). Don't duplicate word lists.
- The experiment loop is `rhal/evaluation/experiment.py::Simulation.run`. It runs a shadow (clean) and a treated agent in lockstep; ASR is paired per step. Keep it that way — un-paired ASR is meaningless here.
- Step records are plain dicts (`Simulation._rec`); `evaluation/metrics.py` consumes them. Add new per-step fields there, then to metrics, then to `report.KEY_COLS` if they should be tabulated.
- LaTeX tables in `paper/tables/` and figures in `paper/figures/` are **generated** by `scripts/make_report_tables.py`; never hand-edit them. Edit prose in `paper/main.tex`. `paper/ieee_paper.tex` is an older hand-typed draft (stale numbers) — see `paper/README.md`.
- `outputs/` is git-ignored; anything the paper needs must be copied into `paper/` by the report script. Docs live in `docs/`, BTP reports in `reports/<phase>/`.

## Conventions
- Type hints + docstrings on public functions; module docstrings explain the *why* and cite the paper equation/section where relevant.
- Prefer simple, explicit code a BTP student can read over clever abstractions. Comment important logic, mark open assumptions with `TODO:`.
- Feature names are fixed regardless of window params (`vol_5`, `dd_20`, ...) — downstream code depends on the names.
- Metrics that are percentages are stored as fractions (0–1) and formatted as % only in reports.
- When you change an operator in `misgen/operators.py`, re-run `scripts/train_detector.py` (the detector is trained on Phi_Minor outputs, not on Phi_Rewrite outputs — that asymmetry is intentional and documented).

## Common tasks
- Add a perturbation type: implement `op_<name>` in `misgen/operators.py`, register in `_MINOR` and `MINOR_TYPES`, add a test in `tests/test_misgen.py` that it changes the text and (if directional) the sentiment sign.
- Add a defence: add a knob to `RiskLayer` or `SurrogatePolicy`, thread it through `Simulation._agent`, add an ablation entry, add a unit test in `tests/test_risk.py`.
- Add a metric: compute in `evaluation/metrics.py`, add to `KEY_COLS`, add a hand-computed case to `tests/test_metrics.py`.
- Use a real LLM: `cp .env.example .env`, set provider+key, `--set agent.mode=llm`. Responses are cached in `outputs/llm_cache/` keyed by prompt hash so re-runs are free and reproducible.

## Known limitations (be honest about these in any write-up)
- The mock agent is a transparent surrogate, not an LLM; LLM-mode results may differ. The harness is identical in both modes.
- The detectability proxy is a small feature-based classifier (train acc ≈ 0.66); it is a stand-in for the paper's LLM detector.
- Placebo ASR is non-trivial: adding *any* documents to the pool reshuffles top-k and flips near-threshold decisions. That is why net ASR is reported.
- Stop-loss/take-profit are evaluated at the daily close, not intraday.
- Synthetic data only by default; the CSV loader accepts real OHLCV but the news generator is template-based.
