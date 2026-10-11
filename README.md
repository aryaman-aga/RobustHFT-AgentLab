# RobustHFT-AgentLab

**A defensive, offline simulation framework for testing and improving LLM-based trading agents under subtle financial misinformation** — with time-series grounding, HFT-inspired microstructure features, an explicit risk-control layer, strict no-lookahead verification, and a placebo-controlled attack metric.

Built as a 4–5 month B.Tech Project (BTP) on top of *AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection* (Liu et al., 2026, arXiv:2605.09185).

> **Safety boundary.** Everything here is a closed simulation. There is no broker connection, no live trading, no public dissemination. All generated misinformation is synthetic, stays inside the simulator, and every record carries `provenance="synthetic"` plus a safety label. Nothing in this repository is investment advice.

---

## What it does in one paragraph

A **MisGen** red-team engine perturbs synthetic financial news through three sub-spaces — behavioural **bias** (agent context), **minor** semantic perturbations (8 types: flipping, sentiment, numerical, temporal, entity, concept, causal, other) and style **rewrite** (news / academic) — gated by similarity and detectability filters and steered by a Pólya-urn history-aware sampler with closed-loop feedback. A **POMDP trading agent** (Perception · Profile · Memory · Action) reads retrieved news and market data and decides Buy/Sell/Hold. Every run drives a **shadow clean agent and a treated agent in lockstep** over the same market path, so Attack Success Rate is a true paired measurement, and a **placebo arm** (meaning-preserving paraphrases) gives the noise floor. Defences are stacked: **temporal grounding** (30-day windows, historical analogs with a strict `e + H ≤ t` admissibility rule, text–market consistency score), **microstructure features** (spread proxy, Garman–Klass RV, order-flow imbalance, volume shock, Amihud illiquidity, range ratio), and a **risk layer** (confidence gate, hold-on-conflict, vol-targeted sizing, stop/take-profit, drawdown throttle). Everything is deterministic, CPU-only, and runs in mock mode without any API key.

## Quick start (fresh folder → results in ~10 minutes)

```bash
# 1. virtual environment (macOS/Linux)
python3 -m venv .venv && source .venv/bin/activate
#    Windows (PowerShell):  python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 2. data, model, tests
python scripts/generate_sample_data.py          # data/sample/{prices,news}.csv (synthetic, labelled clean)
python scripts/train_detector.py                # outputs/models/detector.pkl  (the trained detectability proxy)
python -m pytest tests/ -q                      # full test suite, ~40 s

# 3. experiments
python scripts/run_pipeline.py                  # 18 arms x 100 seeds (~20 min) -> outputs/pipeline/
python scripts/run_ablation.py --group all      # attack / defense / retrieval / kappa -> outputs/ablation/
python scripts/make_report_tables.py            # outputs/report.md + paper/tables/*.tex

# 4. paper (optional; needs a LaTeX install)
make paper
```

Or simply `make all`. Every `make` target uses `.venv/bin/python`; nothing is installed globally.

### Expected console output (abridged)

```
$ python scripts/train_detector.py
saved outputs/models/detector.pkl
train_acc=0.657  threshold=0.499  holdout_clean_fpr=0.100
detection rate by perturbation type (minor / after news rewrite / after academic rewrite):
  flipping   0.19 / 0.14 / 0.12
  ...
$ python scripts/run_pipeline.py
INFO pipeline: config=configs/default.yaml seeds=100 agent=mock llm=mock misgen=mock
INFO pipeline: SAFETY: offline simulation only; all generated text is synthetic and labelled.
INFO pipeline: seed=42 clean__none              MER=0.000 lift=0.00 ASR=0.000 CR=+0.146 DD=-0.054
INFO pipeline: seed=42 injected__none           MER=0.713 lift=1.45 ASR=0.417 CR=+0.171 DD=-0.043
INFO pipeline: seed=42 placebo__none            MER=0.496 lift=1.01 ASR=0.250 CR=+0.076 DD=-0.054
...
```

The exact numbers are reproduced by `seed: 42` in `configs/default.yaml`; see `outputs/report.md` after a run.

## Repository layout

```
configs/            default.yaml (all knobs), ablation.yaml (override matrix)
src/rhal/           the package
  lexicon.py          single source of truth: finance polarity words, antonyms, entities, causes
  config.py           pydantic Config + dotted overrides; LLM keys only from .env
  seeding.py          per-component child seeds (paired comparisons stay paired)
  schemas.py          NewsItem (mandatory provenance), Decision, MarketEvidence, BiasConfig
  data/               synthetic OHLCV+news generator, CSV/yfinance loaders, validators
  features/           core (Eq.19) + microstructure proxies + trailing z-scores (no lookahead)
  grounding/          analog retrieval (e+H<=t), evidence builder, consistency score
  memory/             NewsPool (date-gated), TF-IDF/BM25/embeddings retrieval
  misgen/             operators, Polya-urn sampler, detector (trained), filters, engine (Alg. 1)
  agent/              sentiment, perception, profile, memory, policies (rule/mock/LLM), risk, agent
  llm/                provider abstraction with disk cache; mock, Anthropic, OpenAI-compatible
  backtest/           accounting engine with fees/slippage, next-open execution
  evaluation/         paired Simulation, metrics (MER, lift, ASR, net ASR), report/plots
scripts/            generate_sample_data, train_detector, run_pipeline, run_ablation, make_report_tables
tests/              pytest suite incl. future-mutation no-lookahead tests and accounting invariants
docs/               ARCHITECTURE.md (Mermaid diagrams, data flow, class design), ROADMAP.md (week-wise plan),
                    GLOSSARY.md (terminology)
paper/              conference manuscript (see paper/README.md)
  main.tex            manuscript; every number comes from tables/numbers.tex
  tables/, figures/   GENERATED by scripts/make_report_tables.py -- never hand-edit
reports/midterm/    mid-term BTP report (.md, .tex, .pdf)
frontend/           animated showcase (index.html + app.js + styles.css; data.js is GENERATED by `make frontend`)
data/sample/        synthetic sample CSVs for data.source=csv (written by `make data`, git-ignored)
outputs/            all run artifacts (git-ignored; regenerate with `make all`)
```

Secrets go in `.env` (git-ignored); `.env.example` lists the variables.

## Experiment design

| Arm | Pool | Context | Defence |
|---|---|---|---|
| `clean__*` | clean only | unbiased | per level |
| `placebo__*` | clean + meaning-preserving paraphrases | unbiased | per level |
| `injected__*` | clean + MisGen misinformation, **adaptive** (feedback loop runs against this agent) | attacker bias prompt | per level |
| `frozen__*` | clean + the attack recorded against the undefended agent, **replayed verbatim** | recorded bias prompts | per level |
| `placebo_matched__*` | clean + paraphrases of only the items the recorded attack injected (same count) | unbiased | per level |
| `placebo_bias__*` | as `placebo_matched__*` | recorded bias prompts | per level |

Defence levels: `none` (text-heavy ungrounded agent), `grounded` (temporal grounding + micro features), `grounded_risk` (+ risk layer). Each arm is a paired run: the treated agent's ASR is measured against a shadow clean agent that saw the same market on the same day.

**Common random numbers.** MisGen is seeded by (seed, injection type) — never by arm name — and draws from streams keyed by (day, item), so every arm of a seed shares the attacker's random numbers and renaming an arm cannot change a result. The `frozen__*` arms go further: every defence faces bit-identical documents, so their differences are caused by the defence alone (`frozen__none` reproduces `injected__none` exactly; the pipeline checks this on every seed).

**Statistics.** 100 seeds by default (`experiment.n_seeds`; `--seeds 30` for quick runs). Claims are tested as paired per-seed differences: mean, 95% percentile-bootstrap CI, two-sided sign-flip permutation p-value, Holm correction within each family (`src/rhal/evaluation/stats.py`; comparisons are listed in `scripts/make_report_tables.py`). Results land in `*_stats.csv`, `outputs/report.md` and `paper/tables/stats_*.tex`.

**Matched controls.** The classic placebo paraphrases *every* item and carries no bias prompt, whereas the attack injects only accepted candidates and adds bias prompts. The matched placebos, built from the same tape, split ASR into pool noise, count, bias and meaning: meaning = ASR(`frozen`) − ASR(`placebo_bias`) (`ASR_net_matched`). The matched columns are filled only for arms that face that tape (`frozen__*` and `injected__none`); an adaptive attack on a defended agent injects other items and bias types, so its matched columns are empty. Every placebo variant uses the same paraphrase for each (day, item).

**One-step counterfactual ASR** (`ASR_cf`, `experiment.counterfactual_asr`). Once the treated agent's portfolio and memory drift from the shadow's, later "flips" can come from that drift rather than from the text. Each day a throw-away agent decides with the shadow's memory and portfolio but the treated pool and bias, and is compared with the shadow. It reads only copies, so it changes no other result (tested).

**Metrics.** MER and ASR follow the paper (Eqs. 8–9). We add **MER lift** (MER ÷ synthetic share of the visible pool) and **net ASR** (ASR − placebo ASR), because a doubled retrieval pool alone reshuffles top-k and flips near-threshold decisions; net ASR isolates the effect of *meaning*. Also: signal deviation, ASR|exposed, exposure and flips by perturbation type and by bias, CR, max drawdown, Sharpe-like, turnover, risk-override counts.

**Ablations** (`configs/ablation.yaml`): attack-side (w/o filtering / feedback / bias / minor / rewrite, as in the paper's Table 2), defence-side (none, grounding only, grounding+micro, risk only, full, w/o hold-on-conflict), retrieval breadth k∈{2,4,8}, grounding strength κ∈{0.3,0.7,1.0}.

## Animated showcase (frontend)

`frontend/index.html` is a self-contained, animated walkthrough of the project, built for demos and the viva. It has no external libraries or fonts, works offline, and supports light/dark themes and reduced motion.

* **How it works:** an animated pipeline of one trading day: clean news, the MisGen attacker, retrieval, the agent, grounding, the risk layer, the decision, and the comparison with the clean twin.
* **With vs without:** two synchronised panels replay the same 60 trading days with play/pause, scrubbing and speed controls. Four comparisons: the attack itself, the placebo check, temporal grounding and the risk layer. Each shows decisions popping onto the price chart, ⚡ flips against the clean twin, the text/market/final signal needles, the day's retrieved news as CLEAN / SYNTHETIC / PARAPHRASE cards, and risk-layer vetoes.
* **Results:** animated decomposition of naive ASR (pool noise, count, bias prompts, meaning), net ASR by defence with 95% CIs, and what works and what doesn't.

```bash
make frontend                                   # writes frontend/data.js from a seed-42 replay + the latest outputs/
python -m http.server 8765 --directory frontend # then open http://localhost:8765 (or open index.html directly)
```

Every number on the page comes from `frontend/data.js`; nothing is hand-typed. For safety the page shows only *labels* of generated documents (edit type, rewrite style), never their text.

## Agent modes

| `agent.mode` | What decides | Needs |
|---|---|---|
| `rule` | market evidence only (technical baseline) | nothing |
| `mock` (default) | deterministic surrogate LLM: lexicon sentiment of retrieved docs, bias operators as explicit parameter shifts, grounding shrinkage `s_g = s − κ(1−c)(s − s_mkt)` | nothing |
| `llm` | real LLM via `.env` (`RHAL_LLM_PROVIDER=anthropic|openai`), prompt = Perception+Profile+Memory+Evidence+docs, JSON parsed, disk-cached for reproducibility | API key |

Bias prompts in LLM mode are the paper's Appendix E.2 texts verbatim; in mock mode they map to `BiasConfig` (overconfidence → lower action threshold; loss aversion → ×1.7 downside weight; anchoring → 0.5 blend with yesterday's signal; herding → pull toward recent majority action; confirmation → up-weight agreeing docs).

## Optional extras

```bash
pip install -r requirements-optional.txt     # anthropic, openai, yfinance
cp .env.example .env                          # then set RHAL_LLM_PROVIDER and a key
python scripts/run_pipeline.py --set agent.mode=llm --seeds 1
```
`retrieval.method: embed` uses sentence-transformers if installed, else silently falls back to TF-IDF.
Real historical prices can be loaded with `data.source: csv` (any OHLCV CSV with `date,open,high,low,close,volume`) or via `rhal.data.loaders.download_yfinance`.

## Troubleshooting

* **`ModuleNotFoundError: rhal`** → `pip install -e .` inside the venv, or run scripts from the repo root (they add `src/` to `sys.path`).
* **`not enough history`** → `data.n_days − data.eval_days` must be ≥ 100 (warm-up for 60-day trailing normalisation + 30-day windows).
* **Slow first test run** → the session-scoped fixtures train a detector once (~3 s); subsequent tests are fast.
* **Different numbers than the README** → check `seed`, `configs/default.yaml`, and that `outputs/models/detector.pkl` was trained with the same seed (delete it to retrain).
* **matplotlib display errors** → plots use the Agg backend; no display is needed.
* **pandas ≥ 3 warnings** → the code is written for pandas 3 (copy-on-write); older pandas 2.x also works.
* **LLM mode returns Hold everywhere** → the provider's output could not be parsed as JSON; inspect `outputs/llm_cache/*.json`.

## Reading the code

Start with `docs/ARCHITECTURE.md`, then `src/rhal/evaluation/experiment.py::Simulation.run` (the whole loop is ~60 lines), then `misgen/engine.py` and `agent/policy.py`.

## Citation of the base paper

Liu, Z., Yu, Y., Cao, Y., et al. *AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection.* arXiv:2605.09185, 2026.
