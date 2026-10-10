# 4–5 month BTP roadmap

Assumes ~10–12 h/week. Each week ends with a committed, tested state. Fallback scope is listed per month so the project always has a defensible deliverable.

## Month 1 — Literature + baseline reproduction
| Week | Goal | Deliverable | Risk / fallback |
|---|---|---|---|
| 1 | Read AutoRedTrader closely; annotate Eqs. 1–19, Alg. 1, App. E | 2-page summary; table of paper vs. plan | — |
| 2 | Survey: FinMem, InvestorBench, FinCon, TradingAgents, TradeTrap; adversarial text (ArtPrompt, AutoRedTeamer, LIFE); behavioural finance (Barber–Odean, prospect theory, herding) | `paper/references.bib` with 25+ entries; related-work draft | — |
| 3 | Set up repo, venv, CI-style `make test`; run mock pipeline; understand every module | All 61 tests green on your machine | If Python/packaging issues: pin versions from `requirements.txt` |
| 4 | Reproduce the paper's *structure*: clean vs injected, MER/ASR/CR on synthetic data; sanity-check determinism | `outputs/pipeline/` tables; 1-page "what matches / what differs" note | If numbers look degenerate, tune `rho` and `action_threshold` and document it |

## Month 2 — Simulator + MisGen
| Week | Goal | Deliverable | Risk / fallback |
|---|---|---|---|
| 5 | Study `misgen/operators.py`; add 2–3 new template families to `data/synth.py` (macro, M&A) so operators have richer targets | Tests for new templates | Keep templates minimal if time is short |
| 6 | Detector: add/replace features; measure per-type detection and post-rewrite laundering (`scripts/train_detector.py`) | Detector table (App. of paper) | Feature-based detector is acceptable; skip trained-embedding detector |
| 7 | Feedback loop analysis: plot strategy probabilities over days; verify Pólya-urn reinforcement matters (w/o feedback ablation) | Figure: strategy mix over time | — |
| 8 | (Optional) LLM-mode MisGen with a cheap model on 1 seed; compare ASR to mock | Cached responses in `outputs/llm_cache/` | Fallback: mock only, note as limitation |

## Month 3 — Grounding + HFT features + risk controls
| Week | Goal | Deliverable | Risk / fallback |
|---|---|---|---|
| 9 | Grounding: κ sweep; analog window/horizon sweep; verify admissibility test | Robustness–return frontier figure | — |
| 10 | Microstructure: intraday mode on; compare OFI from tick rule vs close-location proxy | Table: grounded vs grounded+micro | If intraday is slow, use daily proxies |
| 11 | Risk layer: calibrate thresholds; override-count analysis; add trailing-stop variant | Defence ablation table | — |
| 12 | Memory design: `adaptive_trust` on/off; reflection quality | Memory ablation row | Skip if behind schedule |

## Month 4 — Experiments + ablations
| Week | Goal | Deliverable | Risk / fallback |
|---|---|---|---|
| 13 | Full pipeline, 10 seeds; attack ablation (Table 2 analogue) | `outputs/ablation/attack` | 5 seeds if runtime is an issue |
| 14 | Defence ablation + retrieval-k sweep; net-ASR analysis; exposure-by-type | `outputs/ablation/defense`, figures | — |
| 15 | Real-data sanity run: BTC or an equity via `yfinance` CSV with synthetic news attached to real prices | Appendix table | Purely optional |
| 16 | Statistical treatment: paired differences with CIs; write `results.md` | Results draft | — |

## Month 5 — Paper, polish, viva
| Week | Goal | Deliverable |
|---|---|---|
| 17 | Write `paper/main.tex` sections Method/System/Setup; generate tables via `make_report_tables.py` | Full draft |
| 18 | Results/Ablation/Limitations/Ethics; figures polished; advisor review | Revised draft |
| 19 | Slides (architecture, novelty, results, limitations); demo script (`make all`) | 15-slide deck |
| 20 | Viva prep: expected questions (see below); final tag `v1.0` | Submission |

## Milestones
M1 (wk 4): baseline reproduced, tests green. M2 (wk 8): MisGen characterised. M3 (wk 12): all defences implemented and ablatable. M4 (wk 16): final experiments frozen. M5 (wk 20): paper + viva.

## Top risks
1. **Mock agent is "too simple"** → frame as a transparent surrogate; run LLM mode on ≥1 seed to show the harness transfers.
2. **Placebo ASR close to attack ASR** → this is a *finding*; report net ASR and discuss retrieval-slot displacement.
3. **Grounding costs too much clean return** → present as a frontier and pick κ on the knee.
4. **Time** → the fallback scope after Month 3 is already a complete, testable system.

## Expected viva questions
* Why is `e + H ≤ t` necessary for analogs? What breaks with `e < t`?
* Why fit TF-IDF once on a background corpus? What artifact does per-query fitting cause?
* Why train the detector on Φ_Minor only? What would change if trained on Φ_Rewrite too?
* What does net ASR remove, and why can naive ASR overstate attack strength?
* How is each behavioural bias operationalised numerically, and is that faithful to the prompt?
* Why next-open execution rather than same-close? Which lookahead does it prevent?
* What would you need to run this on real news, and what ethical controls would you add?
