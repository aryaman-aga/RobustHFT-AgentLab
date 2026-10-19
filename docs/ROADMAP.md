# Status and roadmap

Last updated 19 Oct 2026. Target: AAAI-27 FinFM workshop, submission deadline 20 Nov 2026 (AAAI template, 7 pages plus references and appendix, double-blind). Archival follow-up: IEEE CAI 2027 (call expected for about 31 Mar 2027).

## Done

**Framework**
- Synthetic market with template news.
- Trading agent: perception, profile, memory and action, with a lexicon surrogate policy.
- MisGen attacker: bias, minor-edit and rewrite operators, similarity and detector filters, Pólya urn.
- Paired twin simulation.

**Defense layer**
- Market evidence with leak-free historical analogs.
- Temporal grounding.
- Risk layer.

**Fair measurement**
- Placebo ladder.
- Frozen-attack replay.
- Common random numbers.
- One-step counterfactual ASR.
- Paired statistics: bootstrap CIs, sign-flip permutation tests, Holm correction per family.
- No-lookahead mutation tests for prices and for text.

**Results and paper**
- 100-seed results, re-run after the text-lookahead fix.
- IEEE paper (8 pages), merged to `main` in PR #3.

**Reviewer round** (branch `feat/reviewer-fixes`)
- Local LLM through Ollama, installed inside `.venv`: reader mode (30 seeds) and decider mode (smoke-tested only), with loud failures.
- Real BTC-USD prices, one historical window per seed.
- Baselines: text-free agent, detection filter, robust aggregation.
- Defense-aware attackers: evidence-aligned and conflict-seeking.
- Shrink-on-conflict rule and position-gap metric.
- Price-impact stress test.
- Generalization figure and main-text baseline table.
- Paper sections on the baselines, the generalization tests (LLM reader, BTC-USD, aware attackers, price impact) and the shrink fix.
- Strict config validation, ruff formatting, refreshed docs.

## Next, up to 20 Nov

| Window | Work |
|---|---|
| 19–26 Oct | AAAI-27 build (`paper/aaai/`): anonymised, 7 pages plus appendix; OpenReview profiles for all three authors (institutional e-mail) |
| 27 Oct – 2 Nov | Internal read-through by all authors |
| 3–9 Nov | Revisions from the read-through |
| 10–20 Nov | Polish, final checks (`make lint test report paper`), submit on OpenReview |

## Later, for an archival venue

- LLM decider at scale (10 seeds, about 20 h on the laptop; run overnight with nothing else running).
- Conflict-seeking attacker against no defense and grounding alone (two arms in `aware_attack`, about 2.5 h).
- Replicate with a frontier API model, cached for reproducibility.
- Real news with an LLM-based attacker. The rule-based operators need template fields, so real headlines need LLM edits.
- Adaptive (online) price impact, where the market reacts while the attacker adapts.
- More assets and longer evaluation windows.
