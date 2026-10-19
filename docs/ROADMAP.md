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
- Local LLM through Ollama, installed inside `.venv`: reader mode and decider mode, with loud failures.
- Real BTC-USD prices, one historical window per seed.
- Baselines: text-free agent, detection filter, robust aggregation.
- Defense-aware attackers: evidence-aligned and conflict-seeking.
- Shrink-on-conflict rule and position-gap metric.
- Price-impact stress test.
- Generalization figure.
- Strict config validation, ruff formatting, refreshed docs.

## Next, up to 20 Nov

| Window | Work |
|---|---|
| 19–26 Oct | Collect the running experiments; regenerate tables and figures; write the paper sections on the LLM replication, real prices, baselines, aware attackers and price impact |
| 27 Oct – 2 Nov | Decider results; update the discussion and limitations; internal read-through |
| 3–9 Nov | AAAI-27 template build (`paper/aaai/`): 7 pages plus appendix, anonymised |
| 10–20 Nov | Polish, final checks (`make lint test report paper`), submit on OpenReview |

## Later, for an archival venue

- Replicate with a frontier API model, cached for reproducibility.
- Real news with an LLM-based attacker. The rule-based operators need template fields, so real headlines need LLM edits.
- Adaptive (online) price impact, where the market reacts while the attacker adapts.
- More assets and longer evaluation windows.
