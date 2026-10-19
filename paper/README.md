# paper/

Manuscript **"RobustHFT-AgentLab: A Defense Layer Against Misinformation Attacks on LLM-Based Trading Agents"** by Aryaman Agarwal, Arusha Habib and Ayana Kumar (NSUT, New Delhi). The defense layer (evidence model, temporal grounding, risk layer) is the main contribution; placebo controls are the measurement tool that credits it only for misinformation-driven flips.

There are two builds of one text:

| Build | File | Format | Use |
|---|---|---|---|
| AAAI-27 | `aaai/main.tex` | AAAI-27 author kit, anonymous, 7 pages + references + technical appendix | Submission to the FinFM workshop (OpenReview, deadline 20 Nov 2026 AoE) |
| IEEE | `main.tex` | IEEEtran conference, with author names | Named preprint and later archival venues |

Both read the same prose (`sections/*.tex`), macros (`preamble.tex`), bibliography, generated tables and figures, so an edit lands in both.

| File | What it is |
|---|---|
| `sections/*.tex` | All prose: abstract, one file per section, ethics, reproducibility and the technical appendix. **Edit the text here.** Every number comes from generated `\val{...}` macros, so it always matches the latest run. |
| `preamble.tex` | Shared packages and macros (`\val`, `\cmp`, `\pholm`, `\rhaltable`). Class-specific packages (fonts, citations) are in each main file. |
| `main.tex`, `aaai/main.tex` | Thin wrappers: document class, title and authors, section order, bibliography style. |
| `aaai/aaai2027.sty`, `aaai/aaai2027.bst` | Unmodified files from the official AAAI-27 author kit. `aaai/references.bib` is a symlink to `references.bib`. |
| `aaai/draft.tex` | Local draft wrapper for tectonic (see Build). |
| `aaai/openreview.md` | **Generated** text for the OpenReview form: title, keywords, TL;DR and the abstract with all numbers filled in (`scripts/export_openreview.py`). |
| `references.bib` | Bibliography. Every entry was checked against the publisher, the ACL Anthology, NeurIPS proceedings, arXiv or the DOI registry (Oct 2026). Acronyms are brace-protected because IEEEtran sentence-cases titles; an `IEEEtranBSTCTL` entry switches on "first author et al." for more than six authors (IEEE build only). |
| `figures/fig_harness.tex` | TikZ diagram of the defense layer and the paired harness (Fig. 1), drawn at its natural size so its fonts match the caption. Algorithm 1 is written in `sections/defense.tex`. |
| `figures/fig_*.pdf` | **Generated** vector figures from `scripts/make_paper_figures.py`: frontier, replay, defense components, ASR decomposition, and `fig_generalization.pdf` (grounding's gain across the LLM reader, real prices, the aligned attacker and price impact; rows appear as their runs exist). Fonts are embedded TrueType (no Type 3). PDFs carry no timestamp, so identical data gives identical files. |
| `tables/numbers.tex`, `tables/stats_*.tex`, `tables/detector.tex`, `tables/tab_baselines.tex` | **Generated** by `scripts/make_report_tables.py` from `outputs/`: `stats_main`, `stats_controls`, `stats_ablation`, `stats_baselines`, `stats_aware`, `stats_conflict`, `stats_impact` (price impact) and `stats_replications` (LLM reader, BTC-USD). `tab_baselines` is the main-text baseline table. |
| `tables/tab_main.tex`, `tables/figmacros.tex` | **Generated** by `scripts/make_paper_figures.py` (Table 1 and numbers that only the figures compute). |
| `tables/main_results.tex`, `tables/ablation_*.tex`, `tables/asr_decomposition.tex` | **Generated** mean ± sd tables. The paper does not use them; they are kept for the BTP reports. |

Never hand-edit a generated file: change the script and run `make report`. Generated tables call `\rhaltable{caption}{body}`, which puts the caption above the table in the IEEE build and below it in the AAAI build, as each style requires.

## Build

- `make paper`: IEEE build, `paper/main.pdf` (tectonic if installed, else `pdflatex` + `bibtex`). The preamble detects the engine: TeX Gyre Termes through fontspec under XeTeX, newtx under pdfLaTeX.
- `make paper-aaai`: AAAI build. `aaai2027.sty` accepts only pdfLaTeX. With pdfLaTeX installed this builds `aaai/main.pdf`; otherwise it compiles `aaai/draft.tex` with tectonic into `aaai/draft.pdf`. The draft wrapper skips only the style's engine check and reads `main.tex` unchanged, so drafts and the real build differ at most by a few line breaks.
- `make paper-aaai-zip`: flat, Overleaf-ready source zip `aaai/aaai27_source.zip`, plus `aaai/openreview.md`. **The submitted PDF must come from pdfLaTeX**: upload the zip to Overleaf, compile `main.tex` with pdfLaTeX, check that it still fits 7 pages, and submit that PDF.

A clean build has no overfull boxes, no undefined citations and no `rhal: missing generated value` warnings; check the `.log` for all three. PDFs and zips are build products and are git-ignored.

Refresh everything after new experiments: `make pipeline ablation price-impact report`, then the builds. The LLM and real-price replications are separate runs (see the root README); `make report` picks them up from `outputs/llm/` and `outputs/real_btc/`.

## Anonymity

- The AAAI build is anonymous: the style's `submission` option prints "Anonymous submission", and `\anonymoustrue` replaces the repository URL with "provided as supplementary material". The draft's text and PDF metadata contain no names, affiliation, e-mail or URL (checked 20 Oct 2026). Check this again on the Overleaf PDF before submitting.
- The IEEE build prints the authors (`\anonymousfalse`); the e-mail line still reads "TBD".
- The repository is private and its git history and BTP reports name the authors. Give reviewers `make supplementary` instead: `aaai/aaai27_supplementary.zip` holds the tracked code, configs, tests, docs and frontend with the identifying README lines removed, and the script refuses to write the zip if any author name, affiliation or repository owner is left.

## Results in the paper

All results come from paired seeds: 100 for the synthetic experiments, 67 BTC-USD windows, 30 seeds for the LLM reader. Every claim in the prose is a paired comparison listed in `scripts/make_report_tables.py` and reported in the appendix tables. Effects that are not detectable (CI includes zero or p_Holm ≥ 0.05) are described as such. Not run at scale: the LLM decider mode and the conflict-seeking attacker against no defense or grounding alone.

Paired comparisons are cited with helpers from `preamble.tex`, all of which expand to the generated `\val{Cmp<Key>}` / `Lo` / `Hi` / `P` macros: `\cmp{CmpKey}` → "11.0 pp [9.1, 12.9]", `\ci{CmpKey}` → "[9.1, 12.9]", `\cmpneg{CmpKey}` → the same with the sign flipped, `\pholm{CmpKey}` → "p_Holm = 0.006" or "p_Holm < 0.001". A difference is called detectable only if its CI excludes zero **and** p_Holm < 0.05. A missing macro prints a red dash and a `rhal: missing generated value` warning instead of failing silently.
