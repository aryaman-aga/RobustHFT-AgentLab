# paper/

Conference manuscript **"Separating Meaning from Noise: Placebo-Controlled Red-Teaming of LLM Trading Agents"** (IEEEtran conference format; 8 pages including references and a statistical appendix).

| File | What it is |
|---|---|
| `main.tex` | The manuscript. Every number in the prose comes from generated `\val{...}` macros, so it always matches the latest run. |
| `references.bib` | Bibliography. Every entry was checked against the publisher, the ACL Anthology, NeurIPS proceedings, arXiv or the DOI registry (Oct 2026). Acronyms are brace-protected because IEEEtran sentence-cases titles; an `IEEEtranBSTCTL` entry switches on "first author et al." for more than six authors. |
| `figures/fig_harness.tex`, `figures/fig_controls.tex` | TikZ diagrams (Fig. 1 harness, Fig. 2 control ladder), drawn at their natural size so their fonts match the captions. |
| `figures/fig_*.pdf` | **Generated** vector figures (Figs. 3--6) from `scripts/make_paper_figures.py`. |
| `tables/numbers.tex`, `tables/stats_*.tex`, `tables/detector.tex` | **Generated** by `scripts/make_report_tables.py` from `outputs/`. |
| `tables/tab_main.tex`, `tables/figmacros.tex` | **Generated** by `scripts/make_paper_figures.py` (Table I and numbers that only the figures compute). |
| `tables/main_results.tex`, `tables/ablation_*.tex`, `tables/asr_decomposition.tex` | **Generated** mean ± sd tables. `main.tex` does not use them; they are kept for the BTP reports. |

Never hand-edit a generated file: change the script and run `make report`.

## Build

`make paper` from the repo root (uses `tectonic` if installed, else `pdflatex` + `bibtex`). The preamble detects the engine: TeX Gyre Termes through fontspec under XeTeX/LuaTeX, newtx under pdfLaTeX, so the same source builds on a submission server. A clean build has no overfull boxes, no undefined citations and no `rhal: missing generated value` warnings; check `main.log` for all three. PDFs are build products and are git-ignored.

Refresh everything after new experiments: `make pipeline ablation report`, then `make paper`.

## Anonymous vs camera-ready

`\anonymoustrue` near the top of `main.tex` prints "Anonymous Authors" and hides the repository URL for double-blind review. For the camera-ready version set `\anonymousfalse`, then fill in the author block (names, affiliation, e-mail) and check the URL in the Reproducibility section.

## Status (100 seeds, leak-free re-run, Oct 2026)

All results come from 100 paired seeds, re-run after the text-lookahead and hash-order fixes. Every claim in the prose is a paired comparison listed in `scripts/make_report_tables.py` and reported in the appendix tables. The placebo ladder, the grounding gain (in both attack settings) and the hold-on-conflict penalty are detectable. Microstructure proxies, the risk layer's drawdown benefit and most κ steps are not, and the text says "no detectable difference" for them.

Paired comparisons are cited with helpers defined in the preamble of `main.tex`, all of which expand to the generated `\val{Cmp<Key>}` / `Lo` / `Hi` / `P` macros: `\cmp{CmpKey}` → "11.0 pp [9.1, 12.9]", `\ci{CmpKey}` → "[9.1, 12.9]", `\cmpneg{CmpKey}` → the same with the sign flipped, `\pholm{CmpKey}` → "p_Holm = 0.006" or "p_Holm < 0.001". A difference is called detectable only if its CI excludes zero **and** p_Holm < 0.05. A missing macro prints a red dash and a `rhal: missing generated value` warning instead of failing silently.
