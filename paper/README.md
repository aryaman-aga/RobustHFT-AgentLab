# paper/

| File | What it is |
|---|---|
| `main.tex` | Conference manuscript (IEEEtran). Every number in the prose comes from `\val{...}` macros defined in `tables/numbers.tex`, so it always matches the latest run. |
| `tables/*.tex` | **Generated** by `scripts/make_report_tables.py` from `outputs/`. Never hand-edit. |
| `figures/*.png` | **Generated**: copied from `outputs/pipeline/` by the same script, so `paper/` compiles from a fresh clone. |
| `references.bib` | Bibliography shared by both drafts. |
| `ieee_paper.tex` | Earlier standalone draft with **hand-typed** tables. Its numbers predate the current outputs and do not match them; it should be merged into `main.tex` (or regenerated) before submission. |

Build: `make paper` from the repo root (uses `tectonic` if installed, else `pdflatex` + `bibtex`). PDFs are build products and are git-ignored.

Refresh numbers after new experiments: `make pipeline ablation report`, then `make paper`.

## Status (30-seed re-run, Oct 2026)

The prose of `main.tex` has been rewritten against the paired tests in `tables/stats_main.tex` / `tables/stats_ablation.tex`. The placebo/net-ASR finding and the grounding gain hold in both attack settings. The microstructure, risk-layer (net ASR, active flips, drawdown), style-rewrite and κ effects show no detectable difference and are reported as such.

Paired comparisons are cited with three helpers defined in the preamble of `main.tex`, all of which expand to the generated `\val{Cmp<Key>}` / `Lo` / `Hi` / `P` macros: `\cmp{CmpKey}` → "8.8 pp [6.1, 11.6]", `\ci{CmpKey}` → "[6.1, 11.6]", `\pholm{CmpKey}` → "p_Holm = 0.041" or "p_Holm < 0.001". A difference is called detectable only if its CI excludes zero **and** p_Holm < 0.05.
