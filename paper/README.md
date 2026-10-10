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

Numbers in `main.tex` regenerate automatically, but some *prose* claims were written against the old 5-seed results and fail the paired tests in `tables/stats_main.tex` / `tables/stats_ablation.tex`. Each is marked with a `% TODO(30-seed)` comment in `main.tex` (abstract, Results, Ablation, Conclusion). In short: the placebo/net-ASR finding and the grounding gain hold strongly; the microstructure, risk-layer-drawdown, style-rewrite and κ-frontier claims do not. Cite `\val{Cmp<Key>}` / `Lo` / `Hi` / `P` macros when rewriting.
