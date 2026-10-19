# RobustHFT-AgentLab -- all targets use the local venv; nothing is installed globally.
PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: venv install install-optional data test test-fast format lint pipeline ablation price-impact report frontend train-detector \
        paper paper-aaai paper-aaai-zip clean all llm-setup llm-up llm-down llm-pull real-data

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-optional:
	$(PIP) install -r requirements-optional.txt

data:
	$(PY) scripts/generate_sample_data.py

train-detector:
	$(PY) scripts/train_detector.py

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -q -x

# Code style: ruff (installed in .venv by `make install`); configuration in pyproject.toml
format:
	.venv/bin/ruff format src scripts tests
	.venv/bin/ruff check --fix src scripts tests

lint:
	.venv/bin/ruff format --check src scripts tests
	.venv/bin/ruff check src scripts tests

pipeline:
	$(PY) scripts/run_pipeline.py --config configs/default.yaml

ablation:
	$(PY) scripts/run_ablation.py --config configs/default.yaml

# Stress test: the frozen attack also moves the shared price path (gamma = 0, 1, 2)
price-impact:
	$(PY) scripts/run_price_impact.py --config configs/default.yaml

# Real BTC-USD daily prices (yfinance, once) -> data/real/btc_usd.csv; then e.g.
#   $(PY) scripts/run_pipeline.py --seeds 67 --out outputs/real_btc --set data.source=real --set grounding.analog_scale=auto
real-data:
	$(PY) scripts/fetch_real_prices.py

# report.md + paper tables/macros, then the paper's vector figures
report:
	$(PY) scripts/make_report_tables.py
	$(PY) scripts/make_paper_figures.py

# Animated showcase: writes frontend/data.js, then open frontend/index.html in a browser (works offline)
frontend:
	$(PY) scripts/export_frontend_data.py

# Uses tectonic if installed (brew install tectonic), else a TeX Live pdflatex+bibtex toolchain.
paper:
	cd paper && if command -v tectonic >/dev/null 2>&1; then tectonic main.tex; \
	else pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; fi

# AAAI-27 (FinFM workshop) build: anonymous; shares paper/preamble.tex and paper/sections/ with the IEEE build.
# aaai2027.sty only accepts pdfLaTeX. Without it, `paper-aaai` writes a tectonic draft (paper/aaai/draft.pdf)
# that skips just that engine check. For the submitted PDF, run `paper-aaai-zip`, upload the zip to Overleaf
# and compile main.tex with pdfLaTeX.
paper-aaai:
	cd paper/aaai && if command -v pdflatex >/dev/null 2>&1; then \
	pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; \
	else tectonic draft.tex; fi

paper-aaai-zip:
	rm -rf paper/aaai/build && mkdir -p paper/aaai/build/src
	cp paper/aaai/main.tex paper/aaai/aaai2027.sty paper/aaai/aaai2027.bst paper/preamble.tex paper/references.bib paper/aaai/build/src/
	cp -R paper/sections paper/tables paper/figures paper/aaai/build/src/
	rm -f paper/aaai/aaai27_source.zip
	cd paper/aaai/build/src && zip -qr ../../aaai27_source.zip .
	$(PY) scripts/export_openreview.py
	@echo "wrote paper/aaai/aaai27_source.zip: upload it to Overleaf and compile main.tex with pdfLaTeX"

# --- Local LLM (Ollama), installed INSIDE .venv: binary, models and state all live under .venv/ollama ---
# make llm-setup  -> pinned Ollama build, checksum-verified (scripts/setup_ollama.sh)
# make llm-up     -> server on 127.0.0.1:11434 in the background (pid + log in .venv/ollama/)
# make llm-pull   -> pull LLM_MODELS into .venv/ollama/models
# make llm-down   -> stop the server
OLLAMA := .venv/bin/ollama
OLLAMA_DIR := .venv/ollama
LLM_MODELS ?= qwen2.5:7b llama3.2:3b

llm-setup:
	bash scripts/setup_ollama.sh

llm-up:
	@test -x $(OLLAMA) || { echo "run 'make llm-setup' first"; exit 1; }
	@if [ -f $(OLLAMA_DIR)/server.pid ] && kill -0 $$(cat $(OLLAMA_DIR)/server.pid) 2>/dev/null; then \
		echo "ollama already running (pid $$(cat $(OLLAMA_DIR)/server.pid))"; \
	else \
		nohup $(OLLAMA) serve > $(OLLAMA_DIR)/server.log 2>&1 & echo $$! > $(OLLAMA_DIR)/server.pid; \
		curl -fs --retry 30 --retry-connrefused --retry-delay 1 http://127.0.0.1:11434/api/tags > /dev/null \
			&& echo "ollama up on 127.0.0.1:11434 (pid $$(cat $(OLLAMA_DIR)/server.pid))" \
			|| { echo "ollama did not start; see $(OLLAMA_DIR)/server.log"; exit 1; }; \
	fi

llm-down:
	@if [ -f $(OLLAMA_DIR)/server.pid ]; then kill $$(cat $(OLLAMA_DIR)/server.pid) 2>/dev/null || true; \
		rm -f $(OLLAMA_DIR)/server.pid; echo "ollama stopped"; else echo "ollama is not running"; fi

llm-pull:
	@for m in $(LLM_MODELS); do $(OLLAMA) pull $$m || exit 1; done

# Full happy path from a fresh clone
all: install data train-detector test pipeline ablation report frontend

clean:
	rm -rf outputs/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	touch outputs/.gitkeep
