# RobustHFT-AgentLab -- all targets use the local venv; nothing is installed globally.
PY := .venv/bin/python
PIP := $(PY) -m pip

.PHONY: venv install install-optional data test test-fast pipeline ablation report train-detector paper clean all

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

pipeline:
	$(PY) scripts/run_pipeline.py --config configs/default.yaml

ablation:
	$(PY) scripts/run_ablation.py --config configs/default.yaml

report:
	$(PY) scripts/make_report_tables.py

# Uses tectonic if installed (brew install tectonic), else a TeX Live pdflatex+bibtex toolchain.
paper:
	cd paper && if command -v tectonic >/dev/null 2>&1; then tectonic main.tex; \
	else pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex; fi

# Full happy path from a fresh clone
all: install data train-detector test pipeline ablation report

clean:
	rm -rf outputs/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	touch outputs/.gitkeep
