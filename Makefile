# Works on Linux/macOS and on Windows with GNU make (Git Bash).
PYTHON ?= python
VENV   ?= .venv
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PY := $(BIN)/python

.PHONY: setup seed test eval check-gold cli api docker-build docker-run

setup:            ## create the venv, install deps, seed the DB
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(MAKE) seed

seed:             ## (re)build data/shop.db deterministically
	$(PY) -m data.seed

test:             ## offline test suite: no API key, no cost
	$(PY) -m pytest -q

eval:             ## execution-accuracy eval; override with MODEL=openai:gpt-4.1-mini
	$(PY) -m evals.run_evals $(if $(MODEL),--model $(MODEL),)

check-gold:       ## verify every gold query runs
	$(PY) -m evals.run_evals --check-gold

cli:              ## interactive chat with live step streaming
	$(PY) -m app.cli

api:              ## serve the API on :8000
	$(PY) -m uvicorn app.api:app --reload --port 8000

docker-build:
	docker build -t sql-analyst-agent .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env -v sql-agent-state:/state sql-analyst-agent
