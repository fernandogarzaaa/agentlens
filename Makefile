PY ?= python

.PHONY: install test lint format typecheck mutation mutation-ci quickstart gate bench build docker serve clean

install:            ## editable install with dev tools
	$(PY) -m pip install -e ".[dev]"

test:               ## full test suite (offline: mock model, no API keys)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy

mutation:           ## testteeth grade of the core (metrics, trajectory, cost, gate); see [tool.testteeth]
	$(PY) -m testteeth run --workers 2

mutation-ci:        ## what CI runs: only core functions changed vs origin/main
	$(PY) -m testteeth run --base origin/main --fail-under 80 --workers 2

quickstart:         ## the README quickstart with the mock model
	agentlens eval examples/travel_desk/suite.yaml --agent mock --mock-rules examples/travel_desk/mock_v1.yaml \
	  --model mock-v1 --trace out/v1.jsonl --out out/v1.json --report out/v1.md --quiet
	agentlens replay out/v1.jsonl --agent mock --mock-rules examples/travel_desk/mock_v2.yaml --model mock-v2 --report out/diff.md
	agentlens gate out/v1.json --config examples/travel_desk/gate.toml

gate:               ## CI regression gate on the travel-desk suite (mock model)
	agentlens eval examples/travel_desk/suite.yaml --agent mock --mock-rules examples/travel_desk/mock_v1.yaml \
	  --model mock-v1 --out out/ci.json --quiet
	agentlens gate out/ci.json --config examples/travel_desk/gate.toml

bench:              ## real-model benchmark (needs an OpenAI-compatible server, see scripts/run_benchmark.sh)
	sh scripts/run_benchmark.sh

build:
	$(PY) -m build

docker:
	docker build -t agentlens .

serve:
	agentlens serve out/v1.jsonl

clean:
	rm -rf build dist out src/*.egg-info .ruff_cache .mypy_cache .pytest_cache
