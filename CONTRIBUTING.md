# Contributing to agentlens

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,langgraph]"
make test        # pytest, fully offline (mock model, no API keys)
make lint        # ruff check + ruff format --check
make typecheck   # mypy (strict on untyped defs)
```

## Ground rules

- **Tests come with the change.** Bug fixes need a test that fails without the fix.
- **The core stays mutation-tested.** `metrics.py`, `trajectory.py`, `cost.py` and `gate.py` are graded
  with [testteeth](https://pypi.org/project/testteeth/) (`make mutation`). Keep the score at or above
  the level recorded in the README; CI runs a scoped, non-blocking check on the functions you changed.
- **The tracer API is a contract.** [docs/tracer-api.md](docs/tracer-api.md) lists the stable surface.
  Changing it needs a deprecation path and a CHANGELOG entry. Other projects (for example sqlpilot)
  instrument themselves with it.
- **Core dependencies stay minimal.** The core needs only PyYAML; anything heavier goes in an extra.
- **Numbers in docs are measured.** Benchmark figures in the README must come from files in `results/`
  produced by `scripts/run_benchmark.sh` (or a documented command). Label anything untested.

## Adding a metric

1. Implement it as a pure function in `metrics.py` or `trajectory.py` with tests in the matching test file.
2. Add it to `CaseResult` / `summarize()` in `harness.py`, to `KNOWN_METRICS` in `gate.py`
   (and `HIGHER_IS_BETTER` / `LOWER_IS_BETTER` in `replay.py`), and to `SUMMARY_ROWS` in `report.py`.
3. Run `make mutation` and close the surviving mutants that matter.

## Releasing

Bump `src/agentlens/_version.py`, update `CHANGELOG.md`, then push a `vX.Y.Z` tag. The release workflow
builds the sdist and wheel, attaches them to a GitHub Release and publishes to PyPI when the
`PYPI_API_TOKEN` secret is configured.
