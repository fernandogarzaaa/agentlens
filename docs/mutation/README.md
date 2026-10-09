# Mutation testing reports

Generated with testteeth 0.2.0 on the pure core (`metrics.py`, `trajectory.py`, `cost.py`, `gate.py`)
with their unit tests, as configured in `[tool.testteeth]` in `pyproject.toml`:

```bash
testteeth run --workers 2 --timeout-factor 5 --json <file>
```

Each run used a clean `git clone` of the commit (plus `src/agentlens/data/pricing.toml`, which
early commits did not track), Python 3.12, on a shared 2-vCPU machine. The JSON reports record the
absolute path of that temporary clone as `root`.

| File | Commit | Mutants | Killed | Survived | Not covered | Timeouts | Score |
|---|---|---|---|---|---|---|---|
| `baseline.txt` / `baseline.json` | `7ce7807` (first commit) | 452 | 428 | 24 | 0 | 0 | 94.7% |
| `after.txt` / `after.json` | `e729130` (after test strengthening) | 442 | 440 | 0 | 2 | 0 | 99.5% |

The baseline was regenerated after its original output files were lost; it reproduces the numbers
of the original run log exactly. The 24 baseline survivors (listed in `baseline.txt`) were in
`evaluate_gate` (threshold boundaries, `>=` vs `>`), `values_equal` / `_arg_ok` (`return False` vs
`return None`), `match_calls` / `argument_match` (a `True` constant), `Pricing.lookup` (glob
specificity), `_metric`, a module constant in `gate.py`, and the Python-version checks guarding the
`tomli` fallback. They were closed with targeted tests or removed
by refactoring, which is also why the mutant count changed. The two "not covered" mutants in the
after run are the `import tomli as tomllib` fallback for Python < 3.11, which never executes on 3.12.
