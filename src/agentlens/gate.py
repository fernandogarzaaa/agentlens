"""CI regression gate: compare an eval summary to absolute thresholds and to a baseline.

.. code-block:: toml

    # gate.toml
    [thresholds]                       # absolute floors / ceilings on summary metrics
    task_success = { min = 0.80 }
    tool_f1 = { min = 0.90 }
    cost_per_task_usd = { max = 0.002 }
    latency_p95_s = { max = 10.0 }

    [baseline]                         # optional: relative to a previous results.json
    path = "results/baseline.json"     # relative to this file; --baseline on the CLI overrides it
    max_drop = { task_success = 0.05, tool_f1 = 0.05 }    # allowed absolute decrease
    max_increase = { cost_per_task_usd = 0.0005 }         # allowed absolute increase

``agentlens gate`` exits 0 when every check passes, 1 when any fails and 2 on a configuration error.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

__all__ = ["GateConfigError", "Check", "GateConfig", "GateResult", "load_gate_config", "evaluate_gate"]

KNOWN_METRICS = frozenset(
    {
        "task_success",
        "tool_precision",
        "tool_recall",
        "tool_f1",
        "tool_f1_micro",
        "args_exact",
        "args_partial",
        "trajectory_match",
        "constraint_pass",
        "answer_pass",
        "error_rate",
        "avg_steps",
        "avg_tokens",
        "total_tokens",
        "cost_per_task_usd",
        "total_cost_usd",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "judge_score_mean",
        "n_cases",
    }
)


class GateConfigError(ValueError):
    """Invalid gate configuration (exit code 2)."""


@dataclass
class Check:
    metric: str
    rule: str  # min | max | max_drop | max_increase
    limit: float
    actual: float | None
    passed: bool
    baseline: float | None = None

    @property
    def message(self) -> str:
        if self.actual is None:
            return f"{self.metric}: missing from results"
        if self.rule == "min":
            return f"{self.metric} = {self.actual:.4g} (min {self.limit:.4g})"
        if self.rule == "max":
            return f"{self.metric} = {self.actual:.4g} (max {self.limit:.4g})"
        if self.baseline is None:
            return f"{self.metric}: missing from baseline"
        delta = self.actual - self.baseline
        return f"{self.metric} = {self.actual:.4g} vs baseline {self.baseline:.4g} (delta {delta:+.4g}, {self.rule} {self.limit:.4g})"


@dataclass
class GateConfig:
    minimums: dict[str, float] = field(default_factory=dict)
    maximums: dict[str, float] = field(default_factory=dict)
    max_drop: dict[str, float] = field(default_factory=dict)
    max_increase: dict[str, float] = field(default_factory=dict)
    baseline_path: Path | None = None

    @property
    def needs_baseline(self) -> bool:
        return bool(self.max_drop or self.max_increase)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _metric(name: str, where: str) -> str:
    if name not in KNOWN_METRICS:
        raise GateConfigError(f"{where}: unknown metric {name!r}; known: {sorted(KNOWN_METRICS)}")
    return name


def parse_gate_config(data: dict[str, Any], base_dir: Path | None = None) -> GateConfig:
    unknown = set(data) - {"thresholds", "baseline"}
    if unknown:
        raise GateConfigError(f"unknown sections {sorted(unknown)}; expected [thresholds] and/or [baseline]")
    cfg = GateConfig()
    thresholds = data.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise GateConfigError("[thresholds] must be a table")
    for name, rule in thresholds.items():
        _metric(name, "thresholds")
        if not isinstance(rule, dict) or not rule or set(rule) - {"min", "max"}:
            raise GateConfigError(f"thresholds.{name} must be a table with 'min' and/or 'max'")
        if "min" in rule:
            cfg.minimums[name] = _number(rule["min"], f"thresholds.{name}.min")
        if "max" in rule:
            cfg.maximums[name] = _number(rule["max"], f"thresholds.{name}.max")
        if name in cfg.minimums and name in cfg.maximums and cfg.minimums[name] > cfg.maximums[name]:
            raise GateConfigError(f"thresholds.{name}: min > max")
    baseline = data.get("baseline", {})
    if not isinstance(baseline, dict):
        raise GateConfigError("[baseline] must be a table")
    unknown = set(baseline) - {"path", "max_drop", "max_increase"}
    if unknown:
        raise GateConfigError(f"[baseline]: unknown keys {sorted(unknown)}")
    for key, target in (("max_drop", cfg.max_drop), ("max_increase", cfg.max_increase)):
        table = baseline.get(key, {})
        if not isinstance(table, dict):
            raise GateConfigError(f"baseline.{key} must be a table")
        for name, limit in table.items():
            _metric(name, f"baseline.{key}")
            value = _number(limit, f"baseline.{key}.{name}")
            if value < 0:
                raise GateConfigError(f"baseline.{key}.{name} must be >= 0")
            target[name] = value
    if "path" in baseline:
        path = Path(str(baseline["path"]))
        cfg.baseline_path = path if path.is_absolute() or base_dir is None else base_dir / path
    if not (cfg.minimums or cfg.maximums or cfg.needs_baseline):
        raise GateConfigError("gate config defines no checks")
    return cfg


def load_gate_config(path: str | Path) -> GateConfig:
    p = Path(path)
    if not p.exists():
        raise GateConfigError(f"gate config not found: {p}")
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise GateConfigError(f"{p}: {exc}") from None
    return parse_gate_config(data, base_dir=p.parent)


@dataclass
class GateResult:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {
                    "metric": c.metric,
                    "rule": c.rule,
                    "limit": c.limit,
                    "actual": c.actual,
                    "baseline": c.baseline,
                    "passed": c.passed,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


def _value(summary: dict[str, Any], metric: str) -> float | None:
    v = summary.get(metric)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


# Floating-point slack so that 0.8 computed as 4/5 still passes "min = 0.8".
_EPS = 1e-9


def evaluate_gate(summary: dict[str, Any], config: GateConfig, baseline: dict[str, Any] | None = None) -> GateResult:
    """Run every check. A metric missing from the summary fails its check."""
    if config.needs_baseline and baseline is None:
        raise GateConfigError("gate config has [baseline] limits but no baseline results were given")
    checks: list[Check] = []
    for metric, limit in config.minimums.items():
        actual = _value(summary, metric)
        checks.append(Check(metric, "min", limit, actual, actual is not None and actual >= limit - _EPS))
    for metric, limit in config.maximums.items():
        actual = _value(summary, metric)
        checks.append(Check(metric, "max", limit, actual, actual is not None and actual <= limit + _EPS))
    if baseline is not None:
        for metric, limit in config.max_drop.items():
            actual, base = _value(summary, metric), _value(baseline, metric)
            ok = actual is not None and base is not None and base - actual <= limit + _EPS
            checks.append(Check(metric, "max_drop", limit, actual, ok, base))
        for metric, limit in config.max_increase.items():
            actual, base = _value(summary, metric), _value(baseline, metric)
            ok = actual is not None and base is not None and actual - base <= limit + _EPS
            checks.append(Check(metric, "max_increase", limit, actual, ok, base))
    return GateResult(checks)
