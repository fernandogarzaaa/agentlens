from pathlib import Path

import pytest

from agentlens.gate import (
    Check,
    GateConfig,
    GateConfigError,
    evaluate_gate,
    load_gate_config,
    parse_gate_config,
)


def test_parse_thresholds_and_baseline(tmp_path):
    cfg = parse_gate_config(
        {
            "thresholds": {
                "task_success": {"min": 0.8},
                "latency_p95_s": {"max": 5},
                "tool_f1": {"min": 0.5, "max": 1},
            },
            "baseline": {
                "path": "base.json",
                "max_drop": {"task_success": 0.05},
                "max_increase": {"cost_per_task_usd": 0},
            },
        },
        base_dir=tmp_path,
    )
    assert cfg.minimums == {"task_success": 0.8, "tool_f1": 0.5}
    assert cfg.maximums == {"latency_p95_s": 5.0, "tool_f1": 1.0}
    assert cfg.max_drop == {"task_success": 0.05}
    assert cfg.max_increase == {"cost_per_task_usd": 0.0}
    assert cfg.baseline_path == tmp_path / "base.json"
    assert cfg.needs_baseline


def test_baseline_path_absolute_or_without_base_dir(tmp_path):
    cfg = parse_gate_config(
        {"baseline": {"path": str(tmp_path / "b.json"), "max_drop": {"tool_f1": 0}}}, base_dir=Path("/elsewhere")
    )
    assert cfg.baseline_path == tmp_path / "b.json"
    cfg2 = parse_gate_config({"baseline": {"path": "b.json", "max_drop": {"tool_f1": 0}}})
    assert cfg2.baseline_path == Path("b.json")


def test_needs_baseline_flags():
    assert not GateConfig(minimums={"tool_f1": 1}).needs_baseline
    assert GateConfig(max_drop={"tool_f1": 0}).needs_baseline
    assert GateConfig(max_increase={"avg_steps": 0}).needs_baseline


@pytest.mark.parametrize(
    "data,match",
    [
        ({}, "no checks"),
        ({"thresholds": {}}, "no checks"),
        ({"limits": {}}, "unknown sections"),
        ({"thresholds": []}, "must be a table"),
        ({"thresholds": {"accuracy": {"min": 1}}}, "unknown metric"),
        ({"thresholds": {"tool_f1": 0.9}}, "min' and/or 'max"),
        ({"thresholds": {"tool_f1": {}}}, "min' and/or 'max"),
        ({"thresholds": {"tool_f1": {"min": 0.5, "avg": 1}}}, "min' and/or 'max"),
        ({"thresholds": {"tool_f1": {"min": "high"}}}, "must be a number"),
        ({"thresholds": {"tool_f1": {"min": True}}}, "must be a number"),
        ({"thresholds": {"tool_f1": {"min": 0.9, "max": 0.5}}}, "min > max"),
        ({"baseline": []}, r"\[baseline\] must be a table"),
        ({"baseline": {"path": "x", "drop": {}}}, "unknown keys"),
        ({"baseline": {"max_drop": []}}, "must be a table"),
        ({"baseline": {"max_increase": 3}}, "must be a table"),
        ({"baseline": {"max_drop": {"nope": 0.1}}}, "unknown metric"),
        ({"baseline": {"max_drop": {"tool_f1": -0.1}}}, ">= 0"),
        ({"baseline": {"max_increase": {"avg_steps": -1}}}, ">= 0"),
        ({"baseline": {"max_drop": {"tool_f1": "x"}}}, "must be a number"),
        ({"baseline": {"path": "only-a-path.json"}}, "no checks"),
    ],
)
def test_parse_errors(data, match):
    with pytest.raises(GateConfigError, match=match):
        parse_gate_config(data)


def test_min_equal_to_max_is_allowed():
    cfg = parse_gate_config({"thresholds": {"tool_f1": {"min": 0.5, "max": 0.5}}})
    assert cfg.minimums["tool_f1"] == cfg.maximums["tool_f1"] == 0.5


def test_zero_drop_is_allowed():
    assert parse_gate_config({"baseline": {"max_drop": {"tool_f1": 0}}}).max_drop == {"tool_f1": 0.0}


def test_load_gate_config(tmp_path):
    f = tmp_path / "gate.toml"
    f.write_text(
        '[thresholds]\ntask_success = { min = 0.9 }\n[baseline]\npath = "b.json"\nmax_drop = { task_success = 0.01 }\n'
    )
    cfg = load_gate_config(f)
    assert cfg.minimums == {"task_success": 0.9}
    assert cfg.baseline_path == tmp_path / "b.json"
    with pytest.raises(GateConfigError, match="not found"):
        load_gate_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[thresholds\n")
    with pytest.raises(GateConfigError, match="bad.toml"):
        load_gate_config(bad)


def _cfg(**kw):
    return GateConfig(**kw)


def test_min_threshold_pass_fail_and_boundary():
    cfg = _cfg(minimums={"task_success": 0.8})
    assert evaluate_gate({"task_success": 0.81}, cfg).passed
    assert evaluate_gate({"task_success": 0.8}, cfg).passed
    assert evaluate_gate({"task_success": 4 / 5}, cfg).passed
    r = evaluate_gate({"task_success": 0.79}, cfg)
    assert not r.passed and r.exit_code == 1
    assert r.failures[0].metric == "task_success"


def test_float_slack_is_tiny():
    cfg = _cfg(minimums={"task_success": 0.8})
    assert evaluate_gate({"task_success": 0.8 - 1e-12}, cfg).passed
    assert not evaluate_gate({"task_success": 0.8 - 1e-6}, cfg).passed
    cfg_max = _cfg(maximums={"latency_p95_s": 2.0})
    assert evaluate_gate({"latency_p95_s": 2.0 + 1e-12}, cfg_max).passed
    assert not evaluate_gate({"latency_p95_s": 2.0 + 1e-6}, cfg_max).passed


def test_max_threshold():
    cfg = _cfg(maximums={"latency_p95_s": 2.0})
    assert evaluate_gate({"latency_p95_s": 1.5}, cfg).passed
    assert evaluate_gate({"latency_p95_s": 2.0}, cfg).passed
    assert not evaluate_gate({"latency_p95_s": 2.5}, cfg).passed


def test_missing_or_non_numeric_metric_fails():
    cfg = _cfg(minimums={"tool_f1": 0.5}, maximums={"error_rate": 0.1})
    r = evaluate_gate({}, cfg)
    assert not r.passed and len(r.failures) == 2
    assert r.checks[0].actual is None and r.checks[0].message == "tool_f1: missing from results"
    r2 = evaluate_gate({"tool_f1": True, "error_rate": "0"}, cfg)
    assert [c.passed for c in r2.checks] == [False, False]


def test_int_metric_values_are_accepted():
    cfg = _cfg(minimums={"n_cases": 10})
    assert evaluate_gate({"n_cases": 10}, cfg).passed
    assert not evaluate_gate({"n_cases": 9}, cfg).passed


def test_max_drop_against_baseline():
    cfg = _cfg(max_drop={"task_success": 0.05})
    assert evaluate_gate({"task_success": 0.86}, cfg, {"task_success": 0.9}).passed
    assert evaluate_gate({"task_success": 0.85}, cfg, {"task_success": 0.9}).passed
    assert evaluate_gate({"task_success": 0.95}, cfg, {"task_success": 0.9}).passed
    r = evaluate_gate({"task_success": 0.80}, cfg, {"task_success": 0.9})
    assert not r.passed
    c = r.checks[0]
    assert (c.rule, c.limit, c.actual, c.baseline) == ("max_drop", 0.05, 0.8, 0.9)


def test_max_increase_against_baseline():
    cfg = _cfg(max_increase={"cost_per_task_usd": 0.001})
    assert evaluate_gate({"cost_per_task_usd": 0.0105}, cfg, {"cost_per_task_usd": 0.01}).passed
    assert evaluate_gate({"cost_per_task_usd": 0.011}, cfg, {"cost_per_task_usd": 0.01}).passed
    assert evaluate_gate({"cost_per_task_usd": 0.005}, cfg, {"cost_per_task_usd": 0.01}).passed
    assert not evaluate_gate({"cost_per_task_usd": 0.0111}, cfg, {"cost_per_task_usd": 0.01}).passed


def test_zero_tolerance_regression():
    cfg = _cfg(max_drop={"tool_f1": 0.0}, max_increase={"avg_steps": 0.0})
    assert evaluate_gate({"tool_f1": 0.9, "avg_steps": 2}, cfg, {"tool_f1": 0.9, "avg_steps": 2}).passed
    assert not evaluate_gate({"tool_f1": 0.89, "avg_steps": 2}, cfg, {"tool_f1": 0.9, "avg_steps": 2}).passed
    assert not evaluate_gate({"tool_f1": 0.9, "avg_steps": 2.1}, cfg, {"tool_f1": 0.9, "avg_steps": 2}).passed


def test_baseline_metric_missing_fails():
    cfg = _cfg(max_drop={"tool_f1": 0.1}, max_increase={"avg_steps": 1})
    r = evaluate_gate({"tool_f1": 0.9, "avg_steps": 1}, cfg, {})
    assert [c.passed for c in r.checks] == [False, False]
    assert r.checks[0].message == "tool_f1: missing from baseline"
    r2 = evaluate_gate({}, cfg, {"tool_f1": 0.9, "avg_steps": 1})
    assert [c.passed for c in r2.checks] == [False, False]


def test_baseline_required_when_configured():
    with pytest.raises(GateConfigError, match="no baseline"):
        evaluate_gate({"tool_f1": 1}, _cfg(max_drop={"tool_f1": 0.1}))
    # thresholds-only configs ignore a baseline
    r = evaluate_gate({"tool_f1": 1}, _cfg(minimums={"tool_f1": 0.5}), {"tool_f1": 0.1})
    assert r.passed and len(r.checks) == 1


def test_all_checks_reported_in_order():
    cfg = _cfg(
        minimums={"tool_f1": 0.5},
        maximums={"error_rate": 0.1},
        max_drop={"tool_f1": 0.1},
        max_increase={"avg_steps": 1},
    )
    r = evaluate_gate({"tool_f1": 0.9, "error_rate": 0.0, "avg_steps": 2}, cfg, {"tool_f1": 0.95, "avg_steps": 1.5})
    assert [c.rule for c in r.checks] == ["min", "max", "max_drop", "max_increase"]
    assert r.passed and r.exit_code == 0 and r.failures == []


def test_check_messages():
    assert Check("tool_f1", "min", 0.9, 0.85, False).message == "tool_f1 = 0.85 (min 0.9)"
    assert Check("latency_p95_s", "max", 2, 2.5, False).message == "latency_p95_s = 2.5 (max 2)"
    m = Check("task_success", "max_drop", 0.05, 0.8, False, baseline=0.9).message
    assert m == "task_success = 0.8 vs baseline 0.9 (delta -0.1, max_drop 0.05)"
    m2 = Check("avg_steps", "max_increase", 0.5, 3, False, baseline=2).message
    assert m2 == "avg_steps = 3 vs baseline 2 (delta +1, max_increase 0.5)"


def test_result_to_dict():
    r = evaluate_gate({"tool_f1": 0.4}, _cfg(minimums={"tool_f1": 0.5}))
    d = r.to_dict()
    assert d["passed"] is False
    assert d["checks"] == [
        {
            "metric": "tool_f1",
            "rule": "min",
            "limit": 0.5,
            "actual": 0.4,
            "baseline": None,
            "passed": False,
            "message": "tool_f1 = 0.4 (min 0.5)",
        }
    ]


def test_tolerance_boundaries_are_inclusive():
    from agentlens.gate import _EPS

    assert _EPS == 1e-9
    assert evaluate_gate({"task_success": 0.8 - 1e-9}, _cfg(minimums={"task_success": 0.8})).passed
    assert evaluate_gate({"latency_p95_s": 2.0 + 1e-9}, _cfg(maximums={"latency_p95_s": 2.0})).passed
    assert evaluate_gate({"tool_f1": 0.0}, _cfg(max_drop={"tool_f1": 0.0}), {"tool_f1": 1e-9}).passed
    assert evaluate_gate({"avg_steps": 1e-9}, _cfg(max_increase={"avg_steps": 0.0}), {"avg_steps": 0.0}).passed
    assert not evaluate_gate({"tool_f1": 0.0}, _cfg(max_drop={"tool_f1": 0.0}), {"tool_f1": 3e-9}).passed
    assert not evaluate_gate({"avg_steps": 3e-9}, _cfg(max_increase={"avg_steps": 0.0}), {"avg_steps": 0.0}).passed
