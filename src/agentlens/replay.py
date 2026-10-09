"""Replay recorded traces against a new agent version and diff the results.

A trace written by ``agentlens eval`` (or any run traced with :meth:`Tracer.run`) carries the run's
input, system prompt, tool schemas and every tool call with its result. Replay feeds the same
inputs to a new agent. Tools are answered from the recording (same tool + same arguments -> the
recorded result) so the comparison is deterministic and needs no live services; pass a suite to
use its live tools and expectations instead.

Without a suite, the *recorded* behaviour is the reference: the recorded tool calls become the
expected calls (strict trajectory) and the diff shows where the new version departs from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentlens.agent import Agent, Tool, Toolset
from agentlens.cases import Case, Suite
from agentlens.harness import (
    ATTR_MAX_STEPS,
    ATTR_SYSTEM_PROMPT,
    ATTR_TOOLS,
    CaseResult,
    EvalResult,
    run_eval,
    score_case,
    summarize,
)
from agentlens.metrics import score_arguments
from agentlens.store import Trace
from agentlens.tracer import ATTR_TOOL_ARGS, ATTR_TOOL_NAME, ATTR_TOOL_RESULT, SpanExporter, SpanKind

__all__ = [
    "RecordedToolset",
    "CaseDiff",
    "DiffReport",
    "suite_from_traces",
    "baseline_from_traces",
    "replay",
    "diff_results",
]

# Metrics where higher is better / lower is better, used for regression direction.
HIGHER_IS_BETTER = (
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
    "judge_score_mean",
)
LOWER_IS_BETTER = (
    "error_rate",
    "avg_steps",
    "avg_tokens",
    "cost_per_task_usd",
    "latency_mean_s",
    "latency_p50_s",
    "latency_p95_s",
)


class RecordedToolset(Toolset):
    """Answers tool calls from a recording: exact name + arguments match, else an error result."""

    def __init__(self, schemas: Sequence[dict[str, Any]], recordings: Sequence[tuple[str, dict[str, Any], Any]]):
        super().__init__([Tool.from_schema(s) for s in schemas])
        self.recordings = list(recordings)
        self.misses = 0

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        for rec_name, rec_args, result in self.recordings:
            if rec_name == name and score_arguments(rec_args, arguments, normalize_strings=False).exact:
                return result
        self.misses += 1
        return {"error": f"replay: no recorded result for {name}({arguments})"}


def _recordings(trace: Trace) -> list[tuple[str, dict[str, Any], Any]]:
    out = []
    for s in sorted((s for s in trace.spans if s.kind == SpanKind.TOOL), key=lambda s: s.start_ns):
        args = s.attributes.get(ATTR_TOOL_ARGS) or {}
        out.append(
            (
                str(s.attributes.get(ATTR_TOOL_NAME, s.name)),
                dict(args) if isinstance(args, dict) else {},
                s.attributes.get(ATTR_TOOL_RESULT),
            )
        )
    return out


def _case_id(trace: Trace, index: int) -> str:
    return trace.case_id or f"trace-{index}-{trace.trace_id[:8]}"


def suite_from_traces(traces: Sequence[Trace], name: str = "replay") -> tuple[Suite, dict[str, RecordedToolset]]:
    """Cases whose expected behaviour is the recorded behaviour, plus per-case recorded toolsets."""
    cases: list[Case] = []
    toolsets: dict[str, RecordedToolset] = {}
    system_prompts = set()
    for i, trace in enumerate(traces):
        root = trace.root
        if root is None or trace.input is None:
            continue
        case_id = _case_id(trace, i)
        inp = trace.input
        cases.append(
            Case(
                id=case_id,
                input=inp if isinstance(inp, (str, list)) else str(inp),
                expected_tool_calls=trace.tool_calls(),
                trajectory="strict",
                max_steps=int(root.attributes.get(ATTR_MAX_STEPS, 8)),
            )
        )
        system_prompts.add(str(root.attributes.get(ATTR_SYSTEM_PROMPT) or ""))
        toolsets[case_id] = RecordedToolset(root.attributes.get(ATTR_TOOLS) or [], _recordings(trace))
    if not cases:
        raise ValueError("no replayable runs found (need root spans with an agentlens.input attribute)")
    prompt = next(iter(system_prompts)) if len(system_prompts) == 1 else None
    return Suite(name=name, cases=cases, system_prompt=prompt or None), toolsets


def baseline_from_traces(traces: Sequence[Trace], suite: Suite) -> EvalResult:
    """Score the recorded runs against ``suite`` (cases matched by case id) without re-running anything."""
    results: list[CaseResult] = []
    for i, trace in enumerate(traces):
        case = suite.get(_case_id(trace, i))
        if case is None:
            continue
        out = trace.output
        results.append(
            score_case(case, trace.tool_calls(), None if out is None else str(out), trace.summary(), error=None)
        )
    return EvalResult(suite=suite.name, agent="recorded", model="", cases=results)


@dataclass
class CaseDiff:
    case_id: str
    status: str  # regression | improvement | changed | unchanged | added | removed
    reasons: list[str] = field(default_factory=list)
    base: dict[str, Any] | None = None
    new: dict[str, Any] | None = None


@dataclass
class DiffReport:
    base_label: str
    new_label: str
    base_summary: dict[str, Any]
    new_summary: dict[str, Any]
    cases: list[CaseDiff]

    @property
    def regressions(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "regression"]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "improvement"]

    def deltas(self) -> dict[str, float]:
        out = {}
        for key in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
            b, n = self.base_summary.get(key), self.new_summary.get(key)
            if isinstance(b, (int, float)) and isinstance(n, (int, float)):
                out[key] = n - b
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base_label,
            "new": self.new_label,
            "base_summary": self.base_summary,
            "new_summary": self.new_summary,
            "deltas": self.deltas(),
            "counts": {
                s: sum(1 for c in self.cases if c.status == s)
                for s in ("regression", "improvement", "changed", "unchanged", "added", "removed")
            },
            "cases": [c.__dict__ for c in self.cases],
        }


def _compare_case(base: CaseResult, new: CaseResult, tolerance: float) -> CaseDiff:
    worse: list[str] = []
    better: list[str] = []
    if base.success and not new.success:
        worse.append("task success: pass -> fail")
    elif new.success and not base.success:
        better.append("task success: fail -> pass")
    for key in ("tool_f1", "args_exact", "args_partial"):
        b, n = getattr(base, key), getattr(new, key)
        if n < b - tolerance:
            worse.append(f"{key} {b:.2f} -> {n:.2f}")
        elif n > b + tolerance:
            better.append(f"{key} {b:.2f} -> {n:.2f}")
    changed: list[str] = []
    base_names = [c["name"] for c in base.tool_calls]
    new_names = [c["name"] for c in new.tool_calls]
    if base_names != new_names:
        changed.append(f"trajectory {base_names} -> {new_names}")
    if (base.final_answer or "") != (new.final_answer or ""):
        changed.append("final answer changed")
    if worse:
        status = "regression"
    elif better:
        status = "improvement"
    elif changed:
        status = "changed"
    else:
        status = "unchanged"
    return CaseDiff(base.case_id, status, worse + better + changed, base.to_dict(), new.to_dict())


def _label(result: EvalResult) -> str:
    if not result.model or result.model in result.agent:
        return result.agent
    return f"{result.agent} ({result.model})"


def diff_results(base: EvalResult, new: EvalResult, tolerance: float = 1e-9) -> DiffReport:
    """Per-case and aggregate comparison of two eval results (cases matched by id)."""
    base_by = {c.case_id: c for c in base.cases}
    new_by = {c.case_id: c for c in new.cases}
    diffs: list[CaseDiff] = []
    for cid, b in base_by.items():
        n = new_by.get(cid)
        if n is None:
            diffs.append(CaseDiff(cid, "removed", ["case missing from new run"], b.to_dict(), None))
        else:
            diffs.append(_compare_case(b, n, tolerance))
    for cid, n in new_by.items():
        if cid not in base_by:
            diffs.append(CaseDiff(cid, "added", ["case not in base run"], None, n.to_dict()))
    common = [c for c in base_by if c in new_by]
    return DiffReport(
        base_label=_label(base),
        new_label=_label(new),
        base_summary=summarize([base_by[c] for c in common]),
        new_summary=summarize([new_by[c] for c in common]),
        cases=diffs,
    )


def replay(
    traces: Sequence[Trace],
    agent: Agent,
    agent_label: str = "candidate",
    model: str = "",
    suite: Suite | None = None,
    exporters: Sequence[SpanExporter] = (),
) -> tuple[EvalResult, EvalResult, DiffReport]:
    """Re-run recorded inputs with ``agent``. Returns ``(baseline, candidate, diff)``."""
    if suite is None:
        replay_suite, toolsets = suite_from_traces(traces)
        baseline = baseline_from_traces(traces, replay_suite)
        candidate = run_eval(
            replay_suite,
            agent,
            agent_label=agent_label,
            model=model,
            exporters=exporters,
            toolsets=dict(toolsets),
        )
    else:
        recorded_ids = {_case_id(t, i) for i, t in enumerate(traces)}
        replay_suite = suite.filter(ids=sorted(recorded_ids))
        baseline = baseline_from_traces(traces, replay_suite)
        candidate = run_eval(replay_suite, agent, agent_label=agent_label, model=model, exporters=exporters)
    baseline.agent = "recorded"
    return baseline, candidate, diff_results(baseline, candidate)
