"""Eval harness: run an agent over a suite, trace every case, score it, aggregate the metrics."""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentlens._version import __version__
from agentlens.agent import Agent, AgentResult, CannedToolset, Toolset, load_toolset
from agentlens.cases import Case, Suite
from agentlens.exporters import InMemoryExporter
from agentlens.metrics import (
    ToolCall,
    argument_match,
    check_answer,
    mean,
    percentile,
    tool_selection,
)
from agentlens.store import group_traces
from agentlens.tracer import RunSummary, SpanExporter, Tracer
from agentlens.trajectory import check_constraints, match_trajectory

if TYPE_CHECKING:
    from agentlens.judge import RubricJudge

__all__ = ["CaseResult", "EvalResult", "score_case", "summarize", "run_eval", "toolset_for_case"]

ATTR_TOOLS = "agentlens.tools"
ATTR_SYSTEM_PROMPT = "agentlens.system_prompt"
ATTR_MAX_STEPS = "agentlens.max_steps"
ATTR_SUITE = "agentlens.suite"


@dataclass
class CaseResult:
    case_id: str
    success: bool
    tool_precision: float
    tool_recall: float
    tool_f1: float
    args_exact: float
    args_partial: float
    trajectory_mode: str
    trajectory_match: bool
    constraints_ok: bool
    answer_ok: bool
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_known: bool
    latency_s: float
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    answer_failures: list[str] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    exhausted: bool = False
    judge_score: float | None = None
    judge_pass: bool | None = None
    trace_id: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseResult:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def score_case(
    case: Case,
    actual_calls: Sequence[ToolCall],
    final_answer: str | None,
    summary: RunSummary | None,
    error: str | None = None,
    exhausted: bool = False,
    judge_score: float | None = None,
    judge_pass: bool | None = None,
) -> CaseResult:
    """Score one finished run against its case. Pure function of its inputs."""
    sel = tool_selection(case.expected_tool_calls, actual_calls)
    args = argument_match(case.expected_tool_calls, actual_calls, allow_extra=case.allow_extra_args)
    actual_names = [c.name for c in actual_calls]
    traj_ok = match_trajectory(case.expected_names, actual_names, case.trajectory)
    steps = summary.steps if summary is not None else 0
    violations = check_constraints(case.constraints, actual_calls, steps)
    answer_ok, answer_failures = check_answer(final_answer, case.final_answer)
    if case.final_answer is not None and case.final_answer.judge and judge_pass is False:
        answer_ok = False
        answer_failures = [*answer_failures, f"judge score {judge_score} below pass threshold"]
    success = error is None and traj_ok and args.all_exact and not violations and answer_ok
    return CaseResult(
        case_id=case.id,
        success=success,
        tool_precision=sel.precision,
        tool_recall=sel.recall,
        tool_f1=sel.f1,
        args_exact=args.exact,
        args_partial=args.partial,
        trajectory_mode=case.trajectory,
        trajectory_match=traj_ok,
        constraints_ok=not violations,
        answer_ok=answer_ok,
        steps=steps,
        input_tokens=summary.input_tokens if summary else 0,
        output_tokens=summary.output_tokens if summary else 0,
        cost_usd=summary.cost_usd if summary else 0.0,
        cost_known=summary.cost_known if summary else False,
        latency_s=summary.latency_s if summary else 0.0,
        tool_calls=[c.to_dict() for c in actual_calls],
        expected_tool_calls=[c.to_dict() for c in case.expected_tool_calls],
        violations=violations,
        answer_failures=answer_failures,
        final_answer=final_answer,
        error=error,
        exhausted=exhausted,
        judge_score=judge_score,
        judge_pass=judge_pass,
        trace_id=summary.trace_id if summary else "",
        tags=list(case.tags),
    )


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Aggregate per-case results. Rates are means over cases (macro); ``tool_f1_micro`` pools calls."""
    n = len(results)
    if n == 0:
        return {"n_cases": 0}
    tp = n_exp = n_act = 0
    for r in results:
        sel = tool_selection(
            [ToolCall.from_dict(c) for c in r.expected_tool_calls], [ToolCall.from_dict(c) for c in r.tool_calls]
        )
        tp += sel.true_positives
        n_exp += sel.n_expected
        n_act += sel.n_actual
    micro_p = tp / n_act if n_act else 1.0
    micro_r = tp / n_exp if n_exp else 1.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    latencies = [r.latency_s for r in results]
    costs = [r.cost_usd for r in results]
    judged = [r.judge_score for r in results if r.judge_score is not None]
    out: dict[str, Any] = {
        "n_cases": n,
        "task_success": mean([float(r.success) for r in results]),
        "tool_precision": mean([r.tool_precision for r in results]),
        "tool_recall": mean([r.tool_recall for r in results]),
        "tool_f1": mean([r.tool_f1 for r in results]),
        "tool_f1_micro": micro_f1,
        "args_exact": mean([r.args_exact for r in results]),
        "args_partial": mean([r.args_partial for r in results]),
        "trajectory_match": mean([float(r.trajectory_match) for r in results]),
        "constraint_pass": mean([float(r.constraints_ok) for r in results]),
        "answer_pass": mean([float(r.answer_ok) for r in results]),
        "error_rate": mean([float(r.error is not None) for r in results]),
        "avg_steps": mean([float(r.steps) for r in results]),
        "avg_tokens": mean([float(r.total_tokens) for r in results]),
        "total_tokens": sum(r.total_tokens for r in results),
        "cost_per_task_usd": mean(costs),
        "total_cost_usd": round(sum(costs), 10),
        "cost_known": all(r.cost_known for r in results),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }
    if judged:
        out["judge_score_mean"] = mean(judged)
    return out


@dataclass
class EvalResult:
    suite: str
    agent: str
    model: str
    cases: list[CaseResult]
    created: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    agentlens_version: str = __version__
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        return summarize(self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "agent": self.agent,
            "model": self.model,
            "created": self.created,
            "agentlens_version": self.agentlens_version,
            "metadata": self.metadata,
            "summary": self.summary,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        if not isinstance(d, dict):
            raise ValueError("results must be a JSON object")
        cases = d.get("cases", [])
        if not isinstance(cases, list) or not all(isinstance(c, dict) for c in cases):
            raise ValueError("results 'cases' must be a list of objects")
        return cls(
            suite=d.get("suite", ""),
            agent=d.get("agent", ""),
            model=d.get("model", ""),
            cases=[CaseResult.from_dict(c) for c in d.get("cases", [])],
            created=d.get("created", ""),
            agentlens_version=d.get("agentlens_version", ""),
            metadata=dict(d.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def toolset_for_case(case: Case, suite: Suite, cache: dict[str, Toolset] | None = None) -> Toolset:
    if case.tools is not None:
        return CannedToolset(case.tools, case.tool_results if case.tool_results is not None else suite.tool_results)
    if suite.toolset:
        cache = cache if cache is not None else {}
        if suite.toolset not in cache:
            cache[suite.toolset] = load_toolset(suite.toolset, suite.base_dir)
        return cache[suite.toolset]
    if suite.tools is not None:
        return CannedToolset(suite.tools, case.tool_results if case.tool_results is not None else suite.tool_results)
    return Toolset()


def run_eval(
    suite: Suite,
    agent: Agent,
    agent_label: str = "agent",
    model: str = "",
    exporters: Sequence[SpanExporter] = (),
    tracer: Tracer | None = None,
    judge: RubricJudge | None = None,
    concurrency: int = 1,
    on_case: Callable[[CaseResult], None] | None = None,
    toolsets: dict[str, Toolset] | None = None,
) -> EvalResult:
    """Run every case, tracing into ``exporters`` (plus an in-memory store used for scoring).

    ``toolsets`` overrides the toolset for specific case ids (replay uses this for recorded tools).
    """
    memory = InMemoryExporter()
    if tracer is None:
        tracer = Tracer(service_name=f"agentlens-eval:{suite.name}")
    tracer.exporters = [memory, *tracer.exporters, *exporters]
    cache: dict[str, Toolset] = {}

    def one(case: Case) -> CaseResult:
        toolset = toolsets[case.id] if toolsets and case.id in toolsets else toolset_for_case(case, suite, cache)
        result: AgentResult
        with tracer.run(
            f"case {case.id}",
            input=case.input if isinstance(case.input, str) else case.input,
            case_id=case.id,
            **{
                ATTR_SUITE: suite.name,
                ATTR_TOOLS: toolset.schemas(),
                ATTR_SYSTEM_PROMPT: suite.system_prompt or "",
                ATTR_MAX_STEPS: case.max_steps,
            },
        ) as root:
            try:
                result = agent.run(
                    case.input, toolset, system_prompt=suite.system_prompt, max_steps=case.max_steps, tracer=tracer
                )
            except Exception as exc:  # noqa: BLE001 - a crashing agent is a failed case, not a crashed eval
                root.record_error(exc)
                result = AgentResult(None, error=f"{type(exc).__name__}: {exc}")
            root.set_output(result.final_answer)
        spans = memory.by_trace(root.trace_id)
        trace = group_traces(spans)[0]
        summary = tracer.summary(root.trace_id) or trace.summary()
        judge_score = judge_pass = None
        if judge is not None and case.final_answer is not None and case.final_answer.judge and result.final_answer:
            verdict = judge.score(case.input_text, result.final_answer)
            judge_score, judge_pass = verdict.score, verdict.passed
        scored = score_case(
            case,
            trace.tool_calls(),
            result.final_answer,
            summary,
            error=result.error,
            exhausted=result.exhausted,
            judge_score=judge_score,
            judge_pass=judge_pass,
        )
        if on_case is not None:
            on_case(scored)
        return scored

    if concurrency <= 1:
        results = [one(c) for c in suite.cases]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, suite.cases))
    return EvalResult(suite=suite.name, agent=agent_label, model=model, cases=results)
