"""Tracing: OpenTelemetry-compatible spans for agent runs, agent steps, LLM calls and tool calls.

Public API (stable since 0.1.0, see docs/tracer-api.md):

    tracer = Tracer(service_name="my-agent", exporters=[JSONLExporter("traces.jsonl")])

    with tracer.run("task-42", input="book me a flight") as run:
        with tracer.agent_step(0):
            with tracer.llm_call("gpt-4o-mini") as llm:
                ...
                llm.record_usage(input_tokens=812, output_tokens=64)
            with tracer.tool_call("search_flights", arguments={"to": "MNL"}) as tool:
                tool.set_result(result)
        run.set_output(answer)

    @trace_tool()            # or trace_llm(model=...), trace_agent_step(), trace_run(), traced(kind=...)
    def search_flights(to: str) -> list[dict]: ...

Span ids follow the W3C / OpenTelemetry format (32 hex chars for the trace id, 16 for the span id),
timestamps are Unix nanoseconds and attribute names follow the OpenTelemetry GenAI semantic
conventions (``gen_ai.*``) where one exists, so spans export to any OTLP collector unchanged.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import secrets
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast

from agentlens.cost import Pricing

__all__ = [
    "SpanKind",
    "Span",
    "SpanExporter",
    "RunSummary",
    "Tracer",
    "configure",
    "get_tracer",
    "set_tracer",
    "current_span",
    "traced",
    "trace_run",
    "trace_agent_step",
    "trace_llm",
    "trace_tool",
]

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind:
    """Values of the ``agentlens.span.kind`` attribute."""

    RUN = "run"
    AGENT_STEP = "agent_step"
    LLM = "llm"
    TOOL = "tool"
    OTHER = "other"

    ALL = (RUN, AGENT_STEP, LLM, TOOL, OTHER)


# Attribute keys. GenAI semantic-convention names where they exist, agentlens.* otherwise.
ATTR_KIND = "agentlens.span.kind"
ATTR_INPUT = "agentlens.input"
ATTR_OUTPUT = "agentlens.output"
ATTR_CASE_ID = "agentlens.case_id"
ATTR_STEP_INDEX = "agentlens.step.index"
ATTR_COST = "agentlens.cost_usd"
ATTR_COST_KNOWN = "agentlens.cost_known"
ATTR_OPERATION = "gen_ai.operation.name"
ATTR_SYSTEM = "gen_ai.system"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_CACHED_TOKENS = "gen_ai.usage.cached_input_tokens"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_TOOL_ARGS = "gen_ai.tool.call.arguments"
ATTR_TOOL_RESULT = "gen_ai.tool.call.result"

_MAX_ATTR_CHARS = 8000


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def _jsonable(value: Any) -> Any:
    """Coerce a value into something JSON-serialisable (str fallback), truncating long strings."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_ATTR_CHARS else value[:_MAX_ATTR_CHARS] + "...[truncated]"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return _jsonable(repr(value))


@dataclass
class Span:
    """One timed operation. Mutable while open, frozen in spirit once ``end_ns`` is set."""

    name: str
    kind: str = SpanKind.OTHER
    trace_id: str = field(default_factory=_new_trace_id)
    span_id: str = field(default_factory=_new_span_id)
    parent_id: str | None = None
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    status: str = "unset"  # unset | ok | error  (OpenTelemetry status codes)
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    _tracer: Tracer | None = field(default=None, repr=False, compare=False)

    # ---- mutation helpers -------------------------------------------------------------
    def set_attribute(self, key: str, value: Any) -> Span:
        self.attributes[key] = _jsonable(value)
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        for k, v in attributes.items():
            self.set_attribute(k, v)
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        self.events.append({"name": name, "time_ns": time.time_ns(), "attributes": _jsonable(attributes or {})})
        return self

    def set_input(self, value: Any) -> Span:
        return self.set_attribute(ATTR_INPUT, value)

    def set_output(self, value: Any) -> Span:
        return self.set_attribute(ATTR_OUTPUT, value)

    def set_result(self, value: Any) -> Span:
        """Record a tool's return value."""
        return self.set_attribute(ATTR_TOOL_RESULT, value)

    def record_error(self, exc: BaseException | str) -> Span:
        self.status = "error"
        if isinstance(exc, BaseException):
            self.status_message = f"{type(exc).__name__}: {exc}"
            self.add_event(
                "exception",
                {
                    "exception.type": type(exc).__name__,
                    "exception.message": str(exc),
                    "exception.stacktrace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                },
            )
        else:
            self.status_message = exc
        return self

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        model: str | None = None,
        cost_usd: float | None = None,
    ) -> Span:
        """Record token usage on an LLM span; cost is computed from the tracer's pricing table.

        Pass ``cost_usd`` to override (for example when the provider returns the billed amount).
        """
        if model:
            self.set_attribute(ATTR_RESPONSE_MODEL, model)
        self.attributes[ATTR_INPUT_TOKENS] = int(self.attributes.get(ATTR_INPUT_TOKENS, 0)) + int(input_tokens)
        self.attributes[ATTR_OUTPUT_TOKENS] = int(self.attributes.get(ATTR_OUTPUT_TOKENS, 0)) + int(output_tokens)
        if cached_input_tokens:
            self.attributes[ATTR_CACHED_TOKENS] = int(self.attributes.get(ATTR_CACHED_TOKENS, 0)) + int(
                cached_input_tokens
            )
        if cost_usd is None and self._tracer is not None:
            price_model = (
                self.attributes.get(ATTR_RESPONSE_MODEL) or self.attributes.get(ATTR_REQUEST_MODEL) or model or ""
            )
            cost = self._tracer.pricing.cost(
                str(price_model),
                input_tokens=int(self.attributes[ATTR_INPUT_TOKENS]),
                output_tokens=int(self.attributes[ATTR_OUTPUT_TOKENS]),
                cached_input_tokens=int(self.attributes.get(ATTR_CACHED_TOKENS, 0)),
            )
            self.attributes[ATTR_COST_KNOWN] = cost is not None
            self.attributes[ATTR_COST] = cost if cost is not None else 0.0
        elif cost_usd is not None:
            self.attributes[ATTR_COST_KNOWN] = True
            self.attributes[ATTR_COST] = float(cost_usd)
        return self

    # ---- read helpers -----------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.time_ns()
        return (end - self.start_ns) / 1e9

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def end(self, end_ns: int | None = None) -> None:
        """Close the span and hand it to the tracer's exporters. Idempotent."""
        if self.end_ns is not None:
            return
        self.end_ns = end_ns if end_ns is not None else time.time_ns()
        if self.status == "unset":
            self.status = "ok"
        if self._tracer is not None:
            self._tracer._on_end(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        return cls(
            name=d["name"],
            kind=d.get("kind", SpanKind.OTHER),
            trace_id=d["trace_id"],
            span_id=d["span_id"],
            parent_id=d.get("parent_id"),
            start_ns=int(d["start_ns"]),
            end_ns=None if d.get("end_ns") is None else int(d["end_ns"]),
            status=d.get("status", "unset"),
            status_message=d.get("status_message", ""),
            attributes=dict(d.get("attributes") or {}),
            events=list(d.get("events") or []),
        )


class SpanExporter(Protocol):
    """Anything with ``export(spans)`` and ``shutdown()``. See agentlens.exporters."""

    def export(self, spans: Sequence[Span]) -> None: ...

    def shutdown(self) -> None: ...


@dataclass
class RunSummary:
    """Token, cost and shape accounting for one run (one trace)."""

    trace_id: str
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True
    latency_s: float = 0.0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_spans(cls, spans: Sequence[Span]) -> RunSummary:
        if not spans:
            raise ValueError("cannot summarise an empty trace")
        summary = cls(trace_id=spans[0].trace_id)
        roots = [s for s in spans if s.parent_id is None]
        for s in spans:
            if s.kind == SpanKind.LLM:
                summary.llm_calls += 1
                summary.input_tokens += int(s.attributes.get(ATTR_INPUT_TOKENS, 0))
                summary.output_tokens += int(s.attributes.get(ATTR_OUTPUT_TOKENS, 0))
                summary.cost_usd += float(s.attributes.get(ATTR_COST, 0.0))
                if not s.attributes.get(ATTR_COST_KNOWN, False):
                    summary.cost_known = False
            elif s.kind == SpanKind.TOOL:
                summary.tool_calls += 1
            elif s.kind == SpanKind.AGENT_STEP:
                summary.steps += 1
            if s.status == "error":
                summary.errors += 1
        if summary.steps == 0:
            # Agents that do not mark steps explicitly: one step per LLM call.
            summary.steps = summary.llm_calls
        timed = roots or list(spans)
        start = min(s.start_ns for s in timed)
        end = max(s.end_ns if s.end_ns is not None else s.start_ns for s in timed)
        summary.latency_s = (end - start) / 1e9
        summary.cost_usd = round(summary.cost_usd, 10)
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cost_known": self.cost_known,
            "latency_s": self.latency_s,
            "errors": self.errors,
        }


_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("agentlens_current_span", default=None)


def current_span() -> Span | None:
    """The innermost open span in this context (thread / asyncio task), if any."""
    return _current_span.get()


class Tracer:
    """Creates spans, tracks parent/child relationships through contextvars, and exports finished spans.

    Thread-safe and asyncio-safe: each thread / task sees its own current span.
    """

    def __init__(
        self,
        service_name: str = "agent",
        exporters: Sequence[SpanExporter] | None = None,
        pricing: Pricing | None = None,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.service_name = service_name
        self.exporters: list[SpanExporter] = list(exporters or [])
        self.pricing = pricing if pricing is not None else Pricing.default()
        self.resource_attributes: dict[str, Any] = {"service.name": service_name, **(resource_attributes or {})}
        self._lock = threading.Lock()
        self._open_traces: dict[str, list[Span]] = {}
        self._summaries: dict[str, RunSummary] = {}

    # ---- low-level span management (used by adapters) -----------------------------------
    def start_span(
        self,
        name: str,
        kind: str = SpanKind.OTHER,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
        new_trace: bool = False,
    ) -> Span:
        """Start a span without making it current. Parent defaults to the current span."""
        if kind not in SpanKind.ALL:
            raise ValueError(f"unknown span kind {kind!r}; expected one of {SpanKind.ALL}")
        if parent is None and not new_trace:
            parent = current_span()
        span = Span(name=name, kind=kind, _tracer=self)
        if parent is not None and not new_trace:
            span.trace_id = parent.trace_id
            span.parent_id = parent.span_id
        span.attributes[ATTR_KIND] = kind
        if attributes:
            span.set_attributes(attributes)
        with self._lock:
            self._open_traces.setdefault(span.trace_id, []).append(span)
        return span

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = SpanKind.OTHER,
        attributes: dict[str, Any] | None = None,
        new_trace: bool = False,
    ) -> Iterator[Span]:
        """Context manager: start a span, make it current, end it (recording any exception)."""
        span = self.start_span(name, kind=kind, attributes=attributes, new_trace=new_trace)
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            span.record_error(exc)
            raise
        finally:
            _current_span.reset(token)
            span.end()

    # ---- typed helpers ----------------------------------------------------------------
    def run(self, name: str = "agent.run", input: Any = None, case_id: str | None = None, **attributes: Any) -> Any:
        """Root span for one agent run (always starts a new trace)."""
        attrs: dict[str, Any] = dict(attributes)
        if input is not None:
            attrs[ATTR_INPUT] = input
        if case_id is not None:
            attrs[ATTR_CASE_ID] = case_id
        return self.span(name, kind=SpanKind.RUN, attributes=attrs, new_trace=True)

    def agent_step(self, index: int | None = None, name: str = "agent.step", **attributes: Any) -> Any:
        attrs: dict[str, Any] = dict(attributes)
        if index is not None:
            attrs[ATTR_STEP_INDEX] = index
        return self.span(name, kind=SpanKind.AGENT_STEP, attributes=attrs)

    def llm_call(self, model: str, name: str | None = None, system: str | None = None, **attributes: Any) -> Any:
        attrs: dict[str, Any] = {ATTR_OPERATION: "chat", ATTR_REQUEST_MODEL: model, **attributes}
        if system:
            attrs[ATTR_SYSTEM] = system
        return self.span(name or f"chat {model}", kind=SpanKind.LLM, attributes=attrs)

    def tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
        **attributes: Any,
    ) -> Any:
        attrs: dict[str, Any] = {
            ATTR_OPERATION: "execute_tool",
            ATTR_TOOL_NAME: tool_name,
            ATTR_TOOL_ARGS: arguments if arguments is not None else {},
            **attributes,
        }
        if call_id:
            attrs[ATTR_TOOL_CALL_ID] = call_id
        return self.span(f"execute_tool {tool_name}", kind=SpanKind.TOOL, attributes=attrs)

    # ---- export ---------------------------------------------------------------------
    def _on_end(self, span: Span) -> None:
        finished: list[Span] | None = None
        with self._lock:
            spans = self._open_traces.get(span.trace_id)
            if span.parent_id is None and spans is not None:
                finished = self._open_traces.pop(span.trace_id)
        if finished is not None:
            summary = RunSummary.from_spans(finished)
            span.attributes.update(
                {
                    "agentlens.run.llm_calls": summary.llm_calls,
                    "agentlens.run.tool_calls": summary.tool_calls,
                    "agentlens.run.steps": summary.steps,
                    "agentlens.run.input_tokens": summary.input_tokens,
                    "agentlens.run.output_tokens": summary.output_tokens,
                    "agentlens.run.cost_usd": summary.cost_usd,
                    "agentlens.run.cost_known": summary.cost_known,
                }
            )
            with self._lock:
                self._summaries[span.trace_id] = summary
        for exporter in self.exporters:
            exporter.export([span])

    def summary(self, trace_id: str) -> RunSummary | None:
        """Accounting for a finished run (root span ended)."""
        with self._lock:
            return self._summaries.get(trace_id)

    def add_exporter(self, exporter: SpanExporter) -> None:
        self.exporters.append(exporter)

    def shutdown(self) -> None:
        for exporter in self.exporters:
            exporter.shutdown()


# ---- global tracer -------------------------------------------------------------------
_global_tracer: Tracer | None = None
_global_lock = threading.Lock()


def configure(
    service_name: str = "agent",
    exporters: Sequence[SpanExporter] | None = None,
    pricing: Pricing | None = None,
    resource_attributes: dict[str, Any] | None = None,
) -> Tracer:
    """Create the process-wide tracer used by the decorators and return it."""
    tracer = Tracer(service_name, exporters=exporters, pricing=pricing, resource_attributes=resource_attributes)
    set_tracer(tracer)
    return tracer


def set_tracer(tracer: Tracer) -> None:
    global _global_tracer
    with _global_lock:
        _global_tracer = tracer


def get_tracer() -> Tracer:
    """The process-wide tracer (a no-exporter tracer is created on first use)."""
    global _global_tracer
    with _global_lock:
        if _global_tracer is None:
            _global_tracer = Tracer()
        return _global_tracer


# ---- decorators ----------------------------------------------------------------------
def _bind_arguments(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": kwargs}
    return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}


def traced(
    kind: str = SpanKind.OTHER,
    name: str | None = None,
    tracer: Tracer | None = None,
    capture_args: bool = True,
    capture_result: bool = True,
    model: str | None = None,
) -> Callable[[F], F]:
    """Decorate a sync or async function so each call becomes a span of ``kind``.

    For ``kind="tool"`` the call arguments become ``gen_ai.tool.call.arguments`` and the return value
    ``gen_ai.tool.call.result``. For ``kind="llm"`` pass ``model``; if the function returns an
    object with an OpenAI-style ``usage`` (dict or attribute) the token counts are recorded.
    For ``kind="run"`` a new trace is started.
    """

    def decorator(fn: F) -> F:
        span_name = name or (fn.__name__ if kind == SpanKind.TOOL else fn.__qualname__)

        def _open(t: Tracer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            call_args = _bind_arguments(fn, args, kwargs) if capture_args else None
            if kind == SpanKind.TOOL:
                return t.tool_call(span_name, arguments=call_args)
            if kind == SpanKind.LLM:
                return t.llm_call(model or "unknown", name=span_name)
            if kind == SpanKind.RUN:
                return t.run(span_name, input=call_args)
            if kind == SpanKind.AGENT_STEP:
                return t.agent_step(name=span_name)
            attrs = {ATTR_INPUT: call_args} if call_args is not None else None
            return t.span(span_name, kind=kind, attributes=attrs)

        def _close(span: Span, result: Any) -> None:
            if kind == SpanKind.LLM:
                usage = _extract_usage(result)
                if usage is not None:
                    span.record_usage(**usage)
            if not capture_result:
                return
            if kind == SpanKind.TOOL:
                span.set_result(result)
            elif kind != SpanKind.LLM:
                span.set_output(result)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                t = tracer or get_tracer()
                with _open(t, args, kwargs) as span:
                    result = await fn(*args, **kwargs)
                    _close(span, result)
                    return result

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t = tracer or get_tracer()
            with _open(t, args, kwargs) as span:
                result = fn(*args, **kwargs)
                _close(span, result)
                return result

        return cast(F, wrapper)

    return decorator


def _extract_usage(result: Any) -> dict[str, Any] | None:
    usage: Any = result.get("usage") if isinstance(result, dict) else getattr(result, "usage", None)
    if usage is None:
        return None
    if not isinstance(usage, dict):
        usage = {
            k: getattr(usage, k, 0) for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
        }
    model = result.get("model") if isinstance(result, dict) else getattr(result, "model", None)
    return {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "model": model if isinstance(model, str) else None,
    }


def trace_run(name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    return traced(SpanKind.RUN, name=name, tracer=tracer)


def trace_agent_step(name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    return traced(SpanKind.AGENT_STEP, name=name, tracer=tracer, capture_args=False, capture_result=False)


def trace_llm(model: str, name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    return traced(SpanKind.LLM, name=name, tracer=tracer, model=model, capture_args=False)


def trace_tool(name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    return traced(SpanKind.TOOL, name=name, tracer=tracer)
