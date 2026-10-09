"""LangGraph / LangChain callback adapter (``pip install "agentlens[langgraph]"``).

    from agentlens import Tracer, JSONLExporter
    from agentlens.adapters.langgraph import AgentLensCallbackHandler

    tracer = Tracer("my-graph", exporters=[JSONLExporter("traces.jsonl")])
    handler = AgentLensCallbackHandler(tracer)
    graph.invoke({"messages": [("user", "hi")]}, config={"callbacks": [handler]})

Mapping: the outermost chain/graph run becomes a ``run`` span (new trace), every LangGraph node
(chains carrying ``langgraph_node`` metadata) an ``agent_step``, chat-model / LLM runs ``llm`` spans
with token usage and cost, and tools ``tool`` spans with arguments and results. Other nested
chains are folded into their nearest traced ancestor to keep traces readable.

To evaluate a LangGraph agent with ``agentlens eval``, wrap it in an object with the
:class:`agentlens.agent.Agent` ``run`` method and pass this handler with ``tracer`` (see
``examples/langgraph_agent.py``).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agentlens.tracer import ATTR_INPUT, Span, SpanKind, Tracer, current_span, get_tracer

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError('the LangGraph adapter needs: pip install "agentlens[langgraph]"') from exc

__all__ = ["AgentLensCallbackHandler"]


def _jsonish(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    content = getattr(value, "content", None)
    if content is not None:
        return {"type": type(value).__name__, "content": _jsonish(content)}
    return repr(value)


def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    params = kwargs.get("invocation_params") or {}
    for key in ("model", "model_name", "model_id"):
        if params.get(key):
            return str(params[key])
    meta = kwargs.get("metadata") or {}
    if meta.get("ls_model_name"):
        return str(meta["ls_model_name"])
    ser_kwargs = (serialized or {}).get("kwargs") or {}
    for key in ("model", "model_name"):
        if ser_kwargs.get(key):
            return str(ser_kwargs[key])
    name = (serialized or {}).get("name") or ((serialized or {}).get("id") or ["unknown"])[-1]
    return str(name)


def _usage(response: Any) -> tuple[int, int, str | None]:
    """Token usage from an ``LLMResult``: ``usage_metadata`` on the message, else ``llm_output``."""
    inp = out = 0
    model = None
    found = False
    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            msg = getattr(gen, "message", None)
            meta = getattr(msg, "usage_metadata", None) if msg is not None else None
            if meta:
                inp += int(meta.get("input_tokens", 0))
                out += int(meta.get("output_tokens", 0))
                found = True
            rmeta = getattr(msg, "response_metadata", None) if msg is not None else None
            if rmeta and rmeta.get("model_name"):
                model = str(rmeta["model_name"])
    if not found:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        model = model or llm_output.get("model_name")
    return inp, out, model


class AgentLensCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that writes agentlens spans."""

    raise_error = False
    run_inline = True  # keep callbacks on the caller's thread so contextvars-based parents line up

    def __init__(self, tracer: Tracer | None = None, run_name: str | None = None) -> None:
        super().__init__()
        self.tracer = tracer or get_tracer()
        self.run_name = run_name
        self._spans: dict[UUID, Span] = {}
        self._parents: dict[UUID, UUID | None] = {}
        self._step = 0

    # ---- bookkeeping ----------------------------------------------------------------
    def _ancestor_span(self, parent_run_id: UUID | None) -> Span | None:
        seen = set()
        while parent_run_id is not None and parent_run_id not in seen:
            seen.add(parent_run_id)
            if parent_run_id in self._spans:
                return self._spans[parent_run_id]
            parent_run_id = self._parents.get(parent_run_id)
        return None

    def _inside_step(self, parent_run_id: UUID | None) -> bool:
        """LangGraph copies node metadata onto every run inside the node; only the outermost is a step."""
        ancestor = self._ancestor_span(parent_run_id)
        return ancestor is not None and ancestor.kind == SpanKind.AGENT_STEP

    def _start(
        self, run_id: UUID, parent_run_id: UUID | None, name: str, kind: str, attributes: dict[str, Any]
    ) -> Span:
        self._parents[run_id] = parent_run_id
        parent = self._ancestor_span(parent_run_id)
        if parent is None and kind != SpanKind.RUN:
            parent = current_span()  # e.g. inside an agentlens eval run
        span = self.tracer.start_span(name, kind=kind, attributes=attributes, parent=parent, new_trace=parent is None)
        self._spans[run_id] = span
        return span

    def _end(self, run_id: UUID, error: BaseException | None = None) -> Span | None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return None
        if error is not None:
            span.record_error(error)
        span.end()
        return span

    # ---- chains / graph nodes ---------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        top_level = self._ancestor_span(parent_run_id) is None and parent_run_id not in self._parents
        if top_level:
            outer = current_span()
            if outer is not None:
                # Already inside a traced run (e.g. agentlens eval): the graph is a child, not a new trace.
                self._start(run_id, parent_run_id, str(name), SpanKind.OTHER, {ATTR_INPUT: _jsonish(inputs)})
            else:
                self._start(
                    run_id, parent_run_id, self.run_name or str(name), SpanKind.RUN, {ATTR_INPUT: _jsonish(inputs)}
                )
        elif metadata.get("langgraph_node") and not self._inside_step(parent_run_id):
            node = str(metadata["langgraph_node"])
            self._start(
                run_id,
                parent_run_id,
                f"node {node}",
                SpanKind.AGENT_STEP,
                {
                    "agentlens.step.index": self._step,
                    "langgraph.node": node,
                    "langgraph.step": metadata.get("langgraph_step"),
                },
            )
            self._step += 1
        else:
            self._parents[run_id] = parent_run_id

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span is not None and span.kind in (SpanKind.RUN, SpanKind.OTHER):
            span.set_output(_jsonish(outputs))
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)

    # ---- models -----------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        model = _model_name(serialized, kwargs)
        self._start(
            run_id,
            parent_run_id,
            f"chat {model}",
            SpanKind.LLM,
            {"gen_ai.operation.name": "chat", "gen_ai.request.model": model},
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        model = _model_name(serialized, kwargs)
        self._start(
            run_id,
            parent_run_id,
            f"completion {model}",
            SpanKind.LLM,
            {"gen_ai.operation.name": "text_completion", "gen_ai.request.model": model},
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span is not None:
            inp, out, model = _usage(response)
            span.record_usage(inp, out, model=model)
            try:
                gen = response.generations[0][0]
                msg = getattr(gen, "message", None)
                text = getattr(msg, "content", None) if msg is not None else getattr(gen, "text", None)
                if text:
                    span.set_output(_jsonish(text))
                calls = getattr(msg, "tool_calls", None) if msg is not None else None
                if calls:
                    span.set_attribute("agentlens.requested_tool_calls", _jsonish(calls))
            except (AttributeError, IndexError, TypeError):
                pass
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)

    # ---- tools ------------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.get("name") or (serialized or {}).get("name") or "tool")
        if inputs is not None:
            args: Any = _jsonish(inputs)
        else:
            try:
                args = json.loads(input_str)
            except (TypeError, ValueError):
                args = {"input": input_str}
            if not isinstance(args, dict):
                args = {"input": args}
        self._start(
            run_id,
            parent_run_id,
            f"execute_tool {name}",
            SpanKind.TOOL,
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name, "gen_ai.tool.call.arguments": args},
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        span = self._spans.get(run_id)
        if span is not None:
            span.set_result(_jsonish(output))
        self._end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)
