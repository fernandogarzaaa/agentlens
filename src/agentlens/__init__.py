"""agentlens: tracing and evaluation for tool-calling LLM agents.

The names exported here are the stable public API (see docs/tracer-api.md for the tracer contract).
"""

from agentlens._version import __version__
from agentlens.agent import AgentResult, CannedToolset, Tool, ToolAgent, Toolset, tool
from agentlens.cases import Case, Suite, load_suite
from agentlens.cost import ModelPrice, Pricing
from agentlens.exporters import InMemoryExporter, JSONLExporter, OTLPExporter, SQLiteExporter
from agentlens.harness import CaseResult, EvalResult, run_eval
from agentlens.llm import MockChatClient, OpenAICompatClient
from agentlens.metrics import ToolCall
from agentlens.store import Trace, load_traces
from agentlens.tracer import (
    RunSummary,
    Span,
    SpanKind,
    Tracer,
    configure,
    current_span,
    get_tracer,
    set_tracer,
    trace_agent_step,
    trace_llm,
    trace_run,
    trace_tool,
    traced,
)

__all__ = [
    "__version__",
    # tracing
    "Tracer",
    "Span",
    "SpanKind",
    "RunSummary",
    "configure",
    "get_tracer",
    "set_tracer",
    "current_span",
    "traced",
    "trace_run",
    "trace_agent_step",
    "trace_llm",
    "trace_tool",
    "Pricing",
    "ModelPrice",
    "InMemoryExporter",
    "JSONLExporter",
    "SQLiteExporter",
    "OTLPExporter",
    "Trace",
    "load_traces",
    # agents
    "Tool",
    "tool",
    "Toolset",
    "CannedToolset",
    "ToolAgent",
    "AgentResult",
    "OpenAICompatClient",
    "MockChatClient",
    # eval
    "ToolCall",
    "Case",
    "Suite",
    "load_suite",
    "run_eval",
    "EvalResult",
    "CaseResult",
]
