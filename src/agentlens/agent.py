"""Built-in agent: an OpenAI-compatible function-calling loop, fully traced.

    from agentlens import ToolAgent, Toolset, OpenAICompatClient, tool

    @tool
    def get_weather(city: str, unit: str = "celsius") -> dict:
        "Current weather for a city."
        ...

    agent = ToolAgent(OpenAICompatClient("http://localhost:8080/v1"), model="qwen2.5-0.5b-instruct")
    result = agent.run("Weather in Paris?", Toolset([get_weather]))

Every step is an ``agent_step`` span holding one ``llm`` span and one ``tool`` span per call.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentlens.llm import ChatClient, LLMError
from agentlens.metrics import values_equal
from agentlens.tracer import Tracer, get_tracer

__all__ = ["Tool", "tool", "Toolset", "CannedToolset", "AgentResult", "Agent", "ToolAgent", "load_toolset"]

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, getattr(types, "UnionType", object)):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    if origin is typing.Literal:
        values = list(typing.get_args(annotation))
        return {"type": _JSON_TYPES.get(type(values[0]), "string"), "enum": values}
    if origin in (list, Sequence, tuple):
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    @classmethod
    def from_function(cls, fn: Callable[..., Any], name: str | None = None, description: str | None = None) -> Tool:
        sig = inspect.signature(fn)
        try:
            hints = typing.get_type_hints(fn)
        except Exception:  # pragma: no cover - unresolvable forward refs
            hints = {}
        props: dict[str, Any] = {}
        required = []
        for pname, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            props[pname] = _json_type(hints.get(pname, str))
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        doc = inspect.getdoc(fn) or ""
        return cls(
            name=name or fn.__name__,
            description=description or (doc.splitlines()[0] if doc else fn.__name__),
            parameters={"type": "object", "properties": props, "required": required},
            fn=fn,
        )

    @classmethod
    def from_schema(cls, schema: dict[str, Any], fn: Callable[..., Any] | None = None) -> Tool:
        fn_schema = schema.get("function", schema)
        return cls(
            name=str(fn_schema["name"]),
            description=str(fn_schema.get("description", "")),
            parameters=dict(fn_schema.get("parameters") or {"type": "object", "properties": {}}),
            fn=fn,
        )


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None) -> Any:
    """Decorator that attaches a :class:`Tool` (JSON schema from type hints) as ``fn.__agentlens_tool__``."""

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__agentlens_tool__ = Tool.from_function(f, name=name, description=description)  # type: ignore[attr-defined]
        return f

    return wrap(fn) if fn is not None else wrap


class Toolset:
    """Named tools the agent may call."""

    def __init__(self, tools: Sequence[Tool | Callable[..., Any]] = ()) -> None:
        self.tools: dict[str, Tool] = {}
        for t in tools:
            self.add(t)

    def add(self, t: Tool | Callable[..., Any]) -> None:
        if not isinstance(t, Tool):
            t = getattr(t, "__agentlens_tool__", None) or Tool.from_function(t)
        self.tools[t.name] = t

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool. Unknown tools, bad arguments and exceptions come back as ``{"error": ...}``
        so the model can see and recover from them, the same way a production agent loop would."""
        t = self.tools.get(name)
        if t is None:
            return {"error": f"unknown tool {name!r}; available: {self.names()}"}
        if t.fn is None:
            return {"error": f"tool {name!r} has no implementation"}
        try:
            return t.fn(**arguments)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - tool failures are data for the model
            return {"error": f"{type(exc).__name__}: {exc}"}


class CannedToolset(Toolset):
    """Tools declared as schemas whose results come from a table (``tool_results`` in a suite)."""

    def __init__(self, schemas: Sequence[dict[str, Any]], results: dict[str, Any] | None = None) -> None:
        super().__init__([Tool.from_schema(s) for s in schemas])
        self.results = dict(results or {})

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.tools:
            return {"error": f"unknown tool {name!r}; available: {self.names()}"}
        spec = self.results.get(name)
        if spec is None:
            return {"status": "ok"}
        if isinstance(spec, list) and all(isinstance(r, dict) and "result" in r for r in spec):
            for rule in spec:
                when = rule.get("when") or {}
                if all(k in arguments and values_equal(v, arguments[k]) for k, v in when.items()):
                    return rule["result"]
            return {"error": "no canned result matches these arguments"}
        return spec


def load_toolset(ref: str, base_dir: Path | None = None) -> Toolset:
    """Load ``path/to/tools.py:NAME`` or ``package.module:NAME``. NAME may be a Toolset, a list of
    functions/tools, or a zero-argument factory returning either."""
    if ":" not in ref:
        raise ValueError(f"toolset reference must look like 'module:attr' or 'file.py:attr', got {ref!r}")
    mod_ref, attr = ref.rsplit(":", 1)
    if mod_ref.endswith(".py"):
        path = Path(mod_ref)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        mod_name = f"agentlens_toolset_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import toolset file {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_ref)
    obj = getattr(module, attr)
    if callable(obj) and not isinstance(obj, Toolset) and not hasattr(obj, "__agentlens_tool__"):
        try:
            if not inspect.signature(obj).parameters:
                obj = obj()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, Toolset):
        return obj
    if isinstance(obj, (list, tuple)):
        return Toolset(list(obj))
    raise ValueError(f"{ref} is not a Toolset or a list of tools")


@dataclass
class AgentResult:
    final_answer: str | None
    steps: int = 0
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    exhausted: bool = False  # stopped at max_steps without a plain-text answer


class Agent(Protocol):
    """What the eval harness runs. Implementations should emit spans through ``tracer``."""

    def run(
        self,
        input: str | list[dict[str, Any]],
        toolset: Toolset,
        system_prompt: str | None = None,
        max_steps: int = 8,
        tracer: Tracer | None = None,
    ) -> AgentResult: ...


def _tool_message(call_id: str, result: Any) -> dict[str, Any]:
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class ToolAgent:
    """The classic loop: ask the model, run the tools it asks for, feed results back, repeat until it
    answers in plain text or ``max_steps`` model calls have been made."""

    def __init__(
        self,
        client: ChatClient,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 512,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def run(
        self,
        input: str | list[dict[str, Any]],
        toolset: Toolset,
        system_prompt: str | None = None,
        max_steps: int = 8,
        tracer: Tracer | None = None,
    ) -> AgentResult:
        t = tracer or get_tracer()
        messages: list[dict[str, Any]] = []
        system = system_prompt or self.system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        if isinstance(input, str):
            messages.append({"role": "user", "content": input})
        else:
            messages.extend(input)
        schemas = toolset.schemas()
        for step in range(max_steps):
            with t.agent_step(step):
                with t.llm_call(self.model) as llm:
                    try:
                        resp = self.client.chat(
                            messages,
                            self.model,
                            tools=schemas or None,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                    except LLMError as exc:
                        llm.record_error(exc)
                        return AgentResult(None, steps=step + 1, error=str(exc), messages=messages)
                    llm.record_usage(resp.input_tokens, resp.output_tokens, model=resp.model or None)
                    llm.set_attribute("gen_ai.response.finish_reasons", [resp.finish_reason])
                    if resp.content:
                        llm.set_output(resp.content)
                messages.append(resp.assistant_message())
                if not resp.tool_calls:
                    return AgentResult(resp.content or "", steps=step + 1, messages=messages)
                for tc in resp.tool_calls:
                    with t.tool_call(tc.name, arguments=tc.arguments, call_id=tc.id) as span:
                        if tc.parse_error:
                            result: Any = {"error": f"arguments were not valid JSON: {tc.raw_arguments[:200]}"}
                            span.record_error("invalid tool-call arguments JSON")
                        else:
                            result = toolset.call(tc.name, tc.arguments)
                            if isinstance(result, dict) and "error" in result and len(result) == 1:
                                span.record_error(str(result["error"]))
                        span.set_result(result)
                    messages.append(_tool_message(tc.id, result))
        return AgentResult(None, steps=max_steps, messages=messages, exhausted=True)
