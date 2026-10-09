"""Test-suite schema: YAML or JSON files of agent test cases.

.. code-block:: yaml

    suite: travel-desk
    system_prompt: You are a travel assistant. Use the tools.
    toolset: tools.py:TOOLSET           # python tools (path relative to this file, or module:attr)
    defaults:
      trajectory: unordered             # strict | unordered | subset | subsequence | none
      max_steps: 6
      allow_extra_args: false
      constraints: {must_not_call: [delete_booking]}
    cases:
      - id: weather-paris
        input: What's the weather in Paris in celsius?
        expected_tool_calls:
          - name: get_weather
            arguments: {city: Paris, unit: {"$any": [celsius, "$absent"]}}
        trajectory: strict
        final_answer: {contains: ["18"]}
        constraints: {max_steps: 3}
        tags: [weather]

Instead of a ``toolset`` a suite (or case) can declare OpenAI-style ``tools`` schemas plus canned
``tool_results`` so no Python is needed::

    tools: [{type: function, function: {name: get_weather, parameters: {...}}}]
    tool_results:
      get_weather:
        - when: {city: Paris}
          result: {temp_c: 18}
        - result: {error: unknown city}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentlens.metrics import AnswerCheck, ToolCall
from agentlens.trajectory import MODES, Constraints

__all__ = ["Case", "Suite", "SuiteError", "load_suite"]


class SuiteError(ValueError):
    """Malformed suite file."""


@dataclass
class Case:
    id: str
    input: str | list[dict[str, Any]]
    expected_tool_calls: list[ToolCall] = field(default_factory=list)
    trajectory: str = "unordered"
    final_answer: AnswerCheck | None = None
    constraints: Constraints = field(default_factory=Constraints)
    allow_extra_args: bool = False
    max_steps: int = 8
    tools: list[dict[str, Any]] | None = None
    tool_results: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        if isinstance(self.input, str):
            return self.input
        users = [m.get("content", "") for m in self.input if m.get("role") == "user"]
        return str(users[-1]) if users else json.dumps(self.input)

    @property
    def expected_names(self) -> list[str]:
        return [c.name for c in self.expected_tool_calls]


@dataclass
class Suite:
    name: str
    cases: list[Case]
    system_prompt: str | None = None
    toolset: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_results: dict[str, Any] | None = None
    base_dir: Path = field(default_factory=Path.cwd)
    description: str = ""

    def filter(self, tags: list[str] | None = None, ids: list[str] | None = None, limit: int | None = None) -> Suite:
        cases = self.cases
        if tags:
            cases = [c for c in cases if set(tags) & set(c.tags)]
        if ids:
            wanted = set(ids)
            cases = [c for c in cases if c.id in wanted]
        if limit is not None:
            cases = cases[:limit]
        return Suite(
            self.name,
            cases,
            self.system_prompt,
            self.toolset,
            self.tools,
            self.tool_results,
            self.base_dir,
            self.description,
        )

    def get(self, case_id: str) -> Case | None:
        return next((c for c in self.cases if c.id == case_id), None)


_CASE_KEYS = {
    "id",
    "input",
    "expected_tool_calls",
    "trajectory",
    "final_answer",
    "constraints",
    "allow_extra_args",
    "max_steps",
    "tools",
    "tool_results",
    "tags",
    "metadata",
}
_SUITE_KEYS = {"suite", "name", "description", "system_prompt", "toolset", "tools", "tool_results", "defaults", "cases"}


def _parse_case(raw: Any, index: int, defaults: dict[str, Any], base_constraints: Constraints) -> Case:
    if not isinstance(raw, dict):
        raise SuiteError(f"case #{index} must be a mapping")
    unknown = set(raw) - _CASE_KEYS
    if unknown:
        raise SuiteError(f"case #{index}: unknown keys {sorted(unknown)}")
    case_id = str(raw.get("id", f"case-{index}"))
    if "input" not in raw:
        raise SuiteError(f"case {case_id!r}: 'input' is required")
    inp = raw["input"]
    if not isinstance(inp, (str, list)):
        raise SuiteError(f"case {case_id!r}: input must be a string or a list of chat messages")
    try:
        expected = [ToolCall.from_dict(c) for c in raw.get("expected_tool_calls") or []]
        final = AnswerCheck.from_dict(raw.get("final_answer"))
        constraints = Constraints.from_dict(raw.get("constraints")).merged_over(base_constraints)
    except (ValueError, TypeError) as exc:
        raise SuiteError(f"case {case_id!r}: {exc}") from None
    mode = str(raw.get("trajectory", defaults.get("trajectory", "unordered")))
    if mode not in MODES:
        raise SuiteError(f"case {case_id!r}: trajectory must be one of {MODES}, got {mode!r}")
    return Case(
        id=case_id,
        input=inp,
        expected_tool_calls=expected,
        trajectory=mode,
        final_answer=final,
        constraints=constraints,
        allow_extra_args=bool(raw.get("allow_extra_args", defaults.get("allow_extra_args", False))),
        max_steps=int(raw.get("max_steps", defaults.get("max_steps", 8))),
        tools=raw.get("tools"),
        tool_results=raw.get("tool_results"),
        tags=[str(t) for t in raw.get("tags") or []],
        metadata=dict(raw.get("metadata") or {}),
    )


def suite_from_dict(data: Any, base_dir: Path | None = None, default_name: str = "suite") -> Suite:
    if isinstance(data, list):
        data = {"cases": data}
    if not isinstance(data, dict):
        raise SuiteError("a suite must be a mapping with a 'cases' list (or a bare list of cases)")
    unknown = set(data) - _SUITE_KEYS
    if unknown:
        raise SuiteError(f"unknown top-level keys {sorted(unknown)}")
    defaults = data.get("defaults")
    defaults = {} if defaults is None else defaults
    if not isinstance(defaults, dict):
        raise SuiteError("'defaults' must be a mapping")
    try:
        base_constraints = Constraints.from_dict(defaults.get("constraints"))
    except ValueError as exc:
        raise SuiteError(f"defaults: {exc}") from None
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError("'cases' must be a non-empty list")
    cases = [_parse_case(c, i, defaults, base_constraints) for i, c in enumerate(raw_cases)]
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SuiteError(f"duplicate case ids: {dupes}")
    return Suite(
        name=str(data.get("suite") or data.get("name") or default_name),
        cases=cases,
        system_prompt=data.get("system_prompt"),
        toolset=data.get("toolset"),
        tools=data.get("tools"),
        tool_results=data.get("tool_results"),
        base_dir=base_dir or Path.cwd(),
        description=str(data.get("description", "")),
    )


def load_suite(path: str | Path) -> Suite:
    """Load a ``.yaml`` / ``.yml`` / ``.json`` / ``.jsonl`` (one case per line) suite file."""
    p = Path(path)
    if not p.exists():
        raise SuiteError(f"suite file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix == ".jsonl":
            data: Any = {"cases": [json.loads(line) for line in text.splitlines() if line.strip()]}
        elif p.suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise SuiteError(f"{p}: {exc}") from None
    return suite_from_dict(data, base_dir=p.parent.resolve(), default_name=p.stem)
