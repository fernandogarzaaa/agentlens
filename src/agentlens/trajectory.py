"""Trajectory matching and trajectory constraints.

A trajectory is the ordered list of tool names an agent called. Match modes:

* ``strict``       same tools, same order, same count
* ``unordered``    same tools and counts, any order
* ``subset``       every expected call (with multiplicity) appears; extra calls are allowed
* ``subsequence``  expected calls appear in this order, possibly with other calls in between
* ``none``         not checked
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentlens.metrics import ToolCall

__all__ = ["MODES", "match_trajectory", "Constraints", "check_constraints"]

MODES = ("strict", "unordered", "subset", "subsequence", "none")


def _is_subsequence(needle: Sequence[str], hay: Sequence[str]) -> bool:
    it = iter(hay)
    return all(any(h == n for h in it) for n in needle)


def match_trajectory(expected: Sequence[str], actual: Sequence[str], mode: str = "strict") -> bool:
    """Whether ``actual`` tool names satisfy ``expected`` under ``mode``."""
    if mode == "strict":
        return list(expected) == list(actual)
    if mode == "unordered":
        return Counter(expected) == Counter(actual)
    if mode == "subset":
        missing = Counter(expected) - Counter(actual)
        return not missing
    if mode == "subsequence":
        return _is_subsequence(expected, actual)
    if mode == "none":
        return True
    raise ValueError(f"unknown trajectory mode {mode!r}; expected one of {MODES}")


@dataclass
class Constraints:
    """Hard rules on a run, checked independently of the expected trajectory."""

    max_steps: int | None = None
    max_tool_calls: int | None = None
    must_call: list[str] = field(default_factory=list)
    must_not_call: list[str] = field(default_factory=list)
    order: list[tuple[str, str]] = field(default_factory=list)  # (a, b): if b is called, a was called first
    no_repeat: bool = False  # the same tool with the same arguments twice is a violation

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Constraints:
        if not d:
            return cls()
        unknown = set(d) - {"max_steps", "max_tool_calls", "must_call", "must_not_call", "order", "no_repeat"}
        if unknown:
            raise ValueError(f"unknown constraints: {sorted(unknown)}")
        order = []
        for pair in d.get("order") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"constraint 'order' entries must be [before, after] pairs, got {pair!r}")
            order.append((str(pair[0]), str(pair[1])))
        return cls(
            max_steps=None if d.get("max_steps") is None else int(d["max_steps"]),
            max_tool_calls=None if d.get("max_tool_calls") is None else int(d["max_tool_calls"]),
            must_call=[str(x) for x in d.get("must_call") or []],
            must_not_call=[str(x) for x in d.get("must_not_call") or []],
            order=order,
            no_repeat=bool(d.get("no_repeat", False)),
        )

    def merged_over(self, base: Constraints) -> Constraints:
        """This case's constraints with unset fields filled from suite defaults."""
        return Constraints(
            max_steps=self.max_steps if self.max_steps is not None else base.max_steps,
            max_tool_calls=self.max_tool_calls if self.max_tool_calls is not None else base.max_tool_calls,
            must_call=self.must_call or base.must_call,
            must_not_call=self.must_not_call or base.must_not_call,
            order=self.order or base.order,
            no_repeat=self.no_repeat or base.no_repeat,
        )


def check_constraints(constraints: Constraints, calls: Sequence[ToolCall], steps: int) -> list[str]:
    """Return human-readable violations (empty list = all constraints hold)."""
    violations: list[str] = []
    names = [c.name for c in calls]
    if constraints.max_steps is not None and steps > constraints.max_steps:
        violations.append(f"{steps} steps > max_steps {constraints.max_steps}")
    if constraints.max_tool_calls is not None and len(calls) > constraints.max_tool_calls:
        violations.append(f"{len(calls)} tool calls > max_tool_calls {constraints.max_tool_calls}")
    for name in constraints.must_call:
        if name not in names:
            violations.append(f"never called required tool {name!r}")
    for name in constraints.must_not_call:
        if name in names:
            violations.append(f"called forbidden tool {name!r}")
    for before, after in constraints.order:
        if after in names and (before not in names or names.index(before) > names.index(after)):
            violations.append(f"{after!r} called before {before!r}")
    if constraints.no_repeat:
        seen: set[tuple[str, str]] = set()
        for c in calls:
            key = (c.name, repr(sorted(c.arguments.items())))
            if key in seen:
                violations.append(f"repeated identical call to {c.name!r}")
                break
            seen.add(key)
    return violations
