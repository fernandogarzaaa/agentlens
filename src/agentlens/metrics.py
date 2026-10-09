"""Per-case metrics: tool selection, argument matching, final-answer checks and aggregation helpers.

Expected argument values may be literals or matchers:

* ``{"$any": [v1, v2, ...]}``   any of the listed values is accepted
* ``"$absent"``                  the argument must not be passed (inside ``$any``: it may be omitted)
* ``{"$regex": "pattern"}``      ``re.fullmatch`` against the actual value (as a string)

Strings are compared after Unicode case folding and whitespace normalisation by default; numbers
compare by value (``2 == 2.0``) but booleans never equal numbers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ABSENT",
    "ToolCall",
    "PRF",
    "ArgumentScore",
    "ArgumentMatch",
    "AnswerCheck",
    "values_equal",
    "score_arguments",
    "tool_selection",
    "match_calls",
    "argument_match",
    "check_answer",
    "percentile",
    "mean",
]

ABSENT = "$absent"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        if "name" not in d:
            raise ValueError(f"tool call needs a 'name': {d!r}")
        args = d.get("arguments", d.get("args")) or {}
        if not isinstance(args, dict):
            raise ValueError(f"tool call arguments must be a mapping: {d!r}")
        return cls(str(d["name"]), dict(args))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


def _norm_str(s: str) -> str:
    return " ".join(s.casefold().split())


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def values_equal(expected: Any, actual: Any, normalize_strings: bool = True) -> bool:
    """Compare an expected value (literal or matcher) with an actual value."""
    if isinstance(expected, dict) and set(expected) == {"$any"}:
        options = expected["$any"]
        return any(values_equal(o, actual, normalize_strings) for o in options if o != ABSENT)
    if isinstance(expected, dict) and set(expected) == {"$regex"}:
        return actual is not None and re.fullmatch(str(expected["$regex"]), str(actual)) is not None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if _is_number(expected):
        if not _is_number(actual):
            return False
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        return _norm_str(expected) == _norm_str(actual) if normalize_strings else expected == actual
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return False
        return all(values_equal(e, a, normalize_strings) for e, a in zip(expected, actual))
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return score_arguments(expected, actual, normalize_strings=normalize_strings).exact
    if expected is None:
        return actual is None
    return bool(expected == actual)


def _may_be_absent(spec: Any) -> bool:
    if spec == ABSENT:
        return True
    return isinstance(spec, dict) and set(spec) == {"$any"} and ABSENT in spec["$any"]


def _arg_ok(spec: Any, actual: dict[str, Any], key: str, normalize_strings: bool) -> bool:
    if key not in actual:
        return _may_be_absent(spec)
    if spec == ABSENT:
        return False
    return values_equal(spec, actual[key], normalize_strings)


@dataclass
class ArgumentScore:
    exact: bool
    partial: float
    matched: list[str]
    mismatched: list[str]
    extra: list[str]


def score_arguments(
    expected: dict[str, Any],
    actual: dict[str, Any],
    allow_extra: bool = False,
    normalize_strings: bool = True,
) -> ArgumentScore:
    """Exact match (every expected key ok, and no unexpected keys unless ``allow_extra``) and partial
    match (fraction of expected keys that are ok)."""
    matched = [k for k in expected if _arg_ok(expected[k], actual, k, normalize_strings)]
    mismatched = [k for k in expected if k not in matched]
    extra = [k for k in actual if k not in expected]
    partial = len(matched) / len(expected) if expected else 1.0
    exact = not mismatched and (allow_extra or not extra)
    return ArgumentScore(exact=exact, partial=partial, matched=matched, mismatched=mismatched, extra=extra)


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    true_positives: int
    n_expected: int
    n_actual: int


def tool_selection(expected: Sequence[ToolCall], actual: Sequence[ToolCall]) -> PRF:
    """Precision / recall / F1 of tool *names* as multisets.

    Calling ``search`` twice when one call was expected gives precision 0.5. When nothing was expected
    and nothing was called the score is perfect; when one side is empty and the other is not, the
    metric that divides by the non-empty side is 0 and the other is 1.
    """
    exp = Counter(c.name for c in expected)
    act = Counter(c.name for c in actual)
    tp = sum((exp & act).values())
    n_exp, n_act = sum(exp.values()), sum(act.values())
    precision = tp / n_act if n_act else (1.0 if n_exp == 0 else 0.0)
    recall = tp / n_exp if n_exp else (1.0 if n_act == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, n_exp, n_act)


def match_calls(
    expected: Sequence[ToolCall],
    actual: Sequence[ToolCall],
    allow_extra: bool = False,
    normalize_strings: bool = True,
) -> list[tuple[int, int | None, ArgumentScore | None]]:
    """Pair each expected call with at most one actual call of the same name.

    Expected calls are taken in order; each picks the unused actual call with the best partial
    argument score (earliest on ties). Returns ``(expected_index, actual_index or None, score)``.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int | None, ArgumentScore | None]] = []
    for i, exp in enumerate(expected):
        best: tuple[int, ArgumentScore] | None = None
        for j, act in enumerate(actual):
            if j in used or act.name != exp.name:
                continue
            score = score_arguments(exp.arguments, act.arguments, allow_extra, normalize_strings)
            if best is None or (score.exact, score.partial) > (best[1].exact, best[1].partial):
                best = (j, score)
        if best is None:
            pairs.append((i, None, None))
        else:
            used.add(best[0])
            pairs.append((i, best[0], best[1]))
    return pairs


@dataclass
class ArgumentMatch:
    exact: float  # fraction of expected calls matched with exactly right arguments
    partial: float  # mean fraction of expected argument keys right (unmatched call = 0)
    all_exact: bool
    pairs: list[tuple[int, int | None, ArgumentScore | None]]


def argument_match(
    expected: Sequence[ToolCall],
    actual: Sequence[ToolCall],
    allow_extra: bool = False,
    normalize_strings: bool = True,
) -> ArgumentMatch:
    if not expected:
        return ArgumentMatch(exact=1.0, partial=1.0, all_exact=True, pairs=[])
    pairs = match_calls(expected, actual, allow_extra, normalize_strings)
    exact_n = sum(1 for _, _, s in pairs if s is not None and s.exact)
    partial_sum = sum(s.partial for _, _, s in pairs if s is not None)
    return ArgumentMatch(
        exact=exact_n / len(expected),
        partial=partial_sum / len(expected),
        all_exact=exact_n == len(expected),
        pairs=pairs,
    )


@dataclass
class AnswerCheck:
    """Final-answer checks. Every check that is set must pass."""

    contains: list[str] = field(default_factory=list)  # all of these substrings
    contains_any: list[str] = field(default_factory=list)  # at least one of these
    not_contains: list[str] = field(default_factory=list)
    regex: str | None = None  # re.search
    equals: str | None = None  # normalised equality
    number: float | None = None  # some number in the answer equals this within tolerance
    tolerance: float = 1e-6
    case_sensitive: bool = False
    judge: bool = False  # also require an LLM-as-judge pass (when the eval runs with a judge)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AnswerCheck | None:
        if not d:
            return None
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown final_answer checks: {sorted(unknown)}")

        def as_list(v: Any) -> list[str]:
            if v is None:
                return []
            return [str(x) for x in v] if isinstance(v, list) else [str(v)]

        return cls(
            contains=as_list(d.get("contains")),
            contains_any=as_list(d.get("contains_any")),
            not_contains=as_list(d.get("not_contains")),
            regex=d.get("regex"),
            equals=None if d.get("equals") is None else str(d["equals"]),
            number=None if d.get("number") is None else float(d["number"]),
            tolerance=float(d.get("tolerance", 1e-6)),
            case_sensitive=bool(d.get("case_sensitive", False)),
            judge=bool(d.get("judge", False)),
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.contains
            or self.contains_any
            or self.not_contains
            or self.regex is not None
            or self.equals is not None
            or self.number is not None
        )


_NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def check_answer(answer: str | None, check: AnswerCheck | None) -> tuple[bool, list[str]]:
    """Run the checks; returns ``(passed, failures)``."""
    if check is None or check.is_empty:
        return True, []
    text = answer or ""

    def norm(s: str) -> str:
        return s if check.case_sensitive else s.casefold()

    hay = norm(text)
    failures: list[str] = []
    for needle in check.contains:
        if norm(needle) not in hay:
            failures.append(f"missing {needle!r}")
    if check.contains_any and not any(norm(n) in hay for n in check.contains_any):
        failures.append(f"none of {check.contains_any!r}")
    for needle in check.not_contains:
        if norm(needle) in hay:
            failures.append(f"contains forbidden {needle!r}")
    if check.regex is not None:
        flags = 0 if check.case_sensitive else re.IGNORECASE
        if re.search(check.regex, text, flags) is None:
            failures.append(f"no match for /{check.regex}/")
    if check.equals is not None:
        same = (
            text.strip() == check.equals.strip() if check.case_sensitive else _norm_str(text) == _norm_str(check.equals)
        )
        if not same:
            failures.append(f"not equal to {check.equals!r}")
    if check.number is not None:
        numbers = [float(m.replace(",", "")) for m in _NUMBER.findall(text)]
        if not any(abs(n - check.number) <= check.tolerance for n in numbers):
            failures.append(f"no number equal to {check.number} (+/- {check.tolerance})")
    return not failures, failures


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]), same as numpy's default method."""
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
