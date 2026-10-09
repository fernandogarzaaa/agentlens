"""Convert Berkeley Function Calling Leaderboard (BFCL v4) categories into agentlens suites.

Data comes from the gorilla repository at a pinned commit (Apache-2.0). Supported categories:
``simple_python``, ``multiple``, ``parallel``, ``parallel_multiple`` (expected calls from
``possible_answer``) and ``irrelevance`` (expected: no tool call at all).

Conversion choices, so numbers are interpreted correctly:

* BFCL parameter types are mapped to JSON Schema (``dict`` -> ``object``, ``float`` -> ``number``,
  ``tuple`` -> ``array``, ``any`` -> untyped) and dots in function names become underscores
  (OpenAI-style APIs reject dots), the same renaming BFCL's own OpenAI handler applies.
* Every acceptable value list becomes ``{"$any": [...]}``; BFCL's ``""`` ("may be omitted")
  becomes ``"$absent"``.
* Scoring uses agentlens' matchers (case/whitespace-insensitive strings, numeric equality), not
  BFCL's AST checker, so results are **agentlens metrics on BFCL data, not official BFCL scores**.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from agentlens.metrics import ABSENT

__all__ = ["BFCL_COMMIT", "CATEGORIES", "download", "convert", "convert_schema", "build_suite"]

BFCL_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/berkeley-function-call-leaderboard/bfcl_eval/data"
)
CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance")
_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "number": "number",
}


def _fetch(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def download(category: str, dest: str | Path, commit: str = BFCL_COMMIT) -> tuple[Path, Path | None]:
    """Download questions (+ possible answers when the category has them) into ``dest``."""
    if category not in CATEGORIES:
        raise ValueError(f"unsupported BFCL category {category!r}; choose from {CATEGORIES}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = BASE_URL.format(commit=commit)
    q_path = dest / f"BFCL_v4_{category}.json"
    q_path.write_text("\n".join(json.dumps(r) for r in _fetch(f"{base}/BFCL_v4_{category}.json")) + "\n")
    if category == "irrelevance":
        return q_path, None
    a_path = dest / f"BFCL_v4_{category}.answers.json"
    a_path.write_text(
        "\n".join(json.dumps(r) for r in _fetch(f"{base}/possible_answer/BFCL_v4_{category}.json")) + "\n"
    )
    return q_path, a_path


def _safe_name(name: str) -> str:
    return name.replace(".", "_")


def convert_schema(node: Any) -> Any:
    """Recursively map BFCL parameter schemas onto JSON Schema."""
    if isinstance(node, list):
        return [convert_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            mapped = _TYPE_MAP.get(value)
            if mapped is not None:
                out["type"] = mapped
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {k: convert_schema(v) for k, v in value.items()}
            continue
        out[key] = convert_schema(value)
    return out


def _tool(fn: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _safe_name(fn["name"]),
            "description": fn.get("description", ""),
            "parameters": convert_schema(fn.get("parameters") or {"type": "object", "properties": {}}),
        },
    }


def _expected_option(option: Any) -> Any:
    """One acceptable value. Inside it, dict values are again lists of acceptable values (BFCL's
    nested format); list elements are literal (but dicts inside lists follow the dict rule)."""
    if isinstance(option, dict):
        return {k: _expected_value(v) for k, v in option.items()}
    if isinstance(option, list):
        return [_expected_option(x) for x in option]
    return option


def _expected_value(options: Any) -> Any:
    if not isinstance(options, list):
        return _expected_option(options)
    values: list[Any] = [ABSENT if o == "" else _expected_option(o) for o in options]
    if len(values) == 1 and values[0] != ABSENT:
        return values[0]
    return {"$any": values}


def convert(category: str, questions: list[dict[str, Any]], answers: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return a suite dict (``agentlens eval`` reads it as JSON)."""
    by_id = {a["id"]: a for a in answers or []}
    trajectory = "unordered" if category.startswith("parallel") else "strict"
    cases = []
    for q in questions:
        turns = q["question"]
        messages = turns[0] if turns and isinstance(turns[0], list) else turns
        expected: list[dict[str, Any]] = []
        if category != "irrelevance":
            gt = by_id.get(q["id"])
            if gt is None:
                continue
            for call in gt["ground_truth"]:
                ((name, args),) = call.items()
                expected.append(
                    {"name": _safe_name(name), "arguments": {k: _expected_value(v) for k, v in args.items()}}
                )
        cases.append(
            {
                "id": q["id"],
                "input": messages,
                "tools": [_tool(f) for f in q["function"]],
                "expected_tool_calls": expected,
                "trajectory": trajectory,
                "max_steps": 1,
                "tags": [f"bfcl:{category}"],
            }
        )
    return {
        "suite": f"bfcl-v4-{category}",
        "description": f"BFCL v4 {category} (gorilla@{BFCL_COMMIT[:7]}), converted by agentlens. Single model turn.",
        "system_prompt": None,
        "cases": cases,
    }


def build_suite(category: str, dest: str | Path, limit: int | None = None, commit: str = BFCL_COMMIT) -> Path:
    """Download + convert; writes ``dest/bfcl_<category>.json`` and returns its path."""
    q_path, a_path = download(category, dest, commit)
    questions = [json.loads(line) for line in q_path.read_text().splitlines() if line.strip()]
    answers = [json.loads(line) for line in a_path.read_text().splitlines() if line.strip()] if a_path else None
    suite = convert(category, questions, answers)
    if limit is not None:
        suite["cases"] = suite["cases"][:limit]
    out = Path(dest) / f"bfcl_{category}.json"
    out.write_text(json.dumps(suite, indent=1) + "\n")
    return out
