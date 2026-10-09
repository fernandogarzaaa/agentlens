"""Read traces back from a JSONL file or SQLite database written by the exporters."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentlens.metrics import ToolCall
from agentlens.tracer import (
    ATTR_CASE_ID,
    ATTR_INPUT,
    ATTR_OUTPUT,
    ATTR_TOOL_ARGS,
    ATTR_TOOL_NAME,
    RunSummary,
    Span,
    SpanKind,
)

__all__ = ["Trace", "group_traces", "load_spans", "load_traces"]

SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


@dataclass
class Trace:
    """All spans that share one trace id: usually one agent run."""

    trace_id: str
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        roots = [s for s in self.spans if s.parent_id is None]
        return min(roots, key=lambda s: s.start_ns) if roots else None

    @property
    def name(self) -> str:
        return self.root.name if self.root else self.trace_id

    @property
    def input(self) -> Any:
        return self.root.attributes.get(ATTR_INPUT) if self.root else None

    @property
    def output(self) -> Any:
        return self.root.attributes.get(ATTR_OUTPUT) if self.root else None

    @property
    def case_id(self) -> str | None:
        value = self.root.attributes.get(ATTR_CASE_ID) if self.root else None
        return str(value) if value is not None else None

    @property
    def start_ns(self) -> int:
        return min(s.start_ns for s in self.spans)

    def tool_calls(self) -> list[ToolCall]:
        """Tool calls in the order they started."""
        tools = sorted((s for s in self.spans if s.kind == SpanKind.TOOL), key=lambda s: s.start_ns)
        calls = []
        for s in tools:
            args = s.attributes.get(ATTR_TOOL_ARGS) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            calls.append(ToolCall(str(s.attributes.get(ATTR_TOOL_NAME, s.name)), dict(args)))
        return calls

    def summary(self) -> RunSummary:
        return RunSummary.from_spans(self.spans)

    def children(self, span: Span) -> list[Span]:
        return sorted((s for s in self.spans if s.parent_id == span.span_id), key=lambda s: s.start_ns)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "spans": [s.to_dict() for s in self.spans]}


def group_traces(spans: Iterable[Span]) -> list[Trace]:
    """Group spans by trace id, ordered by start time."""
    by_id: dict[str, list[Span]] = {}
    for s in spans:
        by_id.setdefault(s.trace_id, []).append(s)
    traces = [Trace(tid, sorted(group, key=lambda s: s.start_ns)) for tid, group in by_id.items()]
    return sorted(traces, key=lambda t: t.start_ns)


def _load_jsonl(path: Path) -> list[Span]:
    spans = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(Span.from_dict(json.loads(line)))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: not a span record ({exc})") from None
    return spans


def _load_sqlite(path: Path) -> list[Span]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT trace_id, span_id, parent_id, name, kind, start_ns, end_ns, status, status_message,"
            " attributes, events FROM spans ORDER BY start_ns"
        ).fetchall()
    finally:
        conn.close()
    return [
        Span.from_dict(
            {
                "trace_id": r[0],
                "span_id": r[1],
                "parent_id": r[2],
                "name": r[3],
                "kind": r[4],
                "start_ns": r[5],
                "end_ns": r[6],
                "status": r[7],
                "status_message": r[8],
                "attributes": json.loads(r[9]),
                "events": json.loads(r[10]),
            }
        )
        for r in rows
    ]


def load_spans(path: str | Path) -> list[Span]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace store not found: {p}")
    if p.suffix.lower() in SQLITE_SUFFIXES:
        return _load_sqlite(p)
    return _load_jsonl(p)


def load_traces(paths: str | Path | Sequence[str | Path]) -> list[Trace]:
    """Load one or more JSONL / SQLite trace stores and group spans into traces."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    spans: list[Span] = []
    for p in paths:
        spans.extend(load_spans(p))
    return group_traces(spans)
