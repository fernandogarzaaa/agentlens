"""Span exporters: in-memory, JSONL file, SQLite database and OTLP/HTTP (JSON encoding).

All exporters are thread-safe and never raise into the traced code path: an OTLP collector that is
down costs you the spans (counted in ``dropped_spans``), not your agent run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentlens import _http
from agentlens._version import __version__
from agentlens.tracer import Span, SpanKind

__all__ = ["InMemoryExporter", "JSONLExporter", "SQLiteExporter", "OTLPExporter", "spans_to_otlp"]

log = logging.getLogger("agentlens.exporters")


class InMemoryExporter:
    """Keeps finished spans in a list. Used by the eval harness and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.spans: list[Span] = []

    def export(self, spans: Sequence[Span]) -> None:
        with self._lock:
            self.spans.extend(spans)

    def by_trace(self, trace_id: str) -> list[Span]:
        with self._lock:
            return [s for s in self.spans if s.trace_id == trace_id]

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()

    def shutdown(self) -> None:
        return None


class JSONLExporter:
    """Appends one JSON object per finished span to a file (``agentlens report`` / ``serve`` read it)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - closed in shutdown()

    def export(self, spans: Sequence[Span]) -> None:
        with self._lock:
            if self._fh.closed:
                return
            for span in spans:
                self._fh.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")
            self._fh.flush()

    def shutdown(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_ns INTEGER NOT NULL,
    end_ns INTEGER,
    status TEXT NOT NULL,
    status_message TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL,
    events TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS spans_kind ON spans(kind);
"""


class SQLiteExporter:
    """Stores spans in a SQLite database (one row per span, attributes as JSON)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def export(self, spans: Sequence[Span]) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.span_id,
                        s.trace_id,
                        s.parent_id,
                        s.name,
                        s.kind,
                        s.start_ns,
                        s.end_ns,
                        s.status,
                        s.status_message,
                        json.dumps(s.attributes, ensure_ascii=False),
                        json.dumps(s.events, ensure_ascii=False),
                    )
                    for s in spans
                ],
            )
            self._conn.commit()

    def shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---- OTLP --------------------------------------------------------------------------------
# OTLP span kinds: 1 INTERNAL, 3 CLIENT. LLM and tool calls leave the process boundary in the
# common case, so they are CLIENT spans; runs and steps are INTERNAL.
_OTLP_KIND = {SpanKind.LLM: 3, SpanKind.TOOL: 3}
_OTLP_STATUS = {"unset": 0, "ok": 1, "error": 2}


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)) and all(isinstance(v, (str, bool, int, float)) for v in value):
        return {"arrayValue": {"values": [_otlp_value(v) for v in value]}}
    # dicts, nested lists, None: OTLP has no null and collectors handle kvlists unevenly, so JSON-encode.
    return {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}


def _otlp_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _otlp_value(v)} for k, v in attrs.items()]


def spans_to_otlp(spans: Sequence[Span], resource_attributes: dict[str, Any]) -> dict[str, Any]:
    """Encode spans as an OTLP/HTTP JSON ``ExportTraceServiceRequest``."""
    otlp_spans = []
    for s in spans:
        span: dict[str, Any] = {
            "traceId": s.trace_id,
            "spanId": s.span_id,
            "name": s.name,
            "kind": _OTLP_KIND.get(s.kind, 1),
            "startTimeUnixNano": str(s.start_ns),
            "endTimeUnixNano": str(s.end_ns if s.end_ns is not None else s.start_ns),
            "attributes": _otlp_attributes(s.attributes),
            "events": [
                {
                    "timeUnixNano": str(e.get("time_ns", s.start_ns)),
                    "name": e.get("name", "event"),
                    "attributes": _otlp_attributes(e.get("attributes") or {}),
                }
                for e in s.events
            ],
            "status": {"code": _OTLP_STATUS.get(s.status, 0), "message": s.status_message},
        }
        if s.parent_id:
            span["parentSpanId"] = s.parent_id
        otlp_spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _otlp_attributes(resource_attributes)},
                "scopeSpans": [{"scope": {"name": "agentlens", "version": __version__}, "spans": otlp_spans}],
            }
        ]
    }


class OTLPExporter:
    """Batches spans and POSTs them to an OTLP/HTTP collector (``/v1/traces``, JSON encoding).

    Works with the OpenTelemetry Collector, Jaeger (>=1.35, OTLP port 4318), Grafana Tempo, Arize
    Phoenix, Langfuse and any other OTLP/HTTP endpoint that accepts JSON. A batch is sent when it is
    full, when a root span (a whole run) finishes, and on ``shutdown()``.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        service_name: str = "agent",
        headers: dict[str, str] | None = None,
        batch_size: int = 64,
        timeout_s: float = 5.0,
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.endpoint = endpoint
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.resource_attributes = {"service.name": service_name, **(resource_attributes or {})}
        self._lock = threading.Lock()
        self._buffer: list[Span] = []
        self.sent_spans = 0
        self.dropped_spans = 0

    def export(self, spans: Sequence[Span]) -> None:
        batch: list[Span] = []
        with self._lock:
            self._buffer.extend(spans)
            if len(self._buffer) >= self.batch_size or any(s.parent_id is None for s in spans):
                batch, self._buffer = self._buffer, []
        if batch:
            self._send(batch)

    def flush(self) -> None:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if batch:
            self._send(batch)

    def _send(self, batch: list[Span]) -> None:
        body = json.dumps(spans_to_otlp(batch, self.resource_attributes)).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, headers=self.headers, method="POST")
        try:
            with _http.urlopen(req, timeout=self.timeout_s) as resp:
                ok = 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("OTLP export to %s failed: %s", self.endpoint, exc)
            ok = False
        with self._lock:
            if ok:
                self.sent_spans += len(batch)
            else:
                self.dropped_spans += len(batch)

    def shutdown(self) -> None:
        self.flush()
