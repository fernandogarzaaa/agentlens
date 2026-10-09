"""Tiny FastAPI trace viewer (``pip install "agentlens[server]"``; ``agentlens serve traces.jsonl``).

Routes: ``/`` run list, ``/traces/{id}`` span waterfall, ``/api/traces``, ``/api/traces/{id}``, ``/health``.
The store is re-read on each request, so a running eval shows up live.
"""

from __future__ import annotations

import html
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentlens.store import Trace, load_traces
from agentlens.tracer import (
    ATTR_INPUT_TOKENS,
    ATTR_OUTPUT_TOKENS,
    ATTR_TOOL_ARGS,
    ATTR_TOOL_RESULT,
    Span,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('the trace viewer needs the server extra: pip install "agentlens[server]"') from exc

__all__ = ["create_app"]

_CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:1.5rem auto;max-width:1200px;padding:0 1rem;color:#1f2328}
a{color:#0969da;text-decoration:none} table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #d1d9e0;padding:.35rem .5rem;text-align:left;vertical-align:top} th{background:#f6f8fa}
.bar{height:12px;border-radius:3px;background:#8c959f;position:relative} .k-llm{background:#8250df}.k-tool{background:#1a7f37}
.k-agent_step{background:#0969da}.k-run{background:#57606a} .err{color:#cf222e;font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:.4rem;border-radius:4px;margin:.2rem 0;max-height:12rem;overflow:auto}
"""


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"


def _short(value: Any, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + " ..."


def _ordered(trace: Trace) -> list[tuple[int, Span]]:
    out: list[tuple[int, Span]] = []

    def walk(span: Span, depth: int) -> None:
        out.append((depth, span))
        for child in trace.children(span):
            walk(child, depth + 1)

    ids = {s.span_id for s in trace.spans}
    for root in sorted(
        (s for s in trace.spans if s.parent_id is None or s.parent_id not in ids), key=lambda s: s.start_ns
    ):
        walk(root, 0)
    return out


def create_app(paths: Sequence[str | Path] | None = None) -> FastAPI:
    """Build the viewer app. ``paths`` defaults to ``$AGENTLENS_TRACES`` (comma-separated)."""
    if paths is None:
        env = os.environ.get("AGENTLENS_TRACES", "traces.jsonl")
        paths = [p for p in env.split(",") if p]
    store_paths = [Path(p) for p in paths]
    app = FastAPI(title="agentlens trace viewer")

    def traces() -> list[Trace]:
        existing = [p for p in store_paths if p.exists()]
        return load_traces(existing) if existing else []

    def find(trace_id: str) -> Trace:
        for t in traces():
            if t.trace_id == trace_id:
                return t
        raise HTTPException(status_code=404, detail="trace not found")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "stores": [str(p) for p in store_paths]}

    @app.get("/api/traces")
    def api_traces() -> list[dict[str, Any]]:
        return [
            {"trace_id": t.trace_id, "name": t.name, "case_id": t.case_id, **t.summary().to_dict()} for t in traces()
        ]

    @app.get("/api/traces/{trace_id}")
    def api_trace(trace_id: str) -> dict[str, Any]:
        t = find(trace_id)
        return {**t.to_dict(), "summary": t.summary().to_dict()}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        e = html.escape
        rows = []
        for t in reversed(traces()):
            s = t.summary()
            rows.append(
                f"<tr><td><a href='/traces/{e(t.trace_id)}'>{e(t.name)}</a></td><td>{s.steps}</td><td>{s.llm_calls}</td>"
                f"<td>{s.tool_calls}</td><td>{s.total_tokens}</td><td>${s.cost_usd:.6f}</td><td>{s.latency_s:.2f}</td>"
                f"<td>{'<span class=err>' + str(s.errors) + '</span>' if s.errors else 0}</td></tr>"
            )
        body = (
            f"<h1>agentlens traces</h1><p>{len(rows)} runs from {e(', '.join(map(str, store_paths)))}</p>"
            "<table><tr><th>Run</th><th>Steps</th><th>LLM calls</th><th>Tool calls</th><th>Tokens</th><th>Cost</th>"
            f"<th>Latency s</th><th>Errors</th></tr>{''.join(rows)}</table>"
        )
        return _page("agentlens traces", body)

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    def trace_page(trace_id: str) -> str:
        e = html.escape
        t = find(trace_id)
        start = min(s.start_ns for s in t.spans)
        end = max((s.end_ns or s.start_ns) for s in t.spans)
        total = max(end - start, 1)
        rows = []
        for depth, s in _ordered(t):
            left = 100 * (s.start_ns - start) / total
            width = max(0.5, 100 * ((s.end_ns or s.start_ns) - s.start_ns) / total)
            detail = []
            a = s.attributes
            if ATTR_TOOL_ARGS in a:
                detail.append(f"<pre>args: {e(_short(a[ATTR_TOOL_ARGS]))}</pre>")
            if ATTR_TOOL_RESULT in a:
                detail.append(f"<pre>result: {e(_short(a[ATTR_TOOL_RESULT]))}</pre>")
            if ATTR_INPUT_TOKENS in a:
                detail.append(
                    f"tokens {a.get(ATTR_INPUT_TOKENS, 0)} in / {a.get(ATTR_OUTPUT_TOKENS, 0)} out, ${float(a.get('agentlens.cost_usd', 0)):.6f}"
                )
            if "agentlens.output" in a and a["agentlens.output"]:
                detail.append(f"<pre>{e(_short(a['agentlens.output']))}</pre>")
            if s.status == "error":
                detail.append(f"<span class=err>{e(s.status_message)}</span>")
            rows.append(
                f"<tr><td style='padding-left:{0.5 + depth * 1.2}rem'>{e(s.name)}</td><td>{e(s.kind)}</td>"
                f"<td>{s.duration_s * 1000:.0f} ms</td><td style='width:35%'><div style='position:relative'>"
                f"<div class='bar k-{e(s.kind)}' style='margin-left:{left:.2f}%;width:{width:.2f}%'></div></div>"
                f"{''.join(detail)}</td></tr>"
            )
        summ = t.summary()
        body = (
            f"<p><a href='/'>&larr; all runs</a></p><h1>{e(t.name)}</h1>"
            f"<p>trace <code>{e(t.trace_id)}</code>: {summ.steps} steps, {summ.llm_calls} LLM calls, {summ.tool_calls} tool calls, "
            f"{summ.total_tokens} tokens, ${summ.cost_usd:.6f}, {summ.latency_s:.2f} s</p>"
            f"<pre>input: {e(_short(t.input, 2000))}</pre><pre>output: {e(_short(t.output, 2000))}</pre>"
            f"<table><tr><th>Span</th><th>Kind</th><th>Duration</th><th>Timeline / details</th></tr>{''.join(rows)}</table>"
        )
        return _page(t.name, body)

    return app
