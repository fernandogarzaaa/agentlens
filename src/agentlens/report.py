"""Markdown and self-contained HTML reports for eval results, diffs, gates and traces."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from typing import Any

from agentlens.gate import GateResult
from agentlens.harness import EvalResult
from agentlens.replay import DiffReport
from agentlens.store import Trace

__all__ = [
    "eval_markdown",
    "eval_html",
    "diff_markdown",
    "diff_html",
    "gate_markdown",
    "traces_markdown",
    "SUMMARY_ROWS",
]

# (key, label, format)
SUMMARY_ROWS: list[tuple[str, str, str]] = [
    ("n_cases", "Cases", "int"),
    ("task_success", "Task success", "pct"),
    ("tool_precision", "Tool precision", "f3"),
    ("tool_recall", "Tool recall", "f3"),
    ("tool_f1", "Tool F1 (macro)", "f3"),
    ("tool_f1_micro", "Tool F1 (micro)", "f3"),
    ("args_exact", "Args exact match", "pct"),
    ("args_partial", "Args partial match", "pct"),
    ("trajectory_match", "Trajectory match", "pct"),
    ("constraint_pass", "Constraints pass", "pct"),
    ("answer_pass", "Final-answer checks pass", "pct"),
    ("judge_score_mean", "Judge score (mean)", "f2"),
    ("error_rate", "Error rate", "pct"),
    ("avg_steps", "Steps / task", "f2"),
    ("avg_tokens", "Tokens / task", "f0"),
    ("cost_per_task_usd", "$ / task", "usd"),
    ("total_cost_usd", "Total $", "usd"),
    ("latency_p50_s", "Latency p50 (s)", "f2"),
    ("latency_p95_s", "Latency p95 (s)", "f2"),
]


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if kind == "pct":
        return f"{100 * float(value):.1f}%"
    if kind == "int":
        return str(int(value))
    if kind == "usd":
        return f"${float(value):.6f}"
    if kind.startswith("f"):
        return f"{float(value):.{int(kind[1:])}f}"
    return str(value)


def _delta(key: str, kind: str, delta: float) -> str:
    if kind == "pct":
        return f"{100 * delta:+.1f} pp"
    if kind == "usd":
        return f"{delta:+.6f}"
    return f"{delta:+.3f}"


def _calls(calls: Sequence[dict[str, Any]]) -> str:
    return (
        ", ".join(f"{c['name']}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})" for c in calls) or "(none)"
    )


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def eval_markdown(result: EvalResult, max_failures: int = 25) -> str:
    s = result.summary
    lines = [
        f"# agentlens eval: {result.suite}",
        "",
        f"Agent **{result.agent}**, model `{result.model or '-'}`, {s.get('n_cases', 0)} cases, {result.created}.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label, kind in SUMMARY_ROWS:
        if key in s:
            lines.append(f"| {label} | {_fmt(s[key], kind)} |")
    if not s.get("cost_known", True):
        lines += ["", "> Some models are missing from the pricing table: their cost is counted as unknown ($0 here)."]
    failed = [c for c in result.cases if not c.success]
    lines += ["", f"## Failed cases ({len(failed)}/{len(result.cases)})", ""]
    if not failed:
        lines.append("None.")
    else:
        lines += ["| Case | Why | Expected | Actual |", "|---|---|---|---|"]
        for c in failed[:max_failures]:
            why = []
            if c.error:
                why.append(f"error: {c.error}")
            if not c.trajectory_match:
                why.append(f"trajectory ({c.trajectory_mode})")
            if c.args_exact < 1:
                why.append(f"args {c.args_exact:.2f}")
            why += c.violations + c.answer_failures
            lines.append(
                f"| {_md_escape(c.case_id)} | {_md_escape('; '.join(why))} | "
                f"{_md_escape(_calls(c.expected_tool_calls))[:300]} | {_md_escape(_calls(c.tool_calls))[:300]} |"
            )
        if len(failed) > max_failures:
            lines.append(f"\n...and {len(failed) - max_failures} more (see the JSON results).")
    return "\n".join(lines) + "\n"


def diff_markdown(diff: DiffReport) -> str:
    deltas = diff.deltas()
    counts = diff.to_dict()["counts"]
    lines = [
        "# agentlens diff",
        "",
        f"Base **{diff.base_label}** vs candidate **{diff.new_label}**: "
        f"{counts['regression']} regressions, {counts['improvement']} improvements, "
        f"{counts['changed']} changed, {counts['unchanged']} unchanged"
        + (f", {counts['added']} added, {counts['removed']} removed" if counts["added"] or counts["removed"] else "")
        + ".",
        "",
        "| Metric | Base | Candidate | Delta |",
        "|---|---|---|---|",
    ]
    for key, label, kind in SUMMARY_ROWS:
        if key in diff.base_summary and key in diff.new_summary:
            d = deltas.get(key)
            lines.append(
                f"| {label} | {_fmt(diff.base_summary[key], kind)} | {_fmt(diff.new_summary[key], kind)} | "
                f"{_delta(key, kind, d) if d is not None else '-'} |"
            )
    for status in ("regression", "improvement", "changed"):
        rows = [c for c in diff.cases if c.status == status]
        if not rows:
            continue
        lines += ["", f"## {status.capitalize()}s ({len(rows)})", "", "| Case | Details |", "|---|---|"]
        for c in rows:
            lines.append(f"| {_md_escape(c.case_id)} | {_md_escape('; '.join(c.reasons))} |")
    return "\n".join(lines) + "\n"


def gate_markdown(result: GateResult) -> str:
    lines = [f"# agentlens gate: {'PASSED' if result.passed else 'FAILED'}", "", "| Check | Result |", "|---|---|"]
    for c in result.checks:
        lines.append(f"| {'pass' if c.passed else '**FAIL**'} | {_md_escape(c.message)} |")
    return "\n".join(lines) + "\n"


def traces_markdown(traces: Sequence[Trace]) -> str:
    lines = [
        f"# agentlens traces ({len(traces)})",
        "",
        "| Trace | Name | Steps | LLM calls | Tool calls | Tokens | $ | Latency (s) | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in traces:
        s = t.summary()
        lines.append(
            f"| `{t.trace_id[:12]}` | {_md_escape(t.name)} | {s.steps} | {s.llm_calls} | {s.tool_calls} | "
            f"{s.total_tokens} | {s.cost_usd:.6f}{'' if s.cost_known else '?'} | {s.latency_s:.2f} | {s.errors} |"
        )
    return "\n".join(lines) + "\n"


# ---- HTML -------------------------------------------------------------------------------
_CSS = """
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1f2328}
h1{font-size:1.6rem;margin-bottom:.2rem} h2{font-size:1.2rem;margin-top:2rem}
.meta{color:#59636e;margin-bottom:1.5rem}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:.75rem}
.card{border:1px solid #d1d9e0;border-radius:8px;padding:.75rem 1rem}
.card .v{font-size:1.4rem;font-weight:600} .card .k{color:#59636e;font-size:.85rem}
table{border-collapse:collapse;width:100%;margin-top:.5rem} th,td{border-bottom:1px solid #d1d9e0;padding:.4rem .5rem;text-align:left;vertical-align:top}
th{background:#f6f8fa} code{font-size:.85em;background:#f6f8fa;padding:.1rem .3rem;border-radius:4px;word-break:break-word}
.pass{color:#1a7f37;font-weight:600} .fail{color:#cf222e;font-weight:600} .chg{color:#9a6700;font-weight:600}
"""


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}"
        "<p class=meta>Generated by agentlens.</p></body></html>\n"
    )


def eval_html(result: EvalResult) -> str:
    s = result.summary
    e = html.escape
    cards = "".join(
        f"<div class=card><div class=v>{e(_fmt(s[k], kind))}</div><div class=k>{e(label)}</div></div>"
        for k, label, kind in SUMMARY_ROWS
        if k in s
    )
    rows = []
    for c in result.cases:
        status = "<span class=pass>pass</span>" if c.success else "<span class=fail>fail</span>"
        why = "; ".join(([f"error: {c.error}"] if c.error else []) + c.violations + c.answer_failures)
        rows.append(
            f"<tr><td>{e(c.case_id)}</td><td>{status}</td><td>{c.tool_f1:.2f}</td><td>{c.args_exact:.2f}</td>"
            f"<td>{'yes' if c.trajectory_match else 'no'}</td><td>{c.steps}</td><td>{c.total_tokens}</td>"
            f"<td>{c.latency_s:.2f}</td><td><code>{e(_calls(c.expected_tool_calls))}</code></td>"
            f"<td><code>{e(_calls(c.tool_calls))}</code></td><td>{e(why)}</td></tr>"
        )
    body = (
        f"<h1>agentlens eval: {e(result.suite)}</h1>"
        f"<p class=meta>Agent <b>{e(result.agent)}</b>, model <code>{e(result.model or '-')}</code>, {e(result.created)}</p>"
        f"<div class=cards>{cards}</div><h2>Cases</h2><table><tr><th>Case</th><th>Result</th><th>Tool F1</th>"
        "<th>Args exact</th><th>Trajectory</th><th>Steps</th><th>Tokens</th><th>Latency s</th><th>Expected</th>"
        f"<th>Actual</th><th>Notes</th></tr>{''.join(rows)}</table>"
    )
    return _page(f"agentlens eval: {result.suite}", body)


def diff_html(diff: DiffReport) -> str:
    e = html.escape
    deltas = diff.deltas()
    metric_rows = "".join(
        f"<tr><td>{e(label)}</td><td>{e(_fmt(diff.base_summary[k], kind))}</td><td>{e(_fmt(diff.new_summary[k], kind))}</td>"
        f"<td>{e(_delta(k, kind, deltas[k])) if k in deltas else '-'}</td></tr>"
        for k, label, kind in SUMMARY_ROWS
        if k in diff.base_summary and k in diff.new_summary
    )
    cls = {"regression": "fail", "improvement": "pass", "changed": "chg"}
    case_rows = "".join(
        f"<tr><td>{e(c.case_id)}</td><td class={cls.get(c.status, '')}>{e(c.status)}</td><td>{e('; '.join(c.reasons))}</td></tr>"
        for c in sorted(
            diff.cases,
            key=lambda c: ("regression", "improvement", "changed", "added", "removed", "unchanged").index(c.status),
        )
    )
    body = (
        f"<h1>agentlens diff</h1><p class=meta>Base <b>{e(diff.base_label)}</b> vs candidate <b>{e(diff.new_label)}</b>: "
        f"{len(diff.regressions)} regressions, {len(diff.improvements)} improvements</p>"
        f"<table><tr><th>Metric</th><th>Base</th><th>Candidate</th><th>Delta</th></tr>{metric_rows}</table>"
        f"<h2>Cases</h2><table><tr><th>Case</th><th>Status</th><th>Details</th></tr>{case_rows}</table>"
    )
    return _page("agentlens diff", body)
