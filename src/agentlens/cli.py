"""``agentlens`` command line: run | eval | report | gate | replay | judge | serve | bfcl."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentlens._version import __version__
from agentlens.agent import Agent, ToolAgent, Toolset, load_toolset
from agentlens.cases import SuiteError, load_suite
from agentlens.cost import Pricing, PricingError
from agentlens.exporters import JSONLExporter, OTLPExporter, SQLiteExporter
from agentlens.gate import GateConfigError, evaluate_gate, load_gate_config
from agentlens.llm import ChatClient, MockChatClient, OpenAICompatClient
from agentlens.store import SQLITE_SUFFIXES
from agentlens.tracer import SpanExporter, Tracer

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2


class CLIError(Exception):
    """Configuration / usage problem: exit code 2."""


# ---- shared option groups ----------------------------------------------------------------
def _agent_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("agent")
    g.add_argument("--agent", default=None, help="openai (default when --model is set), mock, or module:attr")
    g.add_argument("--model", default=None, help="model name sent to the OpenAI-compatible endpoint")
    g.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    g.add_argument("--mock-rules", default=None, help="YAML rules for the mock model (--agent mock)")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    g.add_argument("--label", default=None, help="name for this agent version in reports")


def _trace_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("tracing")
    g.add_argument("--trace", default=None, help="write spans to this .jsonl or .db/.sqlite file")
    g.add_argument(
        "--otlp-endpoint",
        default=None,
        help="also export to an OTLP/HTTP collector, e.g. http://localhost:4318/v1/traces",
    )
    g.add_argument("--pricing", default=None, help="TOML pricing table merged over the bundled one")


def _exporters(args: argparse.Namespace, service: str) -> list[SpanExporter]:
    out: list[SpanExporter] = []
    if args.trace:
        out.append(
            SQLiteExporter(args.trace)
            if Path(args.trace).suffix.lower() in SQLITE_SUFFIXES
            else JSONLExporter(args.trace)
        )
    if args.otlp_endpoint:
        out.append(OTLPExporter(args.otlp_endpoint, service_name=service))
    return out


def _pricing(args: argparse.Namespace) -> Pricing:
    base = Pricing.default()
    if getattr(args, "pricing", None):
        try:
            return base.merged(Pricing.from_toml(args.pricing))
        except (OSError, PricingError) as exc:
            raise CLIError(f"pricing: {exc}") from None
    return base


def _client(args: argparse.Namespace, base_url: str | None = None, mock_rules: str | None = None) -> ChatClient:
    kind = args.agent or ("mock" if args.mock_rules and not args.model else "openai")
    if kind == "mock":
        rules = mock_rules or args.mock_rules
        if not rules:
            raise CLIError("--agent mock needs --mock-rules RULES.yaml")
        return MockChatClient.from_yaml(rules)
    return OpenAICompatClient(
        base_url or args.base_url, api_key=os.environ.get(args.api_key_env), timeout_s=args.timeout
    )


def _build_agent(args: argparse.Namespace) -> tuple[Agent, str, str]:
    """Returns (agent, label, model)."""
    kind = args.agent or ("mock" if args.mock_rules and not args.model else "openai")
    if kind not in ("openai", "mock"):
        if ":" not in kind:
            raise CLIError(f"--agent must be openai, mock or module:attr, got {kind!r}")
        mod_name, attr = kind.rsplit(":", 1)
        sys.path.insert(0, os.getcwd())
        try:
            obj = getattr(importlib.import_module(mod_name), attr)
        except (ImportError, AttributeError) as exc:
            raise CLIError(f"cannot load agent {kind}: {exc}") from None
        if inspect.isclass(obj) or (callable(obj) and not hasattr(obj, "run")):
            obj = obj()
        if not hasattr(obj, "run"):
            raise CLIError(f"{kind} has no run() method")
        return obj, args.label or kind, args.model or ""
    if kind == "openai" and not args.model:
        raise CLIError("--model is required for the openai agent (or use --agent mock --mock-rules FILE)")
    model = args.model or "mock-agent"
    agent = ToolAgent(_client(args), model=model, temperature=args.temperature, max_tokens=args.max_tokens)
    return agent, args.label or f"{kind}:{model}", model


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


# ---- commands ------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    agent, label, model = _build_agent(args)
    toolset = load_toolset(args.toolset, Path.cwd()) if args.toolset else Toolset()
    exporters = _exporters(args, label)
    tracer = Tracer(service_name=label, exporters=exporters, pricing=_pricing(args))
    with tracer.run("agent.run", input=args.prompt) as root:
        result = agent.run(args.prompt, toolset, system_prompt=args.system, max_steps=args.max_steps, tracer=tracer)
        root.set_output(result.final_answer)
    tracer.shutdown()
    summary = tracer.summary(root.trace_id)
    print(
        result.final_answer
        if result.final_answer is not None
        else f"(no answer: {result.error or 'max steps reached'})"
    )
    if summary is not None:
        s = summary
        print(
            f"\n[{s.steps} steps, {s.llm_calls} LLM calls, {s.tool_calls} tool calls, {s.total_tokens} tokens, "
            f"${s.cost_usd:.6f}{'' if s.cost_known else ' (unknown model price)'}, {s.latency_s:.2f}s]",
            file=sys.stderr,
        )
    return EXIT_OK if result.error is None else EXIT_FAIL


def _judge_from_args(args: argparse.Namespace) -> Any:
    if not args.judge_model:
        return None
    from agentlens.judge import Rubric, RubricJudge

    rubric = Rubric.load(args.rubric) if args.rubric else Rubric()
    client = OpenAICompatClient(args.judge_base_url or args.base_url, api_key=os.environ.get(args.api_key_env))
    return RubricJudge(client, args.judge_model, rubric)


def cmd_eval(args: argparse.Namespace) -> int:
    from agentlens.harness import run_eval
    from agentlens.report import eval_html, eval_markdown

    try:
        suite = load_suite(args.suite).filter(tags=args.tags, ids=args.ids, limit=args.limit)
    except SuiteError as exc:
        raise CLIError(str(exc)) from None
    if args.from_traces:
        return _eval_from_traces(args, suite)
    agent, label, model = _build_agent(args)
    tracer = Tracer(service_name=label, exporters=_exporters(args, label), pricing=_pricing(args))
    done = [0]

    def progress(r: Any) -> None:
        done[0] += 1
        if not args.quiet:
            mark = "PASS" if r.success else "FAIL"
            print(f"[{done[0]}/{len(suite.cases)}] {mark} {r.case_id} ({r.latency_s:.1f}s)", file=sys.stderr)

    result = run_eval(
        suite,
        agent,
        agent_label=label,
        model=model,
        tracer=tracer,
        judge=_judge_from_args(args),
        concurrency=args.concurrency,
        on_case=progress,
    )
    tracer.shutdown()
    if args.out:
        result.save(args.out)
    if args.html:
        _write(args.html, eval_html(result))
    md = eval_markdown(result)
    if args.report:
        _write(args.report, md)
    if not args.quiet or not args.out:
        sys.stdout.write(md)
    return EXIT_OK


def _eval_from_traces(args: argparse.Namespace, suite: Any) -> int:
    """Score recorded runs against the suite without calling any model (re-scoring after a suite fix)."""
    from agentlens.replay import baseline_from_traces
    from agentlens.report import eval_html, eval_markdown
    from agentlens.store import load_traces

    result = baseline_from_traces(load_traces(args.from_traces), suite)
    if not result.cases:
        raise CLIError("no recorded runs match the suite's case ids")
    result.agent = args.label or "recorded"
    result.model = args.model or ""
    result.metadata["scored_from_traces"] = [str(p) for p in args.from_traces]
    if args.out:
        result.save(args.out)
    if args.html:
        _write(args.html, eval_html(result))
    md = eval_markdown(result)
    if args.report:
        _write(args.report, md)
    if not args.quiet or not args.out:
        sys.stdout.write(md)
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from agentlens.harness import EvalResult
    from agentlens.replay import diff_results
    from agentlens.report import diff_html, diff_markdown, eval_html, eval_markdown, traces_markdown
    from agentlens.store import load_traces

    if args.traces:
        _write(args.output, traces_markdown(load_traces(args.traces)))
        return EXIT_OK
    if not args.results:
        raise CLIError("give a results.json (or --traces FILE)")
    try:
        result = EvalResult.load(args.results)
        base = EvalResult.load(args.compare) if args.compare else None
    except (OSError, ValueError) as exc:
        raise CLIError(f"cannot read results: {exc}") from None
    if base is not None:
        diff = diff_results(base, result)
        _write(args.output, diff_html(diff) if args.format == "html" else diff_markdown(diff))
        return EXIT_FAIL if (args.fail_on_regression and diff.regressions) else EXIT_OK
    _write(args.output, eval_html(result) if args.format == "html" else eval_markdown(result))
    return EXIT_OK


def cmd_gate(args: argparse.Namespace) -> int:
    from agentlens.report import gate_markdown

    try:
        config = load_gate_config(args.config)
        summary = _load_summary(args.results)
        baseline_path = args.baseline or (str(config.baseline_path) if config.baseline_path else None)
        baseline = _load_summary(baseline_path) if baseline_path and config.needs_baseline else None
        result = evaluate_gate(summary, config, baseline)
    except GateConfigError as exc:
        raise CLIError(str(exc)) from None
    md = gate_markdown(result)
    sys.stdout.write(md)
    if args.report:
        _write(args.report, md)
    if args.json:
        _write(args.json, json.dumps(result.to_dict(), indent=2) + "\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    return result.exit_code


def _load_summary(path: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateConfigError(f"cannot read results {path}: {exc}") from None
    if not isinstance(data, dict):
        raise GateConfigError(f"cannot read results {path}: expected a JSON object")
    if "summary" in data and isinstance(data["summary"], dict):
        return dict(data["summary"])
    if "cases" in data:
        from agentlens.harness import EvalResult

        try:
            return EvalResult.from_dict(data).summary
        except ValueError as exc:
            raise GateConfigError(f"cannot read results {path}: {exc}") from None
    return dict(data)


def cmd_replay(args: argparse.Namespace) -> int:
    from agentlens.replay import replay
    from agentlens.report import diff_html, diff_markdown
    from agentlens.store import load_traces

    traces = load_traces(args.traces)
    suite = None
    if args.suite:
        try:
            suite = load_suite(args.suite)
        except SuiteError as exc:
            raise CLIError(str(exc)) from None
    agent, label, model = _build_agent(args)
    try:
        baseline, candidate, diff = replay(
            traces, agent, agent_label=label, model=model, suite=suite, exporters=_exporters(args, label)
        )
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    if args.out:
        _write(args.out, json.dumps(diff.to_dict(), indent=2, default=str) + "\n")
    if args.candidate_out:
        candidate.save(args.candidate_out)
    if args.html:
        _write(args.html, diff_html(diff))
    md = diff_markdown(diff)
    if args.report:
        _write(args.report, md)
    sys.stdout.write(md)
    return EXIT_FAIL if (args.fail_on_regression and diff.regressions) else EXIT_OK


def cmd_judge(args: argparse.Namespace) -> int:
    from agentlens.calibration import calibrate
    from agentlens.judge import PairwiseJudge, Rubric, RubricJudge

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise CLIError("no labelled rows")
    rubric = Rubric.load(args.rubric) if args.rubric else Rubric()
    client: ChatClient
    if args.agent == "mock" or (args.mock_rules and not args.model):
        client = _client(args)
    else:
        if not args.model:
            raise CLIError("--model is required")
        client = OpenAICompatClient(args.base_url, api_key=os.environ.get(args.api_key_env), timeout_s=args.timeout)
    model = args.model or "mock-judge"
    predictions: list[Any] = []
    human: list[Any] = []
    failed_calls = consecutive = 0
    for i, row in enumerate(rows, 1):
        if args.mode == "pairwise":
            verdict = PairwiseJudge(client, model, rubric, swap=args.swap, max_tokens=args.max_tokens).compare(
                row["question"], row["answer_a"], row["answer_b"]
            )
            predictions.append(verdict.winner)
            failed = any(r.startswith("judge call failed") for r in verdict.raw)
        else:
            v = RubricJudge(client, model, rubric, max_tokens=args.max_tokens).score(
                row["question"], row["answer"], row.get("reference")
            )
            predictions.append(None if v.passed is None else ("pass" if v.passed else "fail"))
            failed = v.reasoning.startswith("judge call failed")
        human.append(row["label"])
        failed_calls += int(failed)
        consecutive = consecutive + 1 if failed else 0
        if consecutive >= args.max_consecutive_failures:
            print(
                f"agentlens: {consecutive} judge calls in a row failed; stopping (is the endpoint up?)", file=sys.stderr
            )
            return EXIT_FAIL
        if not args.quiet:
            print(f"[{i}/{len(rows)}] judge={predictions[-1]} human={row['label']}", file=sys.stderr)
    labels = ["A", "B", "tie"] if args.mode == "pairwise" else ["pass", "fail"]
    report = calibrate(predictions, human, labels=labels)
    report.metadata = {
        "model": model,
        "mode": args.mode,
        "swap": args.swap,
        "data": str(args.data),
        "rubric": rubric.name,
        "failed_calls": failed_calls,
    }
    out = json.dumps(report.to_dict(), indent=2) + "\n"
    _write(args.out, out) if args.out else None
    print(f"n={report.n} accuracy={report.accuracy:.3f} kappa={report.kappa:.3f} abstentions={report.abstentions}")
    if args.predictions:
        _write(
            args.predictions, "".join(json.dumps({"judge": p, "human": h}) + "\n" for p, h in zip(predictions, human))
        )
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from agentlens.server import create_app
    except ImportError:
        raise CLIError('the trace viewer needs: pip install "agentlens[server]"') from None
    uvicorn.run(create_app(args.traces), host=args.host, port=args.port, log_level="warning")
    return EXIT_OK


def cmd_bfcl(args: argparse.Namespace) -> int:
    from agentlens.benchmarks.bfcl import build_suite

    path = build_suite(args.category, args.dest, limit=args.limit)
    print(path)
    return EXIT_OK


# ---- parser -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentlens", description="Tracing and evaluation for tool-calling LLM agents.")
    p.add_argument("--version", action="version", version=f"agentlens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the agent once on a prompt, traced")
    r.add_argument("prompt")
    r.add_argument("--toolset", default=None, help="tools as file.py:NAME or module:NAME")
    r.add_argument("--system", default=None, help="system prompt")
    r.add_argument("--max-steps", type=int, default=8)
    _agent_args(r)
    _trace_args(r)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="evaluate an agent on a YAML/JSON suite")
    e.add_argument("suite")
    e.add_argument("--out", default=None, help="write results JSON here")
    e.add_argument("--report", default=None, help="write the Markdown report here")
    e.add_argument("--html", default=None, help="write an HTML report here")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--tags", nargs="*", default=None)
    e.add_argument("--ids", nargs="*", default=None, help="only these case ids")
    e.add_argument(
        "--from-traces",
        nargs="+",
        default=None,
        help="score recorded runs (trace stores) against the suite instead of running the agent",
    )
    e.add_argument("--concurrency", type=int, default=1)
    e.add_argument("--judge-model", default=None, help="LLM-as-judge model for final_answer.judge cases")
    e.add_argument("--judge-base-url", default=None)
    e.add_argument("--rubric", default=None, help="rubric YAML for the judge")
    e.add_argument("--quiet", action="store_true")
    _agent_args(e)
    _trace_args(e)
    e.set_defaults(func=cmd_eval)

    rp = sub.add_parser("report", help="render results (or a comparison, or a trace store) as Markdown/HTML")
    rp.add_argument("results", nargs="?")
    rp.add_argument("--compare", default=None, help="baseline results.json to diff against")
    rp.add_argument("--traces", nargs="*", default=None, help="summarise trace stores instead")
    rp.add_argument("--format", choices=("md", "html"), default="md")
    rp.add_argument("-o", "--output", default=None)
    rp.add_argument("--fail-on-regression", action="store_true")
    rp.set_defaults(func=cmd_report)

    g = sub.add_parser("gate", help="CI regression gate: exit 1 when metrics breach gate.toml")
    g.add_argument("results")
    g.add_argument("--config", default="gate.toml")
    g.add_argument("--baseline", default=None, help="baseline results.json (overrides [baseline].path)")
    g.add_argument("--report", default=None)
    g.add_argument("--json", default=None)
    g.set_defaults(func=cmd_gate)

    rl = sub.add_parser("replay", help="re-run recorded traces against a new agent version and diff")
    rl.add_argument("traces", nargs="+")
    rl.add_argument("--suite", default=None, help="score against this suite (live tools) instead of the recording")
    rl.add_argument("--out", default=None, help="diff JSON")
    rl.add_argument("--candidate-out", default=None, help="candidate results JSON")
    rl.add_argument("--report", default=None)
    rl.add_argument("--html", default=None)
    rl.add_argument("--fail-on-regression", action="store_true")
    _agent_args(rl)
    _trace_args(rl)
    rl.set_defaults(func=cmd_replay)

    j = sub.add_parser("judge", help="calibrate an LLM judge against human labels (JSONL)")
    j.add_argument(
        "data", help="JSONL: {question, answer, label: pass|fail} or {question, answer_a, answer_b, label: A|B|tie}"
    )
    j.add_argument("--mode", choices=("pointwise", "pairwise"), default="pairwise")
    j.add_argument("--swap", action="store_true", help="pairwise: judge both orders (MT-Bench consistency rule)")
    j.add_argument("--rubric", default=None)
    j.add_argument("--limit", type=int, default=None)
    j.add_argument("--out", default=None)
    j.add_argument("--predictions", default=None)
    j.add_argument("--quiet", action="store_true")
    j.add_argument(
        "--max-consecutive-failures", type=int, default=5, help="abort when the judge endpoint keeps failing"
    )
    _agent_args(j)
    j.set_defaults(func=cmd_judge)

    s = sub.add_parser("serve", help="trace viewer (needs the server extra)")
    s.add_argument("traces", nargs="*", default=["traces.jsonl"])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    b = sub.add_parser("bfcl", help="download a BFCL v4 category and convert it to a suite")
    b.add_argument("category")
    b.add_argument("--dest", default="data/bfcl")
    b.add_argument("--limit", type=int, default=None)
    b.set_defaults(func=cmd_bfcl)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CLIError as exc:
        print(f"agentlens: error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except FileNotFoundError as exc:
        print(f"agentlens: error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
