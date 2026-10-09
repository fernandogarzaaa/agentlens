import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentlens.benchmarks.bfcl import convert, convert_schema
from agentlens.cli import main
from agentlens.metrics import ABSENT

EX = Path("examples/travel_desk")
V1 = ["--agent", "mock", "--mock-rules", str(EX / "mock_v1.yaml"), "--model", "mock-v1"]
V2 = ["--agent", "mock", "--mock-rules", str(EX / "mock_v2.yaml"), "--model", "mock-v2"]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    assert (
        main(
            [
                "eval",
                str(EX / "suite.yaml"),
                *V1,
                "--trace",
                str(d / "v1.jsonl"),
                "--out",
                str(d / "v1.json"),
                "--quiet",
            ]
        )
        == 0
    )
    return d


def test_eval_writes_results_reports_and_traces(recorded, tmp_path, capsys):
    rc = main(
        [
            "eval",
            str(EX / "suite.yaml"),
            *V1,
            "--limit",
            "3",
            "--out",
            str(tmp_path / "r.json"),
            "--report",
            str(tmp_path / "r.md"),
            "--html",
            str(tmp_path / "r.html"),
            "--trace",
            str(tmp_path / "t.db"),
        ]
    )
    assert rc == 0
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["n_cases"] == 3
    assert "Task success" in (tmp_path / "r.md").read_text()
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "t.db").exists()
    out = capsys.readouterr()
    assert "[3/3]" in out.err and "agentlens eval" in out.out


def test_gate_exit_codes(recorded, tmp_path, capsys):
    gate = EX / "gate.toml"
    assert (
        main(
            [
                "gate",
                str(recorded / "v1.json"),
                "--config",
                str(gate),
                "--baseline",
                str(recorded / "v1.json"),
                "--json",
                str(tmp_path / "g.json"),
                "--report",
                str(tmp_path / "g.md"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "g.json").read_text())["passed"] is True
    main(["eval", str(EX / "suite.yaml"), *V2, "--out", str(tmp_path / "v2.json"), "--quiet"])
    assert (
        main(["gate", str(tmp_path / "v2.json"), "--config", str(gate), "--baseline", str(recorded / "v1.json")]) == 1
    )
    assert "FAILED" in capsys.readouterr().out
    bad = tmp_path / "bad.toml"
    bad.write_text("[thresholds]\nnot_a_metric = { min = 1 }\n")
    assert main(["gate", str(recorded / "v1.json"), "--config", str(bad)]) == 2
    assert main(["gate", str(tmp_path / "missing.json"), "--config", str(gate)]) == 2


def test_gate_uses_config_baseline_and_step_summary(recorded, tmp_path, monkeypatch):
    cfg = tmp_path / "gate.toml"
    (tmp_path / "base.json").write_text((recorded / "v1.json").read_text())
    cfg.write_text(
        '[thresholds]\ntask_success = { min = 0.5 }\n[baseline]\npath = "base.json"\nmax_drop = { task_success = 0.0 }\n'
    )
    summary_file = tmp_path / "step.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    assert main(["gate", str(recorded / "v1.json"), "--config", str(cfg)]) == 0
    assert "PASSED" in summary_file.read_text()
    # a bare summary dict works too
    (tmp_path / "s.json").write_text(json.dumps({"task_success": 1.0}))
    assert main(["gate", str(tmp_path / "s.json"), "--config", str(cfg)]) == 0


def test_replay_cli(recorded, tmp_path, capsys):
    rc = main(
        [
            "replay",
            str(recorded / "v1.jsonl"),
            *V2,
            "--out",
            str(tmp_path / "d.json"),
            "--html",
            str(tmp_path / "d.html"),
            "--report",
            str(tmp_path / "d.md"),
            "--candidate-out",
            str(tmp_path / "c.json"),
            "--fail-on-regression",
        ]
    )
    assert rc == 1
    assert json.loads((tmp_path / "d.json").read_text())["counts"]["regression"] == 3
    assert "3 regressions" in capsys.readouterr().out
    assert (
        main(["replay", str(recorded / "v1.jsonl"), *V1, "--suite", str(EX / "suite.yaml"), "--fail-on-regression"])
        == 0
    )
    bad_suite = tmp_path / "bad.yaml"
    bad_suite.write_text("cases: []\n")
    assert main(["replay", str(recorded / "v1.jsonl"), *V1, "--suite", str(bad_suite)]) == 2


def test_report_cli(recorded, tmp_path, capsys):
    assert main(["report", str(recorded / "v1.json")]) == 0
    assert "Task success" in capsys.readouterr().out
    assert main(["report", str(recorded / "v1.json"), "--format", "html", "-o", str(tmp_path / "r.html")]) == 0
    assert (
        main(["report", str(recorded / "v1.json"), "--compare", str(recorded / "v1.json"), "--fail-on-regression"]) == 0
    )
    assert main(["report", "--traces", str(recorded / "v1.jsonl"), "-o", str(tmp_path / "t.md")]) == 0
    assert "case weather-paris" in (tmp_path / "t.md").read_text()
    assert main(["report"]) == 2
    assert main(["report", str(tmp_path / "nope.json")]) == 2


def test_run_cli(tmp_path, capsys):
    rc = main(
        [
            "run",
            "What's the weather in Paris right now?",
            *V1,
            "--toolset",
            str(EX / "tools.py") + ":TOOLSET",
            "--trace",
            str(tmp_path / "t.jsonl"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    assert "18" in out.out and "1 tool calls" in out.err


def test_cli_config_errors(tmp_path, capsys):
    assert main(["eval", str(EX / "suite.yaml")]) == 2  # no --model
    assert main(["eval", str(EX / "suite.yaml"), "--agent", "mock"]) == 2  # no rules
    assert main(["eval", str(tmp_path / "missing.yaml"), *V1]) == 2
    assert main(["eval", str(EX / "suite.yaml"), "--agent", "nocolon"]) == 2
    assert main(["eval", str(EX / "suite.yaml"), "--agent", "nosuchmod:thing"]) == 2
    bad = tmp_path / "p.toml"
    bad.write_text("[models\n")
    assert main(["eval", str(EX / "suite.yaml"), *V1, "--pricing", str(bad)]) == 2
    assert "error" in capsys.readouterr().err


def test_custom_agent_and_pricing(tmp_path, monkeypatch):
    (tmp_path / "myagent.py").write_text(
        "from agentlens import ToolAgent, MockChatClient\n"
        "class Agent(ToolAgent):\n"
        "    def __init__(self):\n"
        f"        super().__init__(MockChatClient.from_yaml({str(EX.resolve() / 'mock_v1.yaml')!r}), model='custom-1')\n"
    )
    (tmp_path / "prices.toml").write_text('[models."custom-*"]\ninput_per_mtok = 1000\noutput_per_mtok = 0\n')
    monkeypatch.chdir(tmp_path)
    suite = str(Path(__file__).resolve().parents[1] / EX / "suite.yaml")
    assert (
        main(
            [
                "eval",
                suite,
                "--agent",
                "myagent:Agent",
                "--limit",
                "2",
                "--pricing",
                "prices.toml",
                "--out",
                "r.json",
                "--quiet",
            ]
        )
        == 0
    )
    s = json.loads(Path("r.json").read_text())["summary"]
    assert s["cost_known"] and s["cost_per_task_usd"] > 0.5


def test_judge_cli_with_mock(tmp_path):
    rules = tmp_path / "judge.yaml"
    rules.write_text('rules:\n  - match: "."\n    steps:\n      - content: "A is better. [[A]]"\n')
    data = tmp_path / "pairs.jsonl"
    data.write_text(
        "".join(
            json.dumps({"question": "q", "answer_a": "a", "answer_b": "b", "label": lab}) + "\n"
            for lab in ["A", "A", "B"]
        )
    )
    out = tmp_path / "cal.json"
    assert (
        main(
            [
                "judge",
                str(data),
                "--agent",
                "mock",
                "--mock-rules",
                str(rules),
                "--out",
                str(out),
                "--predictions",
                str(tmp_path / "p.jsonl"),
                "--quiet",
            ]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert rep["n"] == 3 and rep["accuracy"] == pytest.approx(2 / 3)
    point = tmp_path / "point.jsonl"
    point.write_text(json.dumps({"question": "q", "answer": "a", "label": "pass"}) + "\n")
    rules.write_text('rules:\n  - match: "."\n    steps:\n      - content: \'{"score": 5}\'\n')
    assert (
        main(["judge", str(point), "--mode", "pointwise", "--agent", "mock", "--mock-rules", str(rules), "--quiet"])
        == 0
    )
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert main(["judge", str(empty), "--agent", "mock", "--mock-rules", str(rules)]) == 2


def test_module_entry_point_and_version():
    r = subprocess.run([sys.executable, "-m", "agentlens", "--version"], capture_output=True, text=True, check=True)
    assert r.stdout.startswith("agentlens ")


# ---- server ---------------------------------------------------------------------------------
def test_trace_viewer(recorded, monkeypatch):
    from fastapi.testclient import TestClient

    from agentlens.server import create_app

    client = TestClient(create_app([recorded / "v1.jsonl", recorded / "missing.jsonl"]))
    assert client.get("/health").json()["status"] == "ok"
    traces = client.get("/api/traces").json()
    assert len(traces) == 20 and {"trace_id", "steps", "cost_usd", "case_id"} <= set(traces[0])
    tid = traces[0]["trace_id"]
    detail = client.get(f"/api/traces/{tid}").json()
    assert detail["summary"]["trace_id"] == tid and detail["spans"]
    assert "agentlens traces" in client.get("/").text
    page = client.get(f"/traces/{tid}").text
    assert "execute_tool" in page and "tokens" in page
    assert client.get("/api/traces/nope").status_code == 404
    monkeypatch.setenv("AGENTLENS_TRACES", str(recorded / "v1.jsonl"))
    assert len(TestClient(create_app()).get("/api/traces").json()) == 20
    assert TestClient(create_app([recorded / "none.jsonl"])).get("/api/traces").json() == []


# ---- BFCL conversion ---------------------------------------------------------------------------
def test_bfcl_convert_schema_and_answers():
    schema = {
        "type": "dict",
        "properties": {
            "x": {"type": "float"},
            "t": {"type": "tuple", "items": {"type": "integer"}},
            "a": {"type": "any"},
        },
        "required": ["x"],
    }
    out = convert_schema(schema)
    assert out == {
        "type": "object",
        "properties": {"x": {"type": "number"}, "t": {"type": "array", "items": {"type": "integer"}}, "a": {}},
        "required": ["x"],
    }
    questions = [
        {
            "id": "simple_python_0",
            "question": [[{"role": "user", "content": "area?"}]],
            "function": [{"name": "geo.area", "description": "d", "parameters": schema}],
        },
        {"id": "simple_python_1", "question": [[{"role": "user", "content": "no answer"}]], "function": []},
    ]
    answers = [{"id": "simple_python_0", "ground_truth": [{"geo.area": {"x": [10], "unit": ["m", ""], "t": [[1, 2]]}}]}]
    suite = convert("simple_python", questions, answers)
    (case,) = suite["cases"]
    assert (
        case["tools"][0]["function"]["name"] == "geo_area" and case["max_steps"] == 1 and case["trajectory"] == "strict"
    )
    assert case["expected_tool_calls"] == [
        {"name": "geo_area", "arguments": {"x": 10, "unit": {"$any": ["m", ABSENT]}, "t": [1, 2]}}
    ]
    irr = convert("irrelevance", [{**questions[0], "id": "irrelevance_0"}], None)
    assert irr["cases"][0]["expected_tool_calls"] == []
    par = convert(
        "parallel", [{**questions[0], "id": "p0"}], [{"id": "p0", "ground_truth": [{"geo.area": {"x": [1]}}]}]
    )
    assert par["cases"][0]["trajectory"] == "unordered"


def test_bfcl_suite_scores_with_agentlens(tmp_path):
    from agentlens import MockChatClient, ToolAgent, run_eval
    from agentlens.cases import suite_from_dict
    from agentlens.llm import MockRule

    questions = [
        {
            "id": "q0",
            "question": [[{"role": "user", "content": "area of 10"}]],
            "function": [
                {"name": "geo.area", "parameters": {"type": "dict", "properties": {"x": {"type": "integer"}}}}
            ],
        }
    ]
    answers = [{"id": "q0", "ground_truth": [{"geo.area": {"x": [10], "unit": ["m", ""]}}]}]
    suite = suite_from_dict(convert("simple_python", questions, answers))
    client = MockChatClient([MockRule("area", [{"tool_calls": [{"name": "geo_area", "arguments": {"x": 10}}]}])])
    result = run_eval(suite, ToolAgent(client, "mock-b"))
    assert result.cases[0].success and result.cases[0].exhausted


# ---- LangGraph adapter ---------------------------------------------------------------------------
def test_langgraph_adapter_real_graph():
    pytest.importorskip("langgraph")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool as lc_tool
    from langgraph.prebuilt import create_react_agent

    from agentlens import InMemoryExporter, Tracer
    from agentlens.adapters.langgraph import AgentLensCallbackHandler
    from agentlens.store import group_traces

    @lc_tool
    def get_weather(city: str) -> str:
        """Weather for a city."""
        return f"18C in {city}"

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    msgs = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1"}],
                usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            ),
            AIMessage(
                content="It's 18C in Paris.",
                usage_metadata={"input_tokens": 70, "output_tokens": 8, "total_tokens": 78},
            ),
        ]
    )
    graph = create_react_agent(FakeToolModel(messages=msgs), [get_weather])
    mem = InMemoryExporter()
    tracer = Tracer("lg", exporters=[mem])
    out = graph.invoke(
        {"messages": [("user", "weather in Paris?")]},
        config={"callbacks": [AgentLensCallbackHandler(tracer, run_name="lg-run")]},
    )
    assert out["messages"][-1].content == "It's 18C in Paris."
    (trace,) = group_traces(mem.spans)
    assert trace.root.name == "lg-run" and trace.root.kind == "run"
    assert [c.name for c in trace.tool_calls()] == ["get_weather"]
    assert trace.tool_calls()[0].arguments == {"city": "Paris"}
    s = trace.summary()
    assert s.llm_calls == 2 and s.input_tokens == 120 and s.output_tokens == 18
    assert s.steps == 3  # agent -> tools -> agent
    tool_span = next(sp for sp in mem.spans if sp.kind == "tool")
    assert "18C in Paris" in json.dumps(tool_span.attributes["gen_ai.tool.call.result"])


def test_langgraph_adapter_nested_in_agentlens_run_and_errors():
    pytest.importorskip("langchain_core")
    import uuid

    from agentlens import InMemoryExporter, Tracer
    from agentlens.adapters.langgraph import AgentLensCallbackHandler, _usage

    mem = InMemoryExporter()
    tracer = Tracer(exporters=[mem])
    h = AgentLensCallbackHandler(tracer)
    with tracer.run("outer") as outer:
        root, node, llm, tool = (uuid.uuid4() for _ in range(4))
        h.on_chain_start({"name": "graph"}, {"q": 1}, run_id=root)
        h.on_chain_start(
            {}, {}, run_id=node, parent_run_id=root, metadata={"langgraph_node": "agent", "langgraph_step": 1}
        )
        h.on_llm_start({"kwargs": {"model": "gpt-4o-mini"}}, ["p"], run_id=llm, parent_run_id=node)

        class R:
            generations = []
            llm_output = {"token_usage": {"prompt_tokens": 1000, "completion_tokens": 10}, "model_name": "gpt-4o-mini"}

        h.on_llm_end(R(), run_id=llm)
        h.on_tool_start({"name": "t"}, "plain text", run_id=tool, parent_run_id=node)
        h.on_tool_error(RuntimeError("bad"), run_id=tool)
        h.on_chain_end({}, run_id=node)
        h.on_chain_error(ValueError("x"), run_id=root)
        h.on_llm_error(RuntimeError("never started"), run_id=uuid.uuid4())  # unknown run: ignored
    spans = {s.name: s for s in mem.spans}
    assert spans["graph"].parent_id == outer.span_id and spans["graph"].status == "error"
    assert spans["node agent"].kind == "agent_step"
    assert spans["completion gpt-4o-mini"].attributes["gen_ai.usage.input_tokens"] == 1000
    assert spans["execute_tool t"].attributes["gen_ai.tool.call.arguments"] == {"input": "plain text"}
    assert spans["execute_tool t"].status == "error"
    assert _usage(object()) == (0, 0, None)


def test_eval_from_traces_rescoring(recorded, tmp_path):
    traces = tmp_path / "v1.jsonl"
    traces.write_text((recorded / "v1.jsonl").read_text())
    out = tmp_path / "rescored.json"
    assert (
        main(
            [
                "eval",
                str(EX / "suite.yaml"),
                "--from-traces",
                str(traces),
                "--label",
                "v1",
                "--out",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )
    rescored = json.loads(out.read_text())
    original = json.loads((recorded / "v1.json").read_text())
    for key in ("task_success", "tool_f1", "args_exact", "avg_steps", "total_tokens"):
        assert rescored["summary"][key] == original["summary"][key]
    assert rescored["metadata"]["scored_from_traces"]
    # a later re-run of one case supersedes the earlier trace of that case
    assert (
        main(
            [
                "eval",
                str(EX / "suite.yaml"),
                *V2,
                "--ids",
                "weather-tokyo-f",
                "--trace",
                str(traces),
                "--quiet",
                "--out",
                str(tmp_path / "one.json"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "one.json").read_text())["summary"]["n_cases"] == 1
    assert main(["eval", str(EX / "suite.yaml"), "--from-traces", str(traces), "--out", str(out), "--quiet"]) == 0
    cases = {c["case_id"]: c for c in json.loads(out.read_text())["cases"]}
    assert len(cases) == 20 and cases["weather-tokyo-f"]["success"] is False
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert main(["eval", str(EX / "suite.yaml"), "--from-traces", str(empty)]) == 2


def test_bfcl_nested_possible_answers():
    questions = [
        {
            "id": "q",
            "question": [[{"role": "user", "content": "x"}]],
            "function": [{"name": "f", "parameters": {"type": "dict", "properties": {}}}],
        }
    ]
    answers = [
        {
            "id": "q",
            "ground_truth": [
                {
                    "f": {
                        "info": [{"name": ["John Doe"], "email": ["j@x.com", ""]}],
                        "conds": [[{"field": ["age"], "op": [">", "gt"]}]],
                        "coords": [[1.5, -2.0]],
                    }
                }
            ],
        }
    ]
    (case,) = convert("simple_python", questions, answers)["cases"]
    args = case["expected_tool_calls"][0]["arguments"]
    assert args["info"] == {"name": "John Doe", "email": {"$any": ["j@x.com", ABSENT]}}
    assert args["conds"] == [{"field": "age", "op": {"$any": [">", "gt"]}}]
    assert args["coords"] == [1.5, -2.0]
    from agentlens.metrics import score_arguments

    assert score_arguments(
        args, {"info": {"name": "john doe"}, "conds": [{"field": "age", "op": "gt"}], "coords": [1.5, -2]}
    ).exact


def test_judge_cli_aborts_when_endpoint_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr("agentlens.llm.time.sleep", lambda s: None)
    data = tmp_path / "pairs.jsonl"
    data.write_text(
        "".join(json.dumps({"question": "q", "answer_a": "a", "answer_b": "b", "label": "A"}) + "\n" for _ in range(4))
    )
    rc = main(
        [
            "judge",
            str(data),
            "--model",
            "m",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--timeout",
            "0.5",
            "--max-consecutive-failures",
            "2",
            "--quiet",
        ]
    )
    assert rc == 1
