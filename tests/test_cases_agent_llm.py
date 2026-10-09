import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Literal

import pytest

from agentlens import InMemoryExporter, MockChatClient, OpenAICompatClient, Tool, ToolAgent, Toolset, Tracer, tool
from agentlens._http import is_loopback
from agentlens.agent import CannedToolset, load_toolset
from agentlens.cases import SuiteError, load_suite, suite_from_dict
from agentlens.llm import ChatResponse, LLMError, MockRule, ToolCallRequest, estimate_tokens, parse_chat_response
from agentlens.metrics import ABSENT, ToolCall


# ---- suites -----------------------------------------------------------------------------
def test_suite_from_dict_with_defaults():
    s = suite_from_dict(
        {
            "suite": "demo",
            "defaults": {
                "trajectory": "subset",
                "max_steps": 3,
                "allow_extra_args": True,
                "constraints": {"must_not_call": ["rm"]},
            },
            "cases": [
                {
                    "id": "a",
                    "input": "hi",
                    "expected_tool_calls": [{"name": "t", "arguments": {"x": 1}}],
                    "tags": ["x"],
                },
                {
                    "id": "b",
                    "input": [{"role": "user", "content": "yo"}],
                    "trajectory": "strict",
                    "max_steps": 1,
                    "constraints": {"must_not_call": ["other"]},
                    "final_answer": {"contains": ["ok"]},
                },
            ],
        }
    )
    a, b = s.cases
    assert s.name == "demo"
    assert a.trajectory == "subset" and a.max_steps == 3 and a.allow_extra_args
    assert a.constraints.must_not_call == ["rm"] and a.expected_tool_calls == [ToolCall("t", {"x": 1})]
    assert a.expected_names == ["t"] and a.input_text == "hi"
    assert b.trajectory == "strict" and b.max_steps == 1 and b.constraints.must_not_call == ["other"]
    assert b.input_text == "yo" and b.final_answer.contains == ["ok"]
    assert s.get("b") is b and s.get("zzz") is None


def test_suite_bare_list_and_generated_ids():
    s = suite_from_dict([{"input": "a"}, {"input": "b"}], default_name="x")
    assert [c.id for c in s.cases] == ["case-0", "case-1"] and s.name == "x"
    assert s.cases[0].trajectory == "unordered" and s.cases[0].max_steps == 8


def test_input_text_without_user_message():
    s = suite_from_dict([{"input": [{"role": "system", "content": "s"}]}])
    assert s.cases[0].input_text == json.dumps([{"role": "system", "content": "s"}])


@pytest.mark.parametrize(
    "data,match",
    [
        ("nope", "mapping"),
        ({"cases": []}, "non-empty"),
        ({"cases": [{"id": "a"}]}, "'input' is required"),
        ({"cases": [{"input": 3}]}, "input must be"),
        ({"cases": ["x"]}, "must be a mapping"),
        ({"cases": [{"input": "a", "bogus": 1}]}, "unknown keys"),
        ({"cases": [{"input": "a", "trajectory": "fuzzy"}]}, "trajectory must be"),
        ({"cases": [{"input": "a", "final_answer": {"bad": 1}}]}, "unknown final_answer"),
        ({"cases": [{"input": "a", "constraints": {"bad": 1}}]}, "unknown constraints"),
        ({"cases": [{"input": "a", "expected_tool_calls": [{"arguments": {}}]}]}, "name"),
        ({"cases": [{"id": "x", "input": "a"}, {"id": "x", "input": "b"}]}, "duplicate"),
        ({"cases": [{"input": "a"}], "extra": 1}, "unknown top-level"),
        ({"cases": [{"input": "a"}], "defaults": []}, "'defaults' must be"),
        ({"cases": [{"input": "a"}], "defaults": {"constraints": {"x": 1}}}, "defaults"),
    ],
)
def test_suite_errors(data, match):
    with pytest.raises(SuiteError, match=match):
        suite_from_dict(data)


def test_load_suite_formats(tmp_path):
    (tmp_path / "s.yaml").write_text("suite: y\ncases:\n  - id: a\n    input: hi\n")
    (tmp_path / "s.json").write_text(json.dumps({"cases": [{"id": "a", "input": "hi"}]}))
    (tmp_path / "s.jsonl").write_text('{"id": "a", "input": "hi"}\n\n{"id": "b", "input": "yo"}\n')
    assert load_suite(tmp_path / "s.yaml").name == "y"
    assert load_suite(tmp_path / "s.json").name == "s"
    assert [c.id for c in load_suite(tmp_path / "s.jsonl").cases] == ["a", "b"]
    assert load_suite(tmp_path / "s.yaml").base_dir == tmp_path.resolve()
    (tmp_path / "bad.yaml").write_text("cases: [\n")
    with pytest.raises(SuiteError, match="bad.yaml"):
        load_suite(tmp_path / "bad.yaml")
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "missing.yaml")


def test_suite_filter():
    s = suite_from_dict([{"id": f"c{i}", "input": "x", "tags": ["even" if i % 2 == 0 else "odd"]} for i in range(6)])
    assert [c.id for c in s.filter(tags=["even"]).cases] == ["c0", "c2", "c4"]
    assert [c.id for c in s.filter(ids=["c5", "c1"]).cases] == ["c1", "c5"]
    assert [c.id for c in s.filter(limit=2).cases] == ["c0", "c1"]
    assert len(s.filter().cases) == 6


def test_example_suite_loads():
    s = load_suite("examples/travel_desk/suite.yaml")
    assert len(s.cases) == 20 and s.toolset == "tools.py:TOOLSET"
    cancel = s.get("cancel-booking")
    assert cancel.constraints.must_not_call == ["book_flight"]  # case overrides the suite default
    assert s.get("weather-paris").constraints.must_not_call == ["cancel_booking"]


# ---- tools ------------------------------------------------------------------------------
@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius", days: int | None = None) -> dict:
    """Weather for a city.

    More text."""
    return {"city": city, "unit": unit}


def test_tool_schema_from_type_hints():
    t: Tool = get_weather.__agentlens_tool__
    assert t.name == "get_weather" and t.description == "Weather for a city."
    props = t.parameters["properties"]
    assert props["city"] == {"type": "string"}
    assert props["unit"] == {"type": "string", "enum": ["celsius", "fahrenheit"]}
    assert props["days"] == {"type": "integer"}
    assert t.parameters["required"] == ["city"]
    assert t.schema()["type"] == "function" and t.schema()["function"]["name"] == "get_weather"


def test_tool_schema_misc_types():
    def f(a: float, b: bool, c: list[int], d: dict[str, int], e, *args, **kwargs):
        return None

    t = Tool.from_function(f, name="renamed", description="desc")
    props = t.parameters["properties"]
    assert (t.name, t.description) == ("renamed", "desc")
    assert props == {
        "a": {"type": "number"},
        "b": {"type": "boolean"},
        "c": {"type": "array"},
        "d": {"type": "object"},
        "e": {"type": "string"},
    }
    assert Tool.from_function(lambda: None).description == "<lambda>"


def test_toolset_calls_and_errors():
    def boom():
        raise RuntimeError("down")

    ts = Toolset([get_weather, boom, Tool("schema_only", "", {"type": "object"})])
    assert ts.names() == ["get_weather", "boom", "schema_only"]
    assert ts.call("get_weather", {"city": "X"}) == {"city": "X", "unit": "celsius"}
    assert "unknown tool" in ts.call("nope", {})["error"]
    assert "bad arguments" in ts.call("get_weather", {"town": "X"})["error"]
    assert ts.call("boom", {}) == {"error": "RuntimeError: down"}
    assert "no implementation" in ts.call("schema_only", {})["error"]
    assert len(ts.schemas()) == 3


def test_canned_toolset_rules():
    schema = {"type": "function", "function": {"name": "w", "parameters": {"type": "object", "properties": {}}}}
    other = {"name": "plain"}
    ts = CannedToolset(
        [schema, other, {"name": "fixed"}],
        {"w": [{"when": {"city": "Paris"}, "result": {"t": 18}}, {"result": {"t": 0}}], "fixed": {"v": 1}},
    )
    assert ts.call("w", {"city": "paris"}) == {"t": 18}
    assert ts.call("w", {"city": "Rome"}) == {"t": 0}
    assert ts.call("plain", {}) == {"status": "ok"}
    assert ts.call("fixed", {"a": 1}) == {"v": 1}
    assert "unknown tool" in ts.call("zzz", {})["error"]
    strict = CannedToolset([schema], {"w": [{"when": {"city": "Paris"}, "result": 1}]})
    assert "no canned result" in strict.call("w", {"city": "Rome"})["error"]
    assert "no canned result" in strict.call("w", {})["error"]


def test_load_toolset_variants(tmp_path):
    ts = load_toolset("examples/travel_desk/tools.py:TOOLSET")
    assert "book_flight" in ts.names()
    (tmp_path / "mytools.py").write_text(
        "def ping(x: str) -> str:\n    return x\n\ndef make():\n    from agentlens import Toolset\n    return Toolset([ping])\n\nLIST = [ping]\nNOT = 3\n"
    )
    assert load_toolset("mytools.py:make", tmp_path).names() == ["ping"]
    assert load_toolset(f"{tmp_path}/mytools.py:LIST").names() == ["ping"]
    with pytest.raises(ValueError, match="not a Toolset"):
        load_toolset("json:loads")
    with pytest.raises(ValueError, match="module:attr"):
        load_toolset("no_colon")
    with pytest.raises(ValueError, match="not a Toolset"):
        load_toolset("mytools.py:NOT", tmp_path)


# ---- llm parsing / clients ----------------------------------------------------------------
def test_parse_chat_response_tool_calls_and_usage():
    data = {
        "model": "m",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}},
                        {"function": {"name": "g", "arguments": "not json"}},
                        {"id": "c3", "function": {"name": "h", "arguments": {"b": 2}}},
                        {"id": "c4", "function": {"name": "k", "arguments": "[1]"}},
                        {"id": "c5", "function": {"name": "e", "arguments": ""}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    r = parse_chat_response(data)
    assert r.model == "m" and r.input_tokens == 10 and r.output_tokens == 5 and r.finish_reason == "tool_calls"
    f, g, h, k, e = r.tool_calls
    assert (f.id, f.name, f.arguments, f.parse_error) == ("c1", "f", {"a": 1}, False)
    assert g.id == "call_1" and g.parse_error and g.arguments == {"_raw": "not json"}
    assert h.arguments == {"b": 2} and not h.parse_error
    assert k.parse_error and e.arguments == {} and not e.parse_error
    msg = r.assistant_message()
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    with pytest.raises(LLMError):
        parse_chat_response({"choices": []})


def test_assistant_message_plain():
    assert ChatResponse("hi").assistant_message() == {"role": "assistant", "content": "hi"}
    m = ChatResponse(None, [ToolCallRequest("i", "n", {"x": 1})]).assistant_message()
    assert json.loads(m["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_mock_client_rules_and_steps():
    m = MockChatClient(
        [MockRule("weather", [{"tool_calls": [{"name": "w", "arguments": {"c": "P"}}]}, {"content": "18"}])]
    )
    first = m.chat([{"role": "user", "content": "Weather?"}], "mock-x", tools=[{"x": 1}])
    assert first.tool_calls[0].name == "w" and first.tool_calls[0].arguments == {"c": "P"}
    assert first.input_tokens > 0 and first.output_tokens > 0 and first.finish_reason == "tool_calls"
    second = m.chat([{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": None}], "mock-x")
    assert second.content == "18" and second.finish_reason == "stop"
    third = m.chat([{"role": "user", "content": "weather"}, {"role": "assistant"}, {"role": "assistant"}], "m")
    assert third.content == "I don't know."
    assert m.chat([{"role": "user", "content": "other"}], "m").content == "I don't know."
    assert m.calls == 4


def test_mock_client_from_yaml():
    m = MockChatClient.from_yaml("examples/travel_desk/mock_v1.yaml")
    r = m.chat([{"role": "user", "content": "What's the weather in Paris right now?"}], "mock-v1")
    assert r.tool_calls[0].name == "get_weather"


class _FakeOpenAI(BaseHTTPRequestHandler):
    responses: list = []
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, self.headers.get("Authorization"), body))
        status, payload = type(self).responses.pop(0)
        raw = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_openai():
    _FakeOpenAI.responses, _FakeOpenAI.seen = [], []
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


OK = {
    "model": "m",
    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}


def test_openai_client_request_shape(fake_openai):
    _FakeOpenAI.responses = [(200, OK)]
    c = OpenAICompatClient(f"http://127.0.0.1:{fake_openai.server_port}/v1/", api_key="sk-test", extra_body={"seed": 1})
    r = c.chat([{"role": "user", "content": "hi"}], "m", tools=[{"type": "function"}], max_tokens=7)
    assert r.content == "hello"
    path, auth, body = _FakeOpenAI.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["tools"] == [{"type": "function"}] and body["tool_choice"] == "auto"
    assert body["max_tokens"] == 7 and body["seed"] == 1 and body["temperature"] == 0.0


def test_openai_client_retries_5xx_not_4xx(fake_openai, monkeypatch):
    monkeypatch.setattr("agentlens.llm.time.sleep", lambda s: None)
    url = f"http://127.0.0.1:{fake_openai.server_port}/v1"
    _FakeOpenAI.responses = [(503, {"error": "busy"}), (200, OK)]
    assert OpenAICompatClient(url).chat([], "m").content == "hello"
    _FakeOpenAI.responses = [(400, {"error": "bad"})]
    with pytest.raises(LLMError, match="HTTP 400"):
        OpenAICompatClient(url).chat([], "m")
    _FakeOpenAI.responses = [(500, {}), (500, {}), (500, {})]
    with pytest.raises(LLMError, match="HTTP 500"):
        OpenAICompatClient(url, max_retries=2).chat([], "m")
    _FakeOpenAI.responses = [(200, b"not json")]
    with pytest.raises(LLMError):
        OpenAICompatClient(url, max_retries=0).chat([], "m")
    with pytest.raises(LLMError):
        OpenAICompatClient("http://127.0.0.1:9/v1", max_retries=0, timeout_s=0.5).chat([], "m")


def test_loopback_detection():
    assert is_loopback("http://localhost:8080/v1")
    assert is_loopback("http://127.0.0.2/x")
    assert is_loopback("http://[::1]:80/")
    assert not is_loopback("https://api.openai.com/v1")


# ---- ToolAgent -------------------------------------------------------------------------------
def _agent_with(rules, toolset=None):
    mem = InMemoryExporter()
    tracer = Tracer(exporters=[mem])
    agent = ToolAgent(MockChatClient(rules), model="mock-1", system_prompt="be brief")
    return agent, tracer, mem, toolset or Toolset([get_weather])


def test_tool_agent_loop_traces_everything():
    agent, tracer, mem, ts = _agent_with(
        [
            MockRule(
                "paris", [{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}, {"content": "18C"}]
            )
        ]
    )
    with tracer.run("r") as root:
        result = agent.run("Paris weather", ts, tracer=tracer)
    assert result.final_answer == "18C" and result.steps == 2 and result.error is None and not result.exhausted
    assert result.messages[0] == {"role": "system", "content": "be brief"}
    assert result.messages[-2]["role"] == "tool"
    s = tracer.summary(root.trace_id)
    assert (s.steps, s.llm_calls, s.tool_calls) == (2, 2, 1) and s.cost_usd > 0 and s.cost_known
    tool_span = next(x for x in mem.spans if x.kind == "tool")
    assert tool_span.attributes["gen_ai.tool.call.result"] == {"city": "Paris", "unit": "celsius"}


def test_tool_agent_max_steps_and_errors():
    loop = [MockRule("x", [{"tool_calls": [{"name": "nope", "arguments": {}}]}] * 5)]
    agent, tracer, mem, ts = _agent_with(loop)
    result = agent.run([{"role": "user", "content": "x"}], ts, max_steps=2, tracer=tracer, system_prompt="override")
    assert result.exhausted and result.final_answer is None and result.steps == 2
    assert result.messages[0]["content"] == "override"
    assert all(s.status == "error" for s in mem.spans if s.kind == "tool")


def test_tool_agent_llm_error_and_bad_json():
    class Failing:
        def chat(self, *a, **k):
            raise LLMError("down")

    tracer = Tracer()
    r = ToolAgent(Failing(), "m").run("hi", Toolset(), tracer=tracer)
    assert r.error == "down" and r.final_answer is None

    class BadJSON:
        n = 0

        def chat(self, messages, model, **k):
            self.n += 1
            if self.n == 1:
                return ChatResponse(None, [ToolCallRequest("c", "get_weather", {"_raw": "{"}, "{", parse_error=True)])
            return ChatResponse("ok")

    mem = InMemoryExporter()
    r2 = ToolAgent(BadJSON(), "m").run("hi", Toolset([get_weather]), tracer=Tracer(exporters=[mem]))
    assert r2.final_answer == "ok"
    tool_span = next(s for s in mem.spans if s.kind == "tool")
    assert tool_span.status == "error" and "not valid JSON" in json.dumps(
        tool_span.attributes["gen_ai.tool.call.result"]
    )


def test_absent_marker_constant():
    assert ABSENT == "$absent"
