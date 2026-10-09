import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentlens import JSONLExporter, OTLPExporter, SQLiteExporter, Tracer, load_traces
from agentlens.exporters import spans_to_otlp
from agentlens.metrics import ToolCall
from agentlens.store import group_traces, load_spans


def _record(tracer, n=2):
    for i in range(n):
        with tracer.run(f"run {i}", input=f"q{i}", case_id=f"c{i}") as run:
            with tracer.agent_step(0):
                with tracer.llm_call("gpt-4o-mini") as llm:
                    llm.record_usage(100, 20)
                with tracer.tool_call("search", arguments={"q": i}) as t:
                    t.set_result([i])
                with tracer.tool_call("fetch", arguments={"id": i}):
                    pass
            run.set_output(f"answer {i}")


@pytest.mark.parametrize("suffix", [".jsonl", ".db", ".sqlite"])
def test_round_trip_through_store(tmp_path, suffix):
    path = tmp_path / f"traces{suffix}"
    exporter = JSONLExporter(path) if suffix == ".jsonl" else SQLiteExporter(path)
    tracer = Tracer("svc", exporters=[exporter])
    _record(tracer)
    tracer.shutdown()
    traces = load_traces(path)
    assert len(traces) == 2
    t0 = traces[0]
    assert t0.name == "run 0" and t0.input == "q0" and t0.output == "answer 0" and t0.case_id == "c0"
    assert t0.tool_calls() == [ToolCall("search", {"q": 0}), ToolCall("fetch", {"id": 0})]
    s = t0.summary()
    assert (s.llm_calls, s.tool_calls, s.steps, s.input_tokens, s.output_tokens) == (1, 2, 1, 100, 20)
    assert s.cost_usd == pytest.approx(100 * 0.15e-6 + 20 * 0.6e-6)
    assert [c.name for c in t0.children(t0.root)] == ["agent.step"]
    assert t0.to_dict()["trace_id"] == t0.trace_id


def test_exporters_ignore_spans_after_shutdown(tmp_path):
    j = JSONLExporter(tmp_path / "a.jsonl")
    s = SQLiteExporter(tmp_path / "a.db")
    tracer = Tracer("svc", exporters=[j, s])
    tracer.shutdown()
    tracer.shutdown()  # idempotent
    _record(tracer, 1)  # must not raise
    assert (tmp_path / "a.jsonl").read_text() == ""


def test_jsonl_appends_across_exporters(tmp_path):
    path = tmp_path / "t.jsonl"
    for _ in range(2):
        tr = Tracer(exporters=[JSONLExporter(path)])
        _record(tr, 1)
        tr.shutdown()
    assert len(load_traces(path)) == 2


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_spans(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"name": "x"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_spans(bad)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n\n")
    assert load_traces(blank) == []


def test_store_handles_string_arguments_and_missing_root(tmp_path):
    from agentlens.tracer import Span, SpanKind

    spans = [
        Span(
            "t",
            SpanKind.TOOL,
            trace_id="a" * 32,
            span_id="1",
            parent_id="zz",
            start_ns=5,
            end_ns=6,
            attributes={"gen_ai.tool.name": "x", "gen_ai.tool.call.arguments": '{"k": 1}'},
        ),
        Span(
            "t",
            SpanKind.TOOL,
            trace_id="a" * 32,
            span_id="2",
            parent_id="zz",
            start_ns=7,
            end_ns=8,
            attributes={"gen_ai.tool.call.arguments": "not json"},
        ),
    ]
    (trace,) = group_traces(spans)
    assert trace.root is None and trace.input is None and trace.output is None and trace.case_id is None
    assert trace.name == "a" * 32
    assert trace.tool_calls() == [ToolCall("x", {"k": 1}), ToolCall("t", {"_raw": "not json"})]


def test_load_multiple_stores(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.db"
    ta, tb = Tracer(exporters=[JSONLExporter(a)]), Tracer(exporters=[SQLiteExporter(b)])
    _record(ta, 1)
    _record(tb, 1)
    ta.shutdown()
    tb.shutdown()
    assert len(load_traces([a, b])) == 2


# ---- OTLP ---------------------------------------------------------------------------------
def test_spans_to_otlp_shape():
    tracer = Tracer()
    from agentlens import InMemoryExporter

    mem = InMemoryExporter()
    tracer.add_exporter(mem)
    with tracer.run("r", input={"q": 1}):
        with tracer.tool_call("t", arguments={"a": [1, 2]}) as t:
            t.set_attribute("tags", ["x", "y"])
            t.set_attribute("flag", True)
            t.set_attribute("ratio", 0.5)
            t.set_attribute("nothing", None)
            t.record_error("bad")
    body = spans_to_otlp(mem.spans, {"service.name": "svc"})
    rs = body["resourceSpans"][0]
    assert rs["resource"]["attributes"] == [{"key": "service.name", "value": {"stringValue": "svc"}}]
    spans = rs["scopeSpans"][0]["spans"]
    assert rs["scopeSpans"][0]["scope"]["name"] == "agentlens"
    tool, root = spans
    assert tool["parentSpanId"] == root["spanId"] and "parentSpanId" not in root
    assert tool["kind"] == 3 and root["kind"] == 1
    assert tool["status"] == {"code": 2, "message": "bad"} and root["status"]["code"] == 1
    attrs = {a["key"]: a["value"] for a in tool["attributes"]}
    assert attrs["tags"] == {"arrayValue": {"values": [{"stringValue": "x"}, {"stringValue": "y"}]}}
    assert attrs["flag"] == {"boolValue": True}
    assert attrs["ratio"] == {"doubleValue": 0.5}
    assert attrs["nothing"] == {"stringValue": "null"}
    assert json.loads(attrs["gen_ai.tool.call.arguments"]["stringValue"]) == {"a": [1, 2]}
    assert isinstance(tool["startTimeUnixNano"], str) and int(tool["endTimeUnixNano"]) >= int(tool["startTimeUnixNano"])


class _Collector(BaseHTTPRequestHandler):
    bodies: list = []
    status = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).bodies.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(length))))
        self.send_response(type(self).status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def collector():
    _Collector.bodies = []
    _Collector.status = 200
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    yield server
    server.shutdown()


def test_otlp_exporter_posts_on_root_end(collector):
    url = f"http://127.0.0.1:{collector.server_port}/v1/traces"
    exp = OTLPExporter(url, service_name="svc", headers={"Authorization": "Bearer t"}, batch_size=100)
    tracer = Tracer(exporters=[exp])
    _record(tracer, 2)
    tracer.shutdown()
    assert exp.sent_spans == 10 and exp.dropped_spans == 0
    assert len(_Collector.bodies) == 2  # one flush per finished run
    path, auth, body = _Collector.bodies[0]
    assert path == "/v1/traces" and auth == "Bearer t"
    assert len(body["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 5


def test_otlp_exporter_batches_and_flushes(collector):
    url = f"http://127.0.0.1:{collector.server_port}/v1/traces"
    exp = OTLPExporter(url, batch_size=2)
    tracer = Tracer(exporters=[exp])
    root = tracer.start_span("r", kind="run", new_trace=True)
    for i in range(3):
        tracer.start_span(f"c{i}", parent=root).end()
    assert exp.sent_spans == 2  # first batch of 2 sent, 1 buffered
    exp.flush()
    assert exp.sent_spans == 3
    exp.flush()  # nothing buffered: no request
    assert len(_Collector.bodies) == 2
    with pytest.raises(ValueError):
        OTLPExporter(url, batch_size=0)


def test_otlp_exporter_counts_drops_without_raising(collector):
    _Collector.status = 500
    exp = OTLPExporter(f"http://127.0.0.1:{collector.server_port}/v1/traces")
    tracer = Tracer(exporters=[exp])
    _record(tracer, 1)
    assert exp.dropped_spans == 5 and exp.sent_spans == 0
    dead = OTLPExporter("http://127.0.0.1:9/v1/traces", timeout_s=0.5)
    tracer2 = Tracer(exporters=[dead])
    _record(tracer2, 1)
    assert dead.dropped_spans == 5
