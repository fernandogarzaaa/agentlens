import asyncio
import threading

import pytest

import agentlens
from agentlens import InMemoryExporter, Pricing, Span, SpanKind, Tracer
from agentlens.cost import ModelPrice
from agentlens.tracer import (
    ATTR_COST,
    ATTR_COST_KNOWN,
    ATTR_INPUT,
    ATTR_INPUT_TOKENS,
    ATTR_OUTPUT_TOKENS,
    ATTR_TOOL_ARGS,
    ATTR_TOOL_RESULT,
    RunSummary,
    _jsonable,
    current_span,
)


@pytest.fixture
def mem():
    return InMemoryExporter()


@pytest.fixture
def tracer(mem):
    pricing = Pricing({"m": ModelPrice(1.0, 2.0)})
    return Tracer("svc", exporters=[mem], pricing=pricing)


def test_ids_follow_w3c_format(tracer, mem):
    with tracer.run("r"):
        pass
    (span,) = mem.spans
    assert len(span.trace_id) == 32 and int(span.trace_id, 16) >= 0
    assert len(span.span_id) == 16 and int(span.span_id, 16) >= 0


def test_nesting_parents_and_trace_ids(tracer, mem):
    with tracer.run("task", input="hi", case_id="c1") as run:
        with tracer.agent_step(0) as step:
            with tracer.llm_call("m") as llm:
                llm.record_usage(input_tokens=100, output_tokens=10)
            with tracer.tool_call("search", arguments={"q": "x"}, call_id="call_1") as tool:
                tool.set_result({"hits": 3})
        run.set_output("done")
    by_name = {s.name: s for s in mem.spans}
    assert set(by_name) == {"task", "agent.step", "chat m", "execute_tool search"}
    assert {s.trace_id for s in mem.spans} == {run.trace_id}
    assert run.parent_id is None
    assert step.parent_id == run.span_id
    assert llm.parent_id == step.span_id and tool.parent_id == step.span_id
    assert tool.attributes[ATTR_TOOL_ARGS] == {"q": "x"}
    assert tool.attributes[ATTR_TOOL_RESULT] == {"hits": 3}
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert run.attributes[ATTR_INPUT] == "hi"
    assert run.attributes["agentlens.case_id"] == "c1"
    assert step.attributes["agentlens.step.index"] == 0
    assert all(s.status == "ok" and s.end_ns >= s.start_ns for s in mem.spans)


def test_children_exported_before_root(tracer, mem):
    with tracer.run("r"):
        with tracer.span("child"):
            pass
        assert [s.name for s in mem.spans] == ["child"]
    assert [s.name for s in mem.spans] == ["child", "r"]


def test_run_always_starts_new_trace(tracer, mem):
    with tracer.run("outer") as outer:
        with tracer.run("inner") as inner:
            pass
    assert inner.trace_id != outer.trace_id and inner.parent_id is None


def test_current_span_restored(tracer):
    assert current_span() is None
    with tracer.run("r") as r:
        assert current_span() is r
        with tracer.span("c") as c:
            assert current_span() is c
        assert current_span() is r
    assert current_span() is None


def test_exception_recorded_and_reraised(tracer, mem):
    with pytest.raises(KeyError):
        with tracer.run("r"):
            with tracer.tool_call("t"):
                raise KeyError("boom")
    tool, run = mem.spans
    assert tool.status == "error" and "KeyError" in tool.status_message
    assert tool.events[0]["name"] == "exception"
    assert tool.events[0]["attributes"]["exception.type"] == "KeyError"
    assert run.status == "error"
    assert current_span() is None


def test_usage_and_cost(tracer, mem):
    with tracer.run("r") as run:
        with tracer.llm_call("m") as llm:
            llm.record_usage(input_tokens=1_000_000, output_tokens=500_000)
            llm.record_usage(input_tokens=1_000_000)  # accumulates
    assert llm.attributes[ATTR_INPUT_TOKENS] == 2_000_000
    assert llm.attributes[ATTR_OUTPUT_TOKENS] == 500_000
    assert llm.attributes[ATTR_COST] == pytest.approx(3.0)
    assert llm.attributes[ATTR_COST_KNOWN] is True
    assert run.attributes["agentlens.run.cost_usd"] == pytest.approx(3.0)
    assert run.attributes["agentlens.run.input_tokens"] == 2_000_000
    assert run.attributes["agentlens.run.cost_known"] is True


def test_response_model_used_for_pricing_and_unknown_cost(tracer):
    with tracer.run("r") as run:
        with tracer.llm_call("unknown-model") as a:
            a.record_usage(10, 10, model="m")
        with tracer.llm_call("other") as b:
            b.record_usage(10, 10)
    assert a.attributes["gen_ai.response.model"] == "m"
    assert a.attributes[ATTR_COST_KNOWN] is True and a.attributes[ATTR_COST] > 0
    assert b.attributes[ATTR_COST_KNOWN] is False and b.attributes[ATTR_COST] == 0.0
    s = tracer.summary(run.trace_id)
    assert s is not None and s.cost_known is False and s.llm_calls == 2


def test_explicit_cost_override_and_cached_tokens(tracer):
    with tracer.run("r"):
        with tracer.llm_call("whatever") as llm:
            llm.record_usage(10, 10, cached_input_tokens=4, cost_usd=0.5)
    assert llm.attributes[ATTR_COST] == 0.5 and llm.attributes[ATTR_COST_KNOWN] is True
    assert llm.attributes["gen_ai.usage.cached_input_tokens"] == 4


def test_run_summary_counts(tracer):
    with tracer.run("r") as run:
        for i in range(2):
            with tracer.agent_step(i):
                with tracer.llm_call("m") as llm:
                    llm.record_usage(10, 5)
                with tracer.tool_call("t"):
                    pass
        with tracer.span("bad") as bad:
            bad.record_error("nope")
    s = tracer.summary(run.trace_id)
    assert (s.steps, s.llm_calls, s.tool_calls, s.input_tokens, s.output_tokens, s.total_tokens, s.errors) == (
        2,
        2,
        2,
        20,
        10,
        30,
        1,
    )
    assert s.latency_s >= 0
    assert s.to_dict()["total_tokens"] == 30


def test_summary_steps_default_to_llm_calls():
    spans = [
        Span("r", SpanKind.RUN, trace_id="t", span_id="1", start_ns=0, end_ns=2_000_000_000),
        Span("l", SpanKind.LLM, trace_id="t", span_id="2", parent_id="1", start_ns=1, end_ns=2),
        Span("l", SpanKind.LLM, trace_id="t", span_id="3", parent_id="1", start_ns=2, end_ns=3),
    ]
    s = RunSummary.from_spans(spans)
    assert s.steps == 2 and s.latency_s == 2.0
    with pytest.raises(ValueError):
        RunSummary.from_spans([])


def test_start_span_manual_and_explicit_parent(tracer, mem):
    root = tracer.start_span("root", kind=SpanKind.RUN, new_trace=True)
    child = tracer.start_span("child", kind=SpanKind.TOOL, parent=root, attributes={"x": 1})
    assert child.trace_id == root.trace_id and child.parent_id == root.span_id
    assert current_span() is None  # start_span does not make the span current
    child.end()
    child.end()  # idempotent
    root.end()
    assert [s.name for s in mem.spans] == ["child", "root"]
    assert tracer.summary(root.trace_id).tool_calls == 1


def test_unknown_kind_rejected(tracer):
    with pytest.raises(ValueError, match="unknown span kind"):
        tracer.start_span("x", kind="banana")


def test_span_round_trip_dict():
    s = Span("n", SpanKind.TOOL, attributes={"a": 1}, events=[{"name": "e"}])
    s.end()
    d = s.to_dict()
    back = Span.from_dict(d)
    assert back.to_dict() == d


def test_jsonable_coercion_and_truncation():
    class Obj:
        def __repr__(self):
            return "Obj()"

    assert _jsonable({"a": (1, 2), 3: Obj()}) == {"a": [1, 2], "3": "Obj()"}
    long = "x" * 9000
    assert _jsonable(long).endswith("...[truncated]") and len(_jsonable(long)) < 9000


def test_decorators_sync(mem):
    t = Tracer("svc", exporters=[mem])

    @agentlens.trace_tool(tracer=t)
    def add(a: int, b: int = 2) -> int:
        return a + b

    @agentlens.trace_llm(model="gpt-4o-mini", tracer=t)
    def fake_llm(prompt):
        return {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    @agentlens.trace_agent_step(tracer=t)
    def step():
        fake_llm("hi")
        return add(1)

    @agentlens.trace_run(tracer=t)
    def run(question):
        return step()

    assert run("q") == 3
    names = {s.name.split(".")[-1]: s for s in mem.spans}
    tool = next(s for s in mem.spans if s.kind == SpanKind.TOOL)
    assert tool.attributes[ATTR_TOOL_ARGS] == {"a": 1} and tool.attributes[ATTR_TOOL_RESULT] == 3
    llm = next(s for s in mem.spans if s.kind == SpanKind.LLM)
    assert llm.attributes[ATTR_INPUT_TOKENS] == 1000 and llm.attributes[ATTR_COST] == pytest.approx(0.00021)
    root = next(s for s in mem.spans if s.kind == SpanKind.RUN)
    assert root.attributes[ATTR_INPUT] == {"question": "q"}
    assert root.attributes["agentlens.output"] == 3
    assert "run" in names and len({s.trace_id for s in mem.spans}) == 1


def test_decorator_usage_from_object_and_other_kind(mem):
    t = Tracer("svc", exporters=[mem])

    class Usage:
        prompt_tokens = 7
        completion_tokens = 3

    class Resp:
        usage = Usage()
        model = "x"

    @agentlens.traced(kind=SpanKind.LLM, model="x", tracer=t)
    def llm():
        return Resp()

    @agentlens.traced(tracer=t, name="custom")
    def other(v):
        return v * 2

    @agentlens.traced(kind=SpanKind.LLM, model="x", tracer=t)
    def no_usage():
        return "text"

    llm()
    other(4)
    no_usage()
    a, b, c = mem.spans
    assert a.attributes[ATTR_INPUT_TOKENS] == 7 and a.attributes[ATTR_OUTPUT_TOKENS] == 3
    assert b.name == "custom" and b.attributes[ATTR_INPUT] == {"v": 4} and b.attributes["agentlens.output"] == 8
    assert ATTR_INPUT_TOKENS not in c.attributes


def test_decorator_async_and_exception(mem):
    t = Tracer("svc", exporters=[mem])

    @agentlens.trace_tool(tracer=t)
    async def fetch(url: str) -> str:
        await asyncio.sleep(0)
        return url.upper()

    @agentlens.trace_tool(tracer=t)
    def fails():
        raise RuntimeError("x")

    assert asyncio.run(fetch("a")) == "A"
    with pytest.raises(RuntimeError):
        fails()
    assert mem.spans[0].attributes[ATTR_TOOL_RESULT] == "A"
    assert mem.spans[1].status == "error"


def test_global_tracer_configure(mem):
    t = agentlens.configure("global-svc", exporters=[mem])
    try:
        assert agentlens.get_tracer() is t

        @agentlens.trace_tool()
        def ping():
            return "pong"

        ping()
        assert mem.spans[0].attributes["gen_ai.tool.name"] == "ping"
        assert mem.spans[0].name == "execute_tool ping"
        assert t.resource_attributes["service.name"] == "global-svc"
    finally:
        agentlens.set_tracer(Tracer())


def test_threads_have_independent_context(tracer, mem):
    def worker(i):
        with tracer.run(f"r{i}"):
            with tracer.tool_call("t"):
                pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    roots = [s for s in mem.spans if s.parent_id is None]
    assert len(roots) == 4 and len({s.trace_id for s in roots}) == 4
    for tool in (s for s in mem.spans if s.kind == SpanKind.TOOL):
        assert any(r.span_id == tool.parent_id and r.trace_id == tool.trace_id for r in roots)


def test_add_exporter_and_shutdown(tracer):
    class Rec:
        def __init__(self):
            self.got, self.closed = [], False

        def export(self, spans):
            self.got.extend(spans)

        def shutdown(self):
            self.closed = True

    rec = Rec()
    tracer.add_exporter(rec)
    with tracer.run("r"):
        pass
    tracer.shutdown()
    assert len(rec.got) == 1 and rec.closed
