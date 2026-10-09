# Tracer API (stable)

This page is the contract for code that instruments itself with agentlens (other projects, adapters,
your own agent). Everything listed here is covered by tests and follows semantic versioning: it
will not change incompatibly within 0.x minor releases without a deprecation period and a CHANGELOG
entry. Anything not listed (underscored names, module internals) may change.

Install: `pip install agentlens` (the tracer, all exporters including OTLP, and the store need only
PyYAML).

## Quick reference

```python
from agentlens import Tracer, JSONLExporter, SQLiteExporter, OTLPExporter, Pricing

tracer = Tracer(
    service_name="sqlpilot",  # becomes service.name in OTLP
    exporters=[JSONLExporter("traces.jsonl")],  # any number of exporters
    pricing=Pricing.default(),  # optional; or Pricing.from_toml("prices.toml")
)

with tracer.run("answer-question", input=question, case_id=None) as run:  # root span, new trace
    with tracer.agent_step(0):  # optional grouping
        with tracer.llm_call("gpt-4o-mini") as llm:
            resp = client.chat(...)
            llm.record_usage(
                input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens, model=resp.model
            )  # cost from the pricing table
        with tracer.tool_call("run_sql", arguments={"query": sql}) as tool:
            rows = db.execute(sql)
            tool.set_result({"rows": len(rows)})
    run.set_output(answer)

summary = tracer.summary(run.trace_id)  # RunSummary: tokens, cost_usd, cost_known, steps, latency_s ...
tracer.shutdown()  # flush and close exporters
```

## `Tracer`

| Member | Contract |
|---|---|
| `Tracer(service_name="agent", exporters=None, pricing=None, resource_attributes=None)` | `pricing` defaults to the bundled table. |
| `run(name="agent.run", input=None, case_id=None, **attributes)` | Context manager yielding the root `Span` of a **new trace** (never nested under the current span). |
| `agent_step(index=None, name="agent.step", **attributes)` | Child span of kind `agent_step`. |
| `llm_call(model, name=None, system=None, **attributes)` | Child span of kind `llm`; sets `gen_ai.operation.name=chat` and `gen_ai.request.model`. |
| `tool_call(tool_name, arguments=None, call_id=None, **attributes)` | Child span of kind `tool`; sets `gen_ai.tool.name` and `gen_ai.tool.call.arguments`. |
| `span(name, kind="other", attributes=None, new_trace=False)` | Generic context manager. |
| `start_span(name, kind="other", attributes=None, parent=None, new_trace=False)` | Manual span (not made current). Call `span.end()` yourself. For callback-style integrations. |
| `summary(trace_id) -> RunSummary or None` | Available once the root span has ended. |
| `add_exporter(exporter)`, `shutdown()` | |

Context managers record any exception on the span (`status="error"`, an `exception` event with type,
message and stack trace) and re-raise it. Parent/child links use `contextvars`, so threads and
asyncio tasks each see their own current span.

## `Span`

| Member | Contract |
|---|---|
| `trace_id` (32 hex), `span_id` (16 hex), `parent_id`, `name`, `kind`, `start_ns`, `end_ns`, `status` (`unset`/`ok`/`error`), `status_message`, `attributes`, `events` | W3C / OpenTelemetry formats; Unix-epoch nanoseconds. |
| `set_attribute(key, value)`, `set_attributes(dict)`, `add_event(name, attributes=None)` | Values are coerced to JSON; strings over 8000 chars are truncated. Chainable. |
| `set_input(v)`, `set_output(v)`, `set_result(v)` | `agentlens.input`, `agentlens.output`, `gen_ai.tool.call.result`. |
| `record_usage(input_tokens=0, output_tokens=0, cached_input_tokens=0, model=None, cost_usd=None)` | Accumulates `gen_ai.usage.*`; computes `agentlens.cost_usd` from the tracer's pricing (response model, else request model). Unknown model: cost `0.0` with `agentlens.cost_known=false`. `cost_usd` overrides. |
| `record_error(exc_or_message)` | |
| `end()` | Idempotent. |
| `to_dict()` / `Span.from_dict(d)` | The JSONL record format. |

## Span kinds and attributes

`SpanKind.RUN`, `AGENT_STEP`, `LLM`, `TOOL`, `OTHER` (stored in `agentlens.span.kind`).

GenAI semantic-convention attributes used: `gen_ai.operation.name`, `gen_ai.system`,
`gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.tool.name`, `gen_ai.tool.call.id`,
`gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`.

agentlens attributes: `agentlens.input`, `agentlens.output`, `agentlens.case_id`,
`agentlens.step.index`, `agentlens.cost_usd`, `agentlens.cost_known`, and on the root span after it
ends `agentlens.run.{llm_calls,tool_calls,steps,input_tokens,output_tokens,cost_usd,cost_known}`.

## Decorators and the global tracer

```python
import agentlens

agentlens.configure("my-service", exporters=[agentlens.JSONLExporter("traces.jsonl")])


@agentlens.trace_tool()  # span "execute_tool search", arguments + result captured
def search(query: str) -> list[str]: ...


@agentlens.trace_llm(model="gpt-4o-mini")  # records usage if the return value has an OpenAI-style .usage
def ask(prompt): ...


@agentlens.trace_run()  # new trace per call
def answer(question): ...
```

`traced(kind=..., name=None, tracer=None, capture_args=True, capture_result=True, model=None)` is
the general form; all decorators work on sync and async functions. `get_tracer()`, `set_tracer()`,
`current_span()` are stable.

## Exporters

`InMemoryExporter()`, `JSONLExporter(path)` (append), `SQLiteExporter(path)`,
`OTLPExporter(endpoint="http://localhost:4318/v1/traces", service_name="agent", headers=None, batch_size=64, timeout_s=5.0)`.
An exporter is any object with `export(spans)` and `shutdown()`. Exporters never raise into traced
code; OTLP failures are counted in `dropped_spans`. Loopback endpoints bypass `HTTP(S)_PROXY`.

Read traces back with `agentlens.load_traces(path_or_paths)` -> `list[Trace]` (`root`, `input`,
`output`, `case_id`, `tool_calls()`, `summary()`, `children(span)`).

## Pricing

`Pricing.default()`, `Pricing.from_toml(path)`, `Pricing.from_dict(d)`, `pricing.merged(other)`,
`pricing.lookup(model) -> ModelPrice | None`, `pricing.cost(model, input_tokens, output_tokens, cached_input_tokens) -> float | None`.
Lookup order: exact name, alias, file-path basename without `.gguf`, then the most specific glob.
