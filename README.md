# agentlens

**Tracing and evaluation for tool-calling LLM agents.** OpenTelemetry-compatible spans with token and
cost accounting, trajectory-aware metrics, an LLM-as-judge you can calibrate against human labels,
replay of recorded runs against a new agent version, and a CI gate that fails the build when an agent
regresses.

[![CI](https://github.com/fernandogarzaaa/agentlens/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/agentlens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## The problem

A tool-calling agent can fail in ways a final-answer check never sees: it calls the right tool with a
wrong argument and gets lucky, books before it searches, calls a forbidden tool, loops for ten steps,
or quietly doubles its token bill after a prompt change. Generic LLM eval harnesses score text;
generic tracing tools show you one run at a time. What you need in practice is:

1. **traces** of every run (steps, model calls, tool calls, tokens, dollars) in a standard format;
2. **metrics on the trajectory**, not just the answer: did it pick the right tools, with the right
   arguments, in an acceptable order, within budget?
3. a **judge** for open-ended answers whose agreement with humans you have actually measured;
4. **replay**: run yesterday's inputs through today's agent and see exactly which cases got worse;
5. a **gate** in CI that exits non-zero when any of that drops below a threshold.

agentlens does those five things with a dependency footprint of one package (PyYAML) for the core.

## Quickstart (no API key, no model download)

From a fresh clone:

```bash
git clone https://github.com/fernandogarzaaa/agentlens && cd agentlens
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server]"

# 1. Evaluate an agent on the bundled 20-case travel-desk suite. The "agent" is the built-in
#    OpenAI-compatible tool loop driven by a deterministic mock model (examples/travel_desk/mock_v1.yaml).
agentlens eval examples/travel_desk/suite.yaml --agent mock --mock-rules examples/travel_desk/mock_v1.yaml \
  --model mock-v1 --trace out/v1.jsonl --out out/v1.json --report out/v1.md

# 2. A "new version" of the agent (mock_v2) arrives. Replay the recorded v1 runs against it:
agentlens replay out/v1.jsonl --agent mock --mock-rules examples/travel_desk/mock_v2.yaml --model mock-v2 \
  --suite examples/travel_desk/suite.yaml --report out/diff.md --candidate-out out/v2.json

# 3. CI gate: v1 passes (exit 0), v2 fails (exit 1).
agentlens gate out/v1.json --config examples/travel_desk/gate.toml; echo "exit $?"
agentlens gate out/v2.json --config examples/travel_desk/gate.toml; echo "exit $?"

# 4. Look at the traces in the browser (http://127.0.0.1:8000).
agentlens serve out/v1.jsonl
```

`make quickstart` runs steps 1 to 3. Against a real model, replace the mock flags with
`--model <name> --base-url <OpenAI-compatible URL>` (key from `$OPENAI_API_KEY`, or `--api-key-env`).

## What you get

### Tracer

```python
from agentlens import Tracer, JSONLExporter, OTLPExporter

tracer = Tracer("my-agent", exporters=[JSONLExporter("traces.jsonl"), OTLPExporter("http://localhost:4318/v1/traces")])

with tracer.run("support-ticket", input=question) as run:
    with tracer.agent_step(0):
        with tracer.llm_call("gpt-4o-mini") as llm:
            resp = client.chat.completions.create(...)
            llm.record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens, model=resp.model)
        with tracer.tool_call("lookup_order", arguments={"id": "A17"}) as tool:
            tool.set_result(lookup_order("A17"))
    run.set_output(answer)

print(tracer.summary(run.trace_id))  # steps, llm/tool calls, tokens, cost_usd, cost_known, latency
```

- Spans use W3C trace/span ids and OpenTelemetry GenAI attribute names (`gen_ai.usage.input_tokens`,
  `gen_ai.tool.name`, ...), so the OTLP exporter feeds Jaeger, Tempo, Phoenix, Langfuse or any OTLP/HTTP
  collector unchanged. Local stores: JSONL and SQLite.
- Decorators for code you don't want to restructure: `@trace_tool()`, `@trace_llm(model=...)`,
  `@trace_agent_step()`, `@trace_run()` (sync and async).
- Cost comes from a TOML pricing table (`src/agentlens/data/pricing.toml`, override with `--pricing`).
  Models not in the table are reported as **unknown cost**, never silently as $0.
- The tracer API is a documented, tested contract: [docs/tracer-api.md](docs/tracer-api.md).

### Eval harness

Test cases in YAML/JSON:

```yaml
- id: book-cheapest-mnl-sin
  input: Book the cheapest flight from Manila to Singapore on 2026-11-02 for Ana Cruz.
  expected_tool_calls:
    - name: search_flights
      arguments: {origin: MNL, destination: SIN, date: "2026-11-02"}
    - name: book_flight
      arguments: {flight_id: 5J803, passenger_name: Ana Cruz}
  trajectory: strict                    # strict | unordered | subset | subsequence | none
  final_answer: {contains: [5E94C3]}    # contains / contains_any / not_contains / regex / equals / number
  constraints:
    order: [[search_flights, book_flight]]
    must_not_call: [cancel_booking]
    max_steps: 4
    no_repeat: true
```

Argument values can be matchers: `{"$any": [a, b]}`, `"$absent"` (must not be passed; inside `$any`:
optional) and `{"$regex": "..."}`. Tools come from a Python toolset (`toolset: tools.py:TOOLSET`) or
from declared schemas with canned results, so a suite needs no live services.

| Metric | Definition |
|---|---|
| Tool precision / recall / F1 | Tool names as multisets (calling `search` twice when once was expected halves precision). Macro (mean over cases) and micro (pooled). |
| Args exact / partial | Expected calls paired with actual calls of the same name (best match); exact = all expected arguments right and no unexpected ones; partial = fraction of expected argument keys right. |
| Trajectory match | Per the case's mode. |
| Constraints pass | `max_steps`, `max_tool_calls`, `must_call`, `must_not_call`, `order`, `no_repeat`. |
| Task success | No error, trajectory ok, all expected calls exactly right, constraints hold, answer checks (and judge, if enabled) pass. |
| Steps, tokens, $ / task, latency p50/p95 | From the traces. |

Any agent can be evaluated: the built-in loop (`--model ... --base-url ...`), your own object with a
`run()` method (`--agent mypkg.module:factory`), or a LangGraph graph through the callback adapter
(`agentlens.adapters.langgraph.AgentLensCallbackHandler`, see [examples/langgraph_agent.py](examples/langgraph_agent.py)).

### LLM-as-judge with calibration

`RubricJudge` scores an answer on a rubric (YAML: criteria, scale, pass threshold); `PairwiseJudge`
picks the better of two answers (MT-Bench prompt format, optional order swap against position bias).
`agentlens judge data.jsonl --model ...` runs a judge over human-labelled data and reports accuracy,
Cohen's kappa and the confusion matrix. Unparseable judge outputs count as an `abstain` label so they
lower the scores instead of disappearing.

### Replay and diffs

`agentlens replay traces.jsonl --model new-model` re-runs every recorded input. Tool calls are
answered from the recording (same tool + same arguments returns the recorded result), so the replay is
deterministic and needs no live services; with `--suite` the suite's live tools and expectations are
used instead. The diff report lists per-case regressions, improvements and changed trajectories, and
metric deltas. `--fail-on-regression` makes it a CI check.

`agentlens eval suite.yaml --from-traces traces.jsonl` re-scores recorded runs without calling a
model (useful after fixing a test case).

### CI gate

```toml
# gate.toml
[thresholds]
task_success = { min = 0.90 }
tool_f1 = { min = 0.95 }
cost_per_task_usd = { max = 0.005 }

[baseline]
path = "baseline.json"
max_drop = { task_success = 0.0, args_exact = 0.02 }
max_increase = { avg_steps = 0.5 }
```

`agentlens gate results.json --config gate.toml` exits 0 (pass), 1 (a check failed) or 2 (config
error), and appends a Markdown table to `$GITHUB_STEP_SUMMARY` when it runs in GitHub Actions.

### CLI

```text
agentlens run "prompt" --toolset tools.py:TOOLSET --model M --base-url URL [--trace t.jsonl]
agentlens eval suite.yaml [agent flags] [--out r.json --report r.md --html r.html --trace t.jsonl|t.db --otlp-endpoint URL]
agentlens report r.json [--compare base.json] [--format md|html] | report --traces t.jsonl
agentlens gate r.json --config gate.toml [--baseline base.json]
agentlens replay t.jsonl [agent flags] [--suite suite.yaml] [--fail-on-regression]
agentlens judge labels.jsonl --mode pairwise|pointwise --model M --base-url URL
agentlens serve t.jsonl [--port 8000]          # needs the [server] extra
agentlens bfcl simple_python --dest data/bfcl   # download + convert a BFCL v4 category
```

## Measured results

One small open model on a shared CPU, to show the harness working end to end on real data. These are
**agentlens metrics computed on BFCL data, not official BFCL scores**: agentlens converts BFCL v4
cases into its own suite format (`agentlens bfcl <category>`) and scores them with its own matchers.

Setup: Qwen2.5-0.5B-Instruct, Q4_K_M GGUF, served by llama.cpp (`llama-server --jinja -c 8192 -t 2`)
on a 2-vCPU box shared with other jobs, temperature 0, driven by the built-in OpenAI-compatible tool
loop. Latency is therefore dominated by CPU contention and says nothing about the model or the tool.
Raw results, traces and per-case failures are in [`results/`](results/) (`results/SUMMARY.md` is
generated by `scripts/summarize_results.py`); `scripts/run_benchmark.sh` reproduces the runs.

| Suite | Cases | Task success | Tool F1 | Args exact | Tokens / task | p50 latency |
|---|---|---|---|---|---|---|
| travel-desk (bundled, multi-step) | 20 | 60% | 0.808 | 65% | 1,550 | 17.4 s |
| BFCL v4 `simple_python` (first 100) | 100 | 82% | 1.000 | 82% | 306 | 17.4 s |
| BFCL v4 `multiple` (first 50) | 50 | 78% | 0.960 | 78% | 497 | 29.4 s |
| BFCL v4 `irrelevance` (first 50) | 50 | 34% | n/a | n/a | 343 | 16.3 s |

How to read it:

- **Subsets, not full categories.** The first 100 / 50 / 50 cases of each BFCL category, one run each,
  no confidence intervals. With n = 50, one case is 2 points.
- **Re-scored from traces.** The two BFCL tool-call suites were first scored before a fix for BFCL
  answers that contain nested values (dicts, lists of dicts). They were re-scored from the recorded
  traces with `agentlens eval ... --from-traces`, without calling the model again; that moved
  `simple_python` from 80% to 82% (one case was also re-run after a transient server error) and
  `multiple` from 74% to 78%. travel-desk re-scores identically.
- **Irrelevance** cases offer tools that cannot answer the question; success means the model called no
  tool. The model called a tool anyway in 33 of 50 cases. Tool F1 and argument scores are vacuous here
  (there are no expected calls), so they are shown as n/a.
- **travel-desk** is a 20-case multi-step suite with live (offline) tools. The typical 0.5B failure is
  skipping the search step and booking with an invented flight id, or passing currency names instead
  of ISO codes; see `results/travel_desk_qwen0.5b.md`.
- Cost is $0: a local model has no price in the pricing table, so cost is reported as unknown.

### Judge calibration

The same 0.5B model as a pairwise judge (MT-Bench prompt, `[[A]]` / `[[B]]` / `[[C]]` verdicts, no
order swap) against expert human votes from
[lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
(CC-BY-4.0), built by `scripts/prepare_mtbench.py`.

**n = 120, accuracy = 0.333, Cohen's kappa = 0.023, abstentions = 11.**

| human \ judge | A | B | tie | abstain |
|---|---|---|---|---|
| A (43) | 0 | 0 | 39 | 4 |
| B (35) | 0 | 0 | 30 | 5 |
| tie (42) | 0 | 0 | 40 | 2 |

The 0.5B judge is useless: it answered "tie" on 109 of 120 pairs and never picked a side, so its
accuracy is just the share of human ties and kappa is about zero. A spot check of four raw outputs
showed the model replying with a bare `[[C]]` and no explanation (and once with `[C]`, which is
unparseable and counted as an abstention), so this is the model's behaviour, not a parsing artefact.
The result is here because it is what calibration is for: it tells you not to trust this judge. For
reference, the GPT-4 pairwise verdicts shipped in the same dataset cover only 51 of the 120 pairs; on
those 51 they reach accuracy 0.745 and kappa 0.582 against the human votes.

**Selection bias.** The 120 pairs are turn-1 votes only, restricted to pairs where both answers are at
most 1,200 characters (to keep CPU judging affordable), de-duplicated to one vote per (question, model
pair) and sampled with a fixed seed. Short answers are not representative of MT-Bench as a whole, and
the GPT-4 reference above is on an even smaller, non-random subset.

## Architecture

```text
            your agent / built-in ToolAgent / LangGraph (callback adapter)
                                   |
                             Tracer (spans)  ----> OTLP/HTTP collector (Jaeger, Tempo, Phoenix, ...)
                                   |
                     JSONL / SQLite trace store  ---->  trace viewer (FastAPI)
                                   |
      suite.yaml ---> harness: score_case (metrics, trajectory, constraints, answer checks, judge)
                                   |
                            results.json ----> report (Markdown / HTML)
                               |      \
                 replay (diff vs new agent)   gate (thresholds + baseline) ---> exit code for CI
```

| Module | Role |
|---|---|
| `tracer.py`, `exporters.py`, `store.py`, `cost.py` | spans, exporters, trace store, pricing |
| `metrics.py`, `trajectory.py` | pure scoring functions (mutation-tested) |
| `cases.py`, `harness.py` | suite schema, eval runner, aggregation |
| `llm.py`, `agent.py` | OpenAI-compatible client (stdlib only), mock model, tool loop, toolsets |
| `judge.py`, `calibration.py` | rubric / pairwise judges, kappa and accuracy |
| `replay.py`, `gate.py`, `report.py` | diffs, CI gate, reports |
| `server.py`, `adapters/langgraph.py`, `benchmarks/bfcl.py` | viewer, LangGraph adapter, BFCL converter |

## Quality

QUALITY_PLACEHOLDER

## Limitations

- **One small model.** The only real-model numbers are from Qwen2.5-0.5B (Q4_K_M) on a shared CPU.
  Larger local models and hosted APIs (OpenAI, Anthropic through an OpenAI-compatible proxy, vLLM)
  have not been benchmarked; the client speaks the OpenAI chat-completions protocol and is tested
  against a local fake server and llama.cpp only.
- **Not official BFCL.** agentlens converts BFCL v4 cases and scores them with its own matchers (AST
  checker semantics are approximated by argument matchers); only the `simple_python`, `multiple` and
  `irrelevance` categories were run, on their first 100 / 50 / 50 cases. The `parallel` converter
  exists but was not benchmarked. Do not compare these numbers with the BFCL leaderboard.
- **Judge calibration is small and biased.** 120 turn-1 MT-Bench pairs with short answers (both
  ≤ 1,200 characters), one judge model, no order swap. The pointwise `RubricJudge` has unit tests but
  no human-labelled calibration run.
- **OTLP exporter not tested against a real collector.** It is tested against a local HTTP server that
  checks the OTLP/HTTP JSON payload shape, headers and batching; it has not been run against an
  OpenTelemetry Collector, Jaeger, Tempo, Phoenix or Langfuse.
- **Docker image not built.** The `Dockerfile` has not been built or run (no Docker on the build
  machine).
- **GitHub workflows never run.** `.github/workflows/ci.yml` and `release.yml` have not run on GitHub
  Actions; the commands they call (pytest, ruff, mypy, the mock quickstart and gate) pass locally on
  Python 3.12 only. Python 3.10 and 3.11 are declared but untested.
- **Viewer** (`agentlens serve`) is covered by FastAPI test-client tests and a manual smoke check, not
  by browser tests.
- **Cost** is only as good as `pricing.toml`: prices change, and unknown models are reported as unknown
  cost rather than guessed.

## Development

```bash
pip install -e ".[dev,langgraph]"
make test lint typecheck     # pytest, ruff, mypy
make mutation                # testteeth on the core modules
make bench                   # real-model benchmark (see scripts/run_benchmark.sh)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT © Fernando Garza
