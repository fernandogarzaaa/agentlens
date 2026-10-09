# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

## [0.1.0] - unreleased

### Added
- Tracer: OpenTelemetry-compatible spans (W3C ids, GenAI semantic-convention attributes) for runs,
  agent steps, LLM calls and tool calls; context-manager and decorator APIs (sync and async);
  per-run token and cost accounting from a TOML pricing table (exact names, aliases, globs, GGUF paths).
- Exporters: in-memory, JSONL, SQLite and OTLP/HTTP (JSON); trace store reader.
- Eval harness: YAML/JSON/JSONL suites with expected tool calls and argument matchers (`$any`,
  `$absent`, `$regex`), final-answer checks and trajectory constraints. Metrics: tool-selection
  precision/recall/F1 (macro and micro), argument exact and partial match, trajectory match
  (strict, unordered, subset, subsequence), task success, steps, tokens, $ per task, latency p50/p95.
- Built-in OpenAI-compatible function-calling agent loop, offline mock model, Python and canned toolsets.
- LLM-as-judge (pointwise rubric and pairwise with optional order swap) and calibration
  (Cohen's kappa, accuracy, confusion matrix) against human labels.
- Replay of recorded traces against a new agent version with a regression diff (Markdown/HTML/JSON).
- CI regression gate with absolute thresholds and baseline-relative limits.
- CLI: `agentlens run | eval | report | gate | replay | judge | serve | bfcl`.
- FastAPI trace viewer, LangGraph/LangChain callback adapter, BFCL v4 converter.
- Measured benchmarks with Qwen2.5-0.5B-Instruct (Q4_K_M, llama.cpp, CPU) in `results/`.

### Fixed
- `src/agentlens/data/pricing.toml` was matched by the `data/` rule in `.gitignore` and missing from
  git, so a fresh clone could not load the default pricing table; the rule now only ignores the
  top-level `data/` folder.
