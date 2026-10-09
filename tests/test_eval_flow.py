import json

import pytest

from agentlens import EvalResult, JSONLExporter, MockChatClient, ToolAgent, load_suite, load_traces, run_eval
from agentlens.calibration import ABSTAIN, accuracy, calibrate, cohen_kappa, confusion_matrix
from agentlens.cases import suite_from_dict
from agentlens.harness import CaseResult, score_case, summarize
from agentlens.judge import PairwiseJudge, Rubric, RubricJudge, parse_score, parse_winner
from agentlens.llm import ChatResponse, LLMError, MockRule
from agentlens.metrics import ToolCall
from agentlens.replay import RecordedToolset, diff_results, replay
from agentlens.report import diff_html, diff_markdown, eval_html, eval_markdown, gate_markdown, traces_markdown
from agentlens.tracer import RunSummary, Tracer

SUITE = "examples/travel_desk/suite.yaml"


def mock_agent(version="v1"):
    return ToolAgent(MockChatClient.from_yaml(f"examples/travel_desk/mock_{version}.yaml"), model=f"mock-{version}")


@pytest.fixture(scope="module")
def v1_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("v1")
    trace = d / "v1.jsonl"
    result = run_eval(
        load_suite(SUITE), mock_agent("v1"), agent_label="v1", model="mock-v1", exporters=[JSONLExporter(trace)]
    )
    return result, trace


# ---- score_case / summarize ----------------------------------------------------------------
def _case(**kw):
    base = {
        "id": "c",
        "input": "q",
        "expected_tool_calls": [{"name": "a", "arguments": {"x": 1}}],
        "trajectory": "strict",
    }
    base.update(kw)
    return suite_from_dict([base]).cases[0]


def _summary(**kw):
    s = RunSummary(trace_id="t", steps=1, input_tokens=10, output_tokens=5, cost_usd=0.01, latency_s=1.0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_score_case_success_and_failure_reasons():
    ok = score_case(_case(), [ToolCall("a", {"x": 1})], "done", _summary())
    assert ok.success and ok.tool_f1 == 1.0 and ok.args_exact == 1.0 and ok.trajectory_match
    assert (ok.input_tokens, ok.output_tokens, ok.total_tokens, ok.cost_usd, ok.latency_s) == (10, 5, 15, 0.01, 1.0)
    wrong_args = score_case(_case(), [ToolCall("a", {"x": 2})], "done", _summary())
    assert not wrong_args.success and wrong_args.trajectory_match and wrong_args.args_exact == 0.0
    errored = score_case(_case(), [ToolCall("a", {"x": 1})], None, _summary(), error="boom")
    assert not errored.success and errored.error == "boom"
    answer = score_case(_case(final_answer={"contains": ["42"]}), [ToolCall("a", {"x": 1})], "41", _summary())
    assert not answer.success and not answer.answer_ok and answer.answer_failures == ["missing '42'"]
    violated = score_case(_case(constraints={"max_steps": 1}), [ToolCall("a", {"x": 1})], "x", _summary(steps=2))
    assert not violated.success and not violated.constraints_ok and violated.violations
    no_summary = score_case(_case(), [ToolCall("a", {"x": 1})], "x", None)
    assert no_summary.steps == 0 and not no_summary.cost_known and no_summary.trace_id == ""


def test_score_case_judge_gate():
    case = _case(final_answer={"judge": True})
    passed = score_case(case, [ToolCall("a", {"x": 1})], "x", _summary(), judge_score=5, judge_pass=True)
    assert passed.success and passed.judge_score == 5
    failed = score_case(case, [ToolCall("a", {"x": 1})], "x", _summary(), judge_score=2, judge_pass=False)
    assert not failed.success and "judge score 2" in failed.answer_failures[0]
    skipped = score_case(case, [ToolCall("a", {"x": 1})], "x", _summary())
    assert skipped.success


def test_summarize_aggregates():
    a = score_case(_case(), [ToolCall("a", {"x": 1})], "x", _summary(latency_s=1.0, cost_usd=0.01))
    b = score_case(_case(), [ToolCall("a", {"x": 2}), ToolCall("b")], "x", _summary(latency_s=3.0, cost_usd=0.03))
    s = summarize([a, b])
    assert s["n_cases"] == 2 and s["task_success"] == 0.5
    assert s["tool_precision"] == pytest.approx(0.75) and s["tool_recall"] == 1.0
    assert s["tool_f1_micro"] == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    assert s["args_exact"] == 0.5 and s["latency_p50_s"] == 2.0 and s["latency_mean_s"] == 2.0
    assert s["cost_per_task_usd"] == pytest.approx(0.02) and s["total_cost_usd"] == pytest.approx(0.04)
    assert s["total_tokens"] == 30 and s["avg_tokens"] == 15 and s["cost_known"] is True
    assert "judge_score_mean" not in s
    assert summarize([]) == {"n_cases": 0}
    a.judge_score = 4.0
    assert summarize([a, b])["judge_score_mean"] == 4.0


def test_case_result_round_trip():
    r = score_case(_case(), [ToolCall("a", {"x": 1})], "x", _summary())
    assert CaseResult.from_dict({**r.to_dict(), "unknown_future_field": 1}) == r


# ---- run_eval -------------------------------------------------------------------------------
def test_run_eval_mock_v1(v1_run, tmp_path):
    result, trace = v1_run
    s = result.summary
    assert s["n_cases"] == 20 and s["task_success"] == 0.95 and s["error_rate"] == 0.0
    assert s["cost_known"] and s["total_cost_usd"] > 0
    failed = [c.case_id for c in result.cases if not c.success]
    assert failed == ["price-gap-hnd"]
    booking = next(c for c in result.cases if c.case_id == "book-cheapest-mnl-sin")
    assert [c["name"] for c in booking.tool_calls] == ["search_flights", "book_flight"] and booking.steps == 3
    out = tmp_path / "r.json"
    result.save(out)
    again = EvalResult.load(out)
    assert again.summary == result.summary and again.agent == "v1" and again.model == "mock-v1"
    traces = load_traces(trace)
    assert len(traces) == 20 and traces[0].case_id == "weather-paris"
    assert traces[0].root.attributes["agentlens.tools"][0]["type"] == "function"


def test_run_eval_concurrency_and_crashing_agent():
    suite = load_suite(SUITE).filter(limit=4)
    par = run_eval(suite, mock_agent("v1"), concurrency=2)
    assert [c.case_id for c in par.cases] == [c.id for c in suite.cases] and par.summary["task_success"] == 1.0

    class Crash:
        def run(self, *a, **k):
            raise RuntimeError("kaput")

    seen = []
    crashed = run_eval(suite, Crash(), on_case=seen.append)
    assert all(not c.success and "kaput" in c.error for c in crashed.cases) and len(seen) == 4


def test_run_eval_canned_tools_and_judge():
    suite = suite_from_dict(
        {
            "tools": [
                {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}
            ],
            "tool_results": {"lookup": {"answer": 42}},
            "cases": [
                {
                    "id": "a",
                    "input": "look it up",
                    "expected_tool_calls": [{"name": "lookup"}],
                    "final_answer": {"contains": ["42"], "judge": True},
                },
                {"id": "b", "input": "chat", "final_answer": {"judge": True}},
            ],
        }
    )
    client = MockChatClient(
        [
            MockRule("look", [{"tool_calls": [{"name": "lookup"}]}, {"content": "It is 42."}]),
            MockRule("chat", [{"content": "hi"}]),
        ]
    )
    judge = RubricJudge(
        MockChatClient([MockRule(".", [{"content": '{"reasoning": "fine", "score": 5}'}])]), "mock-judge"
    )
    result = run_eval(suite, ToolAgent(client, "mock-a"), judge=judge)
    a, b = result.cases
    assert a.success and a.judge_score == 5 and a.judge_pass
    assert b.judge_score == 5 and result.summary["judge_score_mean"] == 5


# ---- replay -------------------------------------------------------------------------------
def test_replay_against_recording_flags_v2_regressions(v1_run):
    _, trace = v1_run
    baseline, candidate, diff = replay(load_traces(trace), mock_agent("v2"), agent_label="v2", model="mock-v2")
    assert baseline.summary["task_success"] == 1.0  # the recording is its own reference
    regressed = {c.case_id for c in diff.regressions}
    assert regressed == {"weather-tokyo-f", "book-cheapest-mnl-sin", "price-gap-hnd"}
    assert len(diff.cases) == 20
    assert diff.deltas()["task_success"] == pytest.approx(-0.15)


def test_replay_same_version_is_unchanged(v1_run):
    _, trace = v1_run
    _, _, diff = replay(load_traces(trace), mock_agent("v1"))
    assert {c.status for c in diff.cases} == {"unchanged"}


def test_replay_with_suite_scores_against_expectations(v1_run):
    _, trace = v1_run
    baseline, candidate, diff = replay(load_traces(trace), mock_agent("v2"), suite=load_suite(SUITE))
    assert baseline.summary["task_success"] == 0.95 and candidate.summary["task_success"] == 0.9
    assert {c.case_id for c in diff.improvements} == {"price-gap-hnd"}
    assert {c.case_id for c in diff.regressions} == {"weather-tokyo-f", "book-cheapest-mnl-sin"}


def test_replay_requires_replayable_traces():
    with pytest.raises(ValueError, match="no replayable"):
        replay([], mock_agent("v1"))


def test_recorded_toolset():
    ts = RecordedToolset([{"name": "f"}], [("f", {"a": 1}, {"r": 1}), ("f", {"a": 2}, {"r": 2})])
    assert ts.call("f", {"a": 2}) == {"r": 2}
    assert "no recorded result" in ts.call("f", {"a": 3})["error"] and ts.misses == 1
    assert "no recorded result" in ts.call("g", {"a": 1})["error"]


def test_diff_added_removed_and_changed(v1_run):
    result, _ = v1_run
    other = EvalResult.from_dict(result.to_dict())
    other.cases = other.cases[1:]
    extra = CaseResult.from_dict({**other.cases[0].to_dict(), "case_id": "new-case"})
    changed = CaseResult.from_dict({**other.cases[1].to_dict(), "final_answer": "different words"})
    other.cases = [other.cases[0], changed, *other.cases[2:], extra]
    diff = diff_results(result, other)
    statuses = {c.case_id: c.status for c in diff.cases}
    assert statuses["weather-paris"] == "removed" and statuses["new-case"] == "added"
    assert statuses[changed.case_id] == "changed"
    d = diff.to_dict()
    assert d["counts"]["added"] == 1 and d["counts"]["removed"] == 1 and d["counts"]["changed"] == 1


# ---- reports -----------------------------------------------------------------------------
def test_reports_render(v1_run):
    result, trace = v1_run
    md = eval_markdown(result)
    assert "| Task success | 95.0% |" in md and "price-gap-hnd" in md and "Failed cases (1/20)" in md
    html = eval_html(result)
    assert html.startswith("<!doctype html>") and "price-gap-hnd" in html and "<script" not in html
    _, _, diff = replay(load_traces(trace), mock_agent("v2"))
    dmd = diff_markdown(diff)
    assert "3 regressions" in dmd and "-15.0 pp" in dmd
    assert "regression" in diff_html(diff)
    assert "case weather-paris" in traces_markdown(load_traces(trace))


def test_report_escapes_html():
    r = score_case(_case(id="<img src=x>"), [], "<b>", _summary())
    html = eval_html(EvalResult("s<", "a", "m", [r]))
    assert "<img src=x>" not in html and "&lt;img" in html


def test_gate_markdown():
    from agentlens.gate import GateConfig, evaluate_gate

    md = gate_markdown(evaluate_gate({"tool_f1": 0.5}, GateConfig(minimums={"tool_f1": 0.9})))
    assert "FAILED" in md and "**FAIL**" in md


# ---- judge ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,score",
    [
        ('{"reasoning": "ok", "score": 4}', 4.0),
        ('Sure! {"score": "3", "reasoning": "meh"} done', 3.0),
        ("Rating: [[5]]", 5.0),
        ("score: 2", 2.0),
        ("I'd give it 4/5", 4.0),
        ('{"score": 9}', None),
        ("no idea", None),
        ('{"score": "high"}', None),
    ],
)
def test_parse_score(text, score):
    assert parse_score(text, 1, 5)[0] == score


def test_parse_score_reasoning():
    assert parse_score('{"reasoning": "solid", "score": 4}', 1, 5) == (4.0, "solid")


@pytest.mark.parametrize(
    "text,winner",
    [
        ("Explanation... [[A]]", "A"),
        ("[[B]]", "B"),
        ("tie [[C]]", "tie"),
        ("[[A]] then [[B]]", "B"),
        ("blah\nB", "B"),
        ("Verdict: A.", "A"),
        ("", None),
        ("I prefer the first", None),
    ],
)
def test_parse_winner(text, winner):
    assert parse_winner(text) == winner


def test_rubric_validation_and_load(tmp_path):
    with pytest.raises(ValueError):
        Rubric(scale_min=5, scale_max=1)
    with pytest.raises(ValueError):
        Rubric(pass_threshold=9)
    with pytest.raises(ValueError, match="unknown rubric"):
        Rubric.from_dict({"nme": "x"})
    f = tmp_path / "r.yaml"
    f.write_text("name: strict\ncriteria: [Correct]\nscale_min: 0\nscale_max: 10\npass_threshold: 7\n")
    r = Rubric.load(f)
    assert r.name == "strict" and "- Correct" in r.render()


def test_rubric_judge_scores_and_traces():
    client = MockChatClient([MockRule(".", [{"content": '{"reasoning": "good", "score": 4}'}])])
    tracer = Tracer()
    from agentlens import InMemoryExporter

    mem = InMemoryExporter()
    tracer.add_exporter(mem)
    v = RubricJudge(client, "mock-j", tracer=tracer).score("q", "a", reference="ref")
    assert (v.score, v.passed, v.reasoning) == (4.0, True, "good")
    assert mem.spans[0].attributes["agentlens.judge"] is True and mem.spans[0].kind == "llm"
    low = RubricJudge(MockChatClient([MockRule(".", [{"content": "score: 2"}])]), "m").score("q", "a")
    assert low.passed is False
    junk = RubricJudge(MockChatClient([MockRule(".", [{"content": "??"}])]), "m").score("q", "a")
    assert junk.score is None and junk.passed is None
    system, user = RubricJudge(client, "m").prompt("Q?", "A!", "R.")
    assert "Reference answer" in user and "JSON" in system


class _Seq:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, model, **kw):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResponse(r)


def test_pairwise_judge_with_and_without_swap():
    assert PairwiseJudge(_Seq(["[[A]]"]), "m").compare("q", "a", "b").winner == "A"
    # swapped order: second reply "[[B]]" means the original A won again -> consistent
    assert PairwiseJudge(_Seq(["[[A]]", "[[B]]"]), "m", swap=True).compare("q", "a", "b").winner == "A"
    assert PairwiseJudge(_Seq(["[[A]]", "[[A]]"]), "m", swap=True).compare("q", "a", "b").winner == "tie"
    assert PairwiseJudge(_Seq(["[[C]]", "[[C]]"]), "m", swap=True).compare("q", "a", "b").winner == "tie"
    assert PairwiseJudge(_Seq(["???", "[[A]]"]), "m", swap=True).compare("q", "a", "b").winner == "B"
    assert PairwiseJudge(_Seq(["[[B]]", "???"]), "m", swap=True).compare("q", "a", "b").winner == "B"
    assert PairwiseJudge(_Seq(["?", "?"]), "m", swap=True).compare("q", "a", "b").winner is None
    assert PairwiseJudge(_Seq([LLMError("x")]), "m").compare("q", "a", "b").winner is None
    assert PairwiseJudge(_Seq(["[[A]]", LLMError("x")]), "m", swap=True).compare("q", "a", "b").winner is None
    assert RubricJudge(_Seq([LLMError("down")]), "m").score("q", "a").passed is None


# ---- calibration -------------------------------------------------------------------------------
def test_cohen_kappa_known_values():
    # Classic example: 50 items, two raters, yes/no. p_o = 0.7, p_e = 0.5 -> kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert accuracy(a, b) == pytest.approx(0.7)
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    assert cohen_kappa(["x", "y"], ["y", "x"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        accuracy(["a"], [])
    assert accuracy([], []) == 0.0


def test_confusion_and_calibrate():
    m = confusion_matrix(["A", "B", "B"], ["A", "A", "B"])
    assert m == {"A": {"A": 1, "B": 0}, "B": {"A": 1, "B": 1}}
    with pytest.raises(ValueError):
        confusion_matrix(["A"], [])
    rep = calibrate(["A", None, "tie", "B"], ["A", "B", "tie", "A"], labels=["A", "B", "tie"])
    assert rep.n == 4 and rep.abstentions == 1 and rep.accuracy == 0.5
    assert ABSTAIN in rep.labels and rep.confusion["B"][ABSTAIN] == 1
    assert rep.human_distribution == {"A": 2, "B": 1, "tie": 1}
    assert json.loads(json.dumps(rep.to_dict()))["n"] == 4
    with pytest.raises(ValueError):
        calibrate(["A"], [])
