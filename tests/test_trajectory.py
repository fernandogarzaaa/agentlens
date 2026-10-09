import pytest

from agentlens.metrics import ToolCall
from agentlens.trajectory import MODES, Constraints, check_constraints, match_trajectory


def calls(*names, **args):
    return [ToolCall(n, dict(args)) for n in names]


@pytest.mark.parametrize(
    "mode,expected,actual,ok",
    [
        ("strict", ["a", "b"], ["a", "b"], True),
        ("strict", ["a", "b"], ["b", "a"], False),
        ("strict", ["a"], ["a", "a"], False),
        ("strict", [], [], True),
        ("strict", [], ["a"], False),
        ("unordered", ["a", "b"], ["b", "a"], True),
        ("unordered", ["a", "a"], ["a"], False),
        ("unordered", ["a"], ["a", "a"], False),
        ("unordered", ["a"], ["a", "b"], False),
        ("subset", ["a"], ["b", "a", "c"], True),
        ("subset", ["a", "a"], ["a", "b"], False),
        ("subset", ["a", "a"], ["a", "b", "a"], True),
        ("subset", [], ["x"], True),
        ("subset", ["a"], [], False),
        ("subsequence", ["a", "c"], ["a", "b", "c"], True),
        ("subsequence", ["c", "a"], ["a", "b", "c"], False),
        ("subsequence", ["a", "a"], ["a", "b"], False),
        ("subsequence", ["a", "a"], ["a", "b", "a"], True),
        ("subsequence", [], ["a"], True),
        ("subsequence", ["a"], [], False),
        ("none", ["a"], ["zzz"], True),
    ],
)
def test_match_trajectory(mode, expected, actual, ok):
    assert match_trajectory(expected, actual, mode) is ok


def test_match_trajectory_default_mode_is_strict():
    assert match_trajectory(["a", "b"], ["a", "b"])
    assert not match_trajectory(["a", "b"], ["b", "a"])


def test_unknown_mode():
    with pytest.raises(ValueError, match="unknown trajectory mode"):
        match_trajectory([], [], "fuzzy")
    assert set(MODES) == {"strict", "unordered", "subset", "subsequence", "none"}


def test_strict_accepts_tuples():
    assert match_trajectory(("a", "b"), ["a", "b"], "strict")


# ---- Constraints.from_dict / merged_over -------------------------------------------------
def test_constraints_from_dict_full():
    c = Constraints.from_dict(
        {
            "max_steps": "3",
            "max_tool_calls": 2,
            "must_call": ["a"],
            "must_not_call": ["b"],
            "order": [["a", "b"]],
            "no_repeat": True,
        }
    )
    assert c == Constraints(3, 2, ["a"], ["b"], [("a", "b")], True)


def test_constraints_from_dict_empty_and_defaults():
    assert Constraints.from_dict(None) == Constraints()
    assert Constraints.from_dict({}) == Constraints()
    c = Constraints.from_dict({"must_call": None, "order": None})
    assert c.must_call == [] and c.order == [] and c.max_steps is None and c.no_repeat is False


def test_constraints_from_dict_rejects_unknown_and_bad_order():
    with pytest.raises(ValueError, match="unknown constraints"):
        Constraints.from_dict({"max_stepz": 1})
    with pytest.raises(ValueError, match="pairs"):
        Constraints.from_dict({"order": [["a", "b", "c"]]})
    with pytest.raises(ValueError, match="pairs"):
        Constraints.from_dict({"order": ["ab"]})


def test_merged_over_prefers_case_values():
    base = Constraints(
        max_steps=5, max_tool_calls=4, must_call=["x"], must_not_call=["y"], order=[("p", "q")], no_repeat=True
    )
    case = Constraints(max_steps=2, must_not_call=["z"])
    m = case.merged_over(base)
    assert m.max_steps == 2
    assert m.max_tool_calls == 4
    assert m.must_call == ["x"]
    assert m.must_not_call == ["z"]
    assert m.order == [("p", "q")]
    assert m.no_repeat is True
    full = Constraints(1, 1, ["a"], ["b"], [("c", "d")], False).merged_over(base)
    assert full == Constraints(1, 1, ["a"], ["b"], [("c", "d")], True)
    assert Constraints().merged_over(Constraints()) == Constraints()
    assert Constraints(max_steps=0).merged_over(base).max_steps == 0
    assert Constraints(max_tool_calls=0).merged_over(base).max_tool_calls == 0


# ---- check_constraints ------------------------------------------------------------------
def test_no_constraints_no_violations():
    assert check_constraints(Constraints(), calls("a", "a"), steps=99) == []


def test_max_steps_boundary():
    c = Constraints(max_steps=3)
    assert check_constraints(c, [], steps=3) == []
    assert check_constraints(c, [], steps=4) == ["4 steps > max_steps 3"]
    assert check_constraints(Constraints(max_steps=0), [], steps=0) == []
    assert check_constraints(Constraints(max_steps=0), [], steps=1) == ["1 steps > max_steps 0"]


def test_max_tool_calls_boundary():
    c = Constraints(max_tool_calls=2)
    assert check_constraints(c, calls("a", "b"), 1) == []
    assert check_constraints(c, calls("a", "b", "c"), 1) == ["3 tool calls > max_tool_calls 2"]
    assert check_constraints(Constraints(max_tool_calls=0), calls("a"), 1) == ["1 tool calls > max_tool_calls 0"]
    assert check_constraints(Constraints(max_tool_calls=0), [], 1) == []


def test_must_call_and_must_not_call():
    c = Constraints(must_call=["search", "book"], must_not_call=["cancel"])
    assert check_constraints(c, calls("search", "book"), 2) == []
    v = check_constraints(c, calls("search", "cancel"), 2)
    assert v == ["never called required tool 'book'", "called forbidden tool 'cancel'"]


def test_order_constraint():
    c = Constraints(order=[("search", "book")])
    assert check_constraints(c, calls("search", "book"), 2) == []
    assert check_constraints(c, calls("other"), 1) == []  # 'book' never called: nothing to order
    assert check_constraints(c, calls("search"), 1) == []
    assert check_constraints(c, calls("book"), 1) == ["'book' called before 'search'"]
    assert check_constraints(c, calls("book", "search"), 2) == ["'book' called before 'search'"]
    assert check_constraints(c, calls("search", "book", "search"), 3) == []


def test_order_uses_first_occurrences():
    c = Constraints(order=[("a", "b")])
    assert check_constraints(c, calls("a", "x", "b"), 3) == []
    assert check_constraints(c, calls("b", "a", "b"), 3) == ["'b' called before 'a'"]


def test_no_repeat_detects_identical_calls_only():
    c = Constraints(no_repeat=True)
    assert check_constraints(c, [ToolCall("a", {"x": 1}), ToolCall("a", {"x": 2})], 2) == []
    v = check_constraints(c, [ToolCall("a", {"x": 1, "y": 2}), ToolCall("a", {"y": 2, "x": 1})], 2)
    assert v == ["repeated identical call to 'a'"]
    triple = [ToolCall("a", {}), ToolCall("a", {}), ToolCall("a", {})]
    assert check_constraints(c, triple, 3) == ["repeated identical call to 'a'"]
    assert check_constraints(c, [ToolCall("a", {}), ToolCall("b", {})], 2) == []
    assert check_constraints(Constraints(no_repeat=False), triple, 3) == []


def test_all_violations_collected():
    c = Constraints(
        max_steps=1, max_tool_calls=1, must_call=["z"], must_not_call=["a"], order=[("q", "a")], no_repeat=True
    )
    v = check_constraints(c, calls("a", "a"), 2)
    assert len(v) == 6


def test_order_constraint_with_same_tool_on_both_sides_never_fires():
    c = Constraints(order=[("a", "a")])
    assert check_constraints(c, calls("a"), 1) == []
    assert check_constraints(c, calls("b", "a"), 2) == []
