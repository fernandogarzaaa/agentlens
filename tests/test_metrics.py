import math

import pytest

from agentlens.metrics import (
    ABSENT,
    AnswerCheck,
    ToolCall,
    argument_match,
    check_answer,
    match_calls,
    mean,
    percentile,
    score_arguments,
    tool_selection,
    values_equal,
)


def tc(name, **args):
    return ToolCall(name, args)


# ---- ToolCall -------------------------------------------------------------------------
def test_toolcall_from_dict_accepts_arguments_or_args():
    assert ToolCall.from_dict({"name": "a", "arguments": {"x": 1}}) == tc("a", x=1)
    assert ToolCall.from_dict({"name": "a", "args": {"x": 1}}) == tc("a", x=1)
    assert ToolCall.from_dict({"name": "a"}) == ToolCall("a", {})
    assert ToolCall.from_dict({"name": "a", "arguments": None}).arguments == {}


def test_toolcall_from_dict_rejects_bad_input():
    with pytest.raises(ValueError, match="name"):
        ToolCall.from_dict({"arguments": {}})
    with pytest.raises(ValueError, match="mapping"):
        ToolCall.from_dict({"name": "a", "arguments": [1, 2]})


def test_toolcall_round_trip():
    call = tc("search", q="x", n=3)
    assert ToolCall.from_dict(call.to_dict()) == call
    assert call.to_dict() == {"name": "search", "arguments": {"q": "x", "n": 3}}


# ---- values_equal ---------------------------------------------------------------------
def test_strings_normalised_by_default():
    assert values_equal("New  York", " new york ")
    assert values_equal("Straße", "STRASSE")  # casefold, not lower
    assert not values_equal("New York", "Newark")


def test_strings_strict_when_normalisation_off():
    assert not values_equal("Paris", "paris", normalize_strings=False)
    assert values_equal("Paris", "Paris", normalize_strings=False)


def test_string_never_equals_non_string():
    assert not values_equal("5", 5)
    assert not values_equal("None", None)


def test_numbers_compare_by_value():
    assert values_equal(2, 2.0)
    assert values_equal(0.1 + 0.2, 0.3)
    assert not values_equal(2, 2.001)
    assert not values_equal(2, "2")
    assert not values_equal(0, None)


def test_numbers_tiny_differences_tolerated_large_not():
    assert values_equal(1e-12, 0)  # abs_tol 1e-9
    assert not values_equal(1e-6, 0)
    assert values_equal(1e12, 1e12 + 1)  # rel_tol 1e-9
    assert not values_equal(1e3, 1e3 + 0.01)


def test_booleans_never_equal_numbers():
    assert values_equal(True, True)
    assert not values_equal(True, False)
    assert not values_equal(True, 1)
    assert not values_equal(1, True)
    assert not values_equal(False, 0)
    assert not values_equal(0, False)


def test_lists_elementwise_and_length():
    assert values_equal([1, "A"], [1.0, "a"])
    assert values_equal([1, 2], (1, 2))
    assert not values_equal([1, 2], [2, 1])
    assert not values_equal([1, 2], [1, 2, 3])
    assert not values_equal([1, 2], [1])
    assert not values_equal([1], "1")


def test_nested_dicts_use_argument_rules():
    assert values_equal({"a": 1, "b": {"c": "X"}}, {"a": 1.0, "b": {"c": "x"}})
    assert not values_equal({"a": 1}, {"a": 1, "extra": 2})
    assert not values_equal({"a": 1}, {"a": 2})
    assert not values_equal({"a": 1}, [("a", 1)])


def test_none_matches_only_none():
    assert values_equal(None, None)
    assert not values_equal(None, 0)
    assert not values_equal(None, "")


def test_any_matcher():
    spec = {"$any": ["celsius", "metric"]}
    assert values_equal(spec, "Celsius")
    assert values_equal(spec, "metric")
    assert not values_equal(spec, "kelvin")
    assert not values_equal({"$any": [ABSENT]}, ABSENT)  # $absent is not a value
    assert not values_equal({"$any": []}, "x")


def test_regex_matcher_is_fullmatch():
    assert values_equal({"$regex": r"\d+ / \d+"}, "540 / 4")
    assert not values_equal({"$regex": r"\d+"}, "540 / 4")
    assert values_equal({"$regex": r"\d+"}, 540)  # compared as a string
    assert not values_equal({"$regex": ".*"}, None)


def test_other_types_fall_back_to_equality():
    assert values_equal((1,), [1])
    assert values_equal(frozenset({1}), frozenset({1}))
    assert not values_equal(frozenset({1}), frozenset({2}))


def test_dict_with_extra_keys_is_not_a_matcher():
    assert values_equal({"$any": 1, "x": 2}, {"$any": 1, "x": 2})
    assert not values_equal({"$any": 1, "x": 2}, "x")
    assert values_equal({"$regex": "a", "x": 1}, {"$regex": "a", "x": 1})


# ---- score_arguments ------------------------------------------------------------------
def test_exact_argument_match():
    s = score_arguments({"city": "Paris", "unit": "c"}, {"city": "paris", "unit": "C"})
    assert s.exact and s.partial == 1.0
    assert s.matched == ["city", "unit"] and s.mismatched == [] and s.extra == []


def test_partial_argument_match():
    s = score_arguments({"a": 1, "b": 2, "c": 3, "d": 4}, {"a": 1, "b": 0, "c": 3})
    assert not s.exact
    assert s.partial == 0.5
    assert s.mismatched == ["b", "d"]


def test_extra_arguments_break_exact_unless_allowed():
    s = score_arguments({"a": 1}, {"a": 1, "b": 2})
    assert not s.exact and s.partial == 1.0 and s.extra == ["b"]
    assert score_arguments({"a": 1}, {"a": 1, "b": 2}, allow_extra=True).exact


def test_no_expected_arguments():
    assert score_arguments({}, {}).exact
    assert score_arguments({}, {}).partial == 1.0
    assert not score_arguments({}, {"x": 1}).exact
    assert score_arguments({}, {"x": 1}).partial == 1.0


def test_absent_and_optional_arguments():
    spec = {"x": 4, "z": {"$any": [ABSENT, 0]}}
    assert score_arguments(spec, {"x": 4}).exact
    assert score_arguments(spec, {"x": 4, "z": 0}).exact
    assert not score_arguments(spec, {"x": 4, "z": 1}).exact
    forbidden = {"x": 4, "debug": ABSENT}
    assert score_arguments(forbidden, {"x": 4}).exact
    s = score_arguments(forbidden, {"x": 4, "debug": True})
    assert not s.exact and s.mismatched == ["debug"] and s.partial == 0.5


def test_missing_required_argument():
    s = score_arguments({"x": 1, "y": 2}, {"x": 1})
    assert not s.exact and s.partial == 0.5 and s.mismatched == ["y"]


def test_string_normalisation_flag_reaches_arguments():
    assert score_arguments({"c": "Paris"}, {"c": "PARIS"}).exact
    assert not score_arguments({"c": "Paris"}, {"c": "PARIS"}, normalize_strings=False).exact


# ---- tool_selection -------------------------------------------------------------------
def test_tool_selection_perfect():
    r = tool_selection([tc("a"), tc("b")], [tc("b"), tc("a")])
    assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)
    assert (r.true_positives, r.n_expected, r.n_actual) == (2, 2, 2)


def test_tool_selection_multiset_counts_duplicates():
    r = tool_selection([tc("search")], [tc("search"), tc("search")])
    assert r.precision == 0.5 and r.recall == 1.0
    assert r.f1 == pytest.approx(2 / 3)
    assert r.true_positives == 1


def test_tool_selection_partial():
    r = tool_selection([tc("a"), tc("b"), tc("c")], [tc("a"), tc("x")])
    assert r.precision == 0.5
    assert r.recall == pytest.approx(1 / 3)
    assert r.f1 == pytest.approx(0.4)


def test_tool_selection_ignores_arguments():
    assert tool_selection([tc("a", x=1)], [tc("a", x=2)]).f1 == 1.0


def test_tool_selection_empty_cases():
    both = tool_selection([], [])
    assert (both.precision, both.recall, both.f1) == (1.0, 1.0, 1.0)
    missed = tool_selection([tc("a")], [])
    assert (missed.precision, missed.recall, missed.f1) == (0.0, 0.0, 0.0)
    spurious = tool_selection([], [tc("a")])
    assert (spurious.precision, spurious.recall, spurious.f1) == (0.0, 0.0, 0.0)
    wrong = tool_selection([tc("a")], [tc("b")])
    assert (wrong.precision, wrong.recall, wrong.f1) == (0.0, 0.0, 0.0)


def test_tool_selection_empty_actual_precision_is_zero_but_recall_one_when_nothing_expected():
    # precision divides by actual, recall by expected
    r = tool_selection([], [tc("a"), tc("b")])
    assert r.recall == 0.0 and r.precision == 0.0 and r.n_actual == 2 and r.n_expected == 0


# ---- match_calls / argument_match ---------------------------------------------------------
def test_match_calls_pairs_by_name_and_best_arguments():
    expected = [tc("w", city="Paris"), tc("w", city="London")]
    actual = [tc("w", city="London"), tc("w", city="Paris")]
    pairs = match_calls(expected, actual)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)]
    assert all(s is not None and s.exact for _, _, s in pairs)


def test_match_calls_prefers_exact_over_earlier_partial():
    expected = [tc("f", a=1, b=2)]
    actual = [tc("f", a=1, b=9), tc("f", a=1, b=2)]
    ((i, j, s),) = match_calls(expected, actual)
    assert j == 1 and s.exact


def test_match_calls_prefers_higher_partial_and_earliest_on_tie():
    expected = [tc("f", a=1, b=2, c=3)]
    actual = [tc("f", a=0, b=0, c=3), tc("f", a=1, b=2, c=0), tc("f", a=1, b=2, c=0)]
    ((_, j, s),) = match_calls(expected, actual)
    assert j == 1 and s.partial == pytest.approx(2 / 3)


def test_match_calls_exactness_beats_partial_with_extras():
    # actual[0] has every expected key right plus an extra (partial 1.0, not exact);
    # actual[1] is exact. Exactness wins.
    expected = [tc("f", a=1)]
    actual = [tc("f", a=1, z=0), tc("f", a=1)]
    ((_, j, s),) = match_calls(expected, actual)
    assert j == 1 and s.exact


def test_match_calls_unmatched_and_reuse():
    expected = [tc("a"), tc("a"), tc("b")]
    actual = [tc("a")]
    pairs = match_calls(expected, actual)
    assert pairs[0][1] == 0
    assert pairs[1][1] is None and pairs[1][2] is None  # actual call not reused
    assert pairs[2][1] is None


def test_argument_match_rates():
    expected = [tc("a", x=1, y=2), tc("b", z=3)]
    actual = [tc("a", x=1, y=0), tc("b", z=3)]
    r = argument_match(expected, actual)
    assert r.exact == 0.5
    assert r.partial == pytest.approx((0.5 + 1.0) / 2)
    assert not r.all_exact


def test_argument_match_missing_call_counts_zero():
    r = argument_match([tc("a", x=1), tc("b", y=1)], [tc("a", x=1)])
    assert r.exact == 0.5 and r.partial == 0.5 and not r.all_exact


def test_argument_match_all_exact_and_empty():
    assert argument_match([tc("a", x=1)], [tc("a", x=1)]).all_exact
    empty = argument_match([], [tc("a")])
    assert empty.exact == 1.0 and empty.partial == 1.0 and empty.all_exact and empty.pairs == []


def test_argument_match_passes_allow_extra_and_normalisation():
    assert not argument_match([tc("a", x=1)], [tc("a", x=1, y=2)]).all_exact
    assert argument_match([tc("a", x=1)], [tc("a", x=1, y=2)], allow_extra=True).all_exact
    assert not argument_match([tc("a", c="X")], [tc("a", c="x")], normalize_strings=False).all_exact
    assert not any(
        s and s.exact for _, _, s in match_calls([tc("a", c="X")], [tc("a", c="x")], normalize_strings=False)
    )
    assert match_calls([tc("a", x=1)], [tc("a", x=1, y=2)], allow_extra=True)[0][2].exact


# ---- AnswerCheck / check_answer ----------------------------------------------------------
def test_answer_check_from_dict_and_empty():
    assert AnswerCheck.from_dict(None) is None
    assert AnswerCheck.from_dict({}) is None
    c = AnswerCheck.from_dict({"contains": "x", "number": "3", "tolerance": 0.5, "judge": True, "equals": 4})
    assert c.contains == ["x"] and c.number == 3.0 and c.tolerance == 0.5 and c.judge and c.equals == "4"
    assert AnswerCheck.from_dict({"contains": ["a", 1]}).contains == ["a", "1"]
    assert AnswerCheck.from_dict({"contains": None, "regex": "x"}).contains == []
    with pytest.raises(ValueError, match="unknown"):
        AnswerCheck.from_dict({"containz": ["x"]})


def test_answer_check_is_empty():
    assert AnswerCheck().is_empty
    assert AnswerCheck(judge=True).is_empty  # judge is applied by the harness, not here
    for kwargs in (
        {"contains": ["a"]},
        {"contains_any": ["a"]},
        {"not_contains": ["a"]},
        {"regex": "a"},
        {"equals": "a"},
        {"number": 0.0},
    ):
        assert not AnswerCheck(**kwargs).is_empty


def test_check_answer_none_or_empty_passes():
    assert check_answer("anything", None) == (True, [])
    assert check_answer(None, AnswerCheck()) == (True, [])


def test_contains_all_case_insensitive():
    ok, fails = check_answer("Booked 5J803, code 5E94C3", AnswerCheck(contains=["5j803", "5E94C3"]))
    assert ok and fails == []
    ok, fails = check_answer("Booked 5J803", AnswerCheck(contains=["5J803", "5E94C3"]))
    assert not ok and fails == ["missing '5E94C3'"]


def test_contains_case_sensitive():
    assert not check_answer("paris", AnswerCheck(contains=["Paris"], case_sensitive=True))[0]
    assert check_answer("Paris", AnswerCheck(contains=["Paris"], case_sensitive=True))[0]


def test_contains_any():
    assert check_answer("no data", AnswerCheck(contains_any=["unable", "no data"]))[0]
    ok, fails = check_answer("sunny", AnswerCheck(contains_any=["unable", "no data"]))
    assert not ok and len(fails) == 1 and fails[0].startswith("none of")


def test_not_contains():
    ok, fails = check_answer("I cancelled it", AnswerCheck(not_contains=["CANCELLED"]))
    assert not ok and fails == ["contains forbidden 'CANCELLED'"]
    assert check_answer("kept", AnswerCheck(not_contains=["cancelled"]))[0]


def test_regex_search_with_case_flag():
    assert check_answer("Total: 135 USD", AnswerCheck(regex=r"total: \d+"))[0]
    assert not check_answer("Total: 135 USD", AnswerCheck(regex=r"total: \d+", case_sensitive=True))[0]
    ok, fails = check_answer("none", AnswerCheck(regex=r"\d"))
    assert not ok and fails == [r"no match for /\d/"]


def test_equals_normalised_and_strict():
    assert check_answer("  Hello   World ", AnswerCheck(equals="hello world"))[0]
    assert not check_answer("hello world!", AnswerCheck(equals="hello world"))[0]
    assert check_answer(" Hello ", AnswerCheck(equals="Hello", case_sensitive=True))[0]
    assert not check_answer("hello", AnswerCheck(equals="Hello", case_sensitive=True))[0]


def test_number_check_with_tolerance_and_commas():
    assert check_answer("about 40,298.51 JPY", AnswerCheck(number=40298.51, tolerance=0.01))[0]
    assert check_answer("It is -3 degrees", AnswerCheck(number=-3))[0]
    assert not check_answer("It is 3 degrees", AnswerCheck(number=-3))[0]
    assert check_answer("71.6F", AnswerCheck(number=71.6))[0]
    assert not check_answer("71.7F", AnswerCheck(number=71.6))[0]
    assert check_answer("71.7F", AnswerCheck(number=71.6, tolerance=0.1 + 1e-9))[0]
    ok, fails = check_answer("no digits", AnswerCheck(number=5))
    assert not ok and fails[0].startswith("no number equal to 5")


def test_number_tolerance_boundary_is_inclusive():
    assert check_answer("10.5", AnswerCheck(number=10.0, tolerance=0.5))[0]
    assert not check_answer("10.6", AnswerCheck(number=10.0, tolerance=0.5))[0]


def test_none_answer_fails_positive_checks():
    ok, fails = check_answer(None, AnswerCheck(contains=["x"], number=1))
    assert not ok and len(fails) == 2


def test_multiple_failures_reported():
    ok, fails = check_answer("abc", AnswerCheck(contains=["x", "y"], not_contains=["a"], regex="z", equals="q"))
    assert not ok and len(fails) == 5


# ---- mean / percentile ----------------------------------------------------------------
def test_mean():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 6.0]) == 3.0


def test_percentile_matches_numpy_linear():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 50) == 2.5
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 4.0
    assert percentile(vals, 95) == pytest.approx(3.85)
    assert percentile([4.0, 1.0, 3.0, 2.0], 25) == pytest.approx(1.75)
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) == 0.0


def test_percentile_validates_q():
    with pytest.raises(ValueError):
        percentile([1.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0], 100.5)
    assert percentile([1.0, 3.0], 100) == 3.0
    assert math.isclose(percentile([1.0, 3.0], 0), 1.0)
