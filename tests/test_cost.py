import pytest

from agentlens.cost import ModelPrice, Pricing, PricingError


def test_model_price_cost_per_million():
    p = ModelPrice(input_per_mtok=2.5, output_per_mtok=10.0)
    assert p.cost(1_000_000, 0) == 2.5
    assert p.cost(0, 1_000_000) == 10.0
    assert p.cost(1000, 500) == pytest.approx(0.0025 + 0.005)
    assert p.cost(0, 0) == 0.0


def test_cached_tokens_use_cached_rate_and_default_to_input_rate():
    p = ModelPrice(2.0, 8.0, cached_input_per_mtok=0.5)
    # 1M input of which 400k cached: 600k * 2 + 400k * 0.5 = 1.2 + 0.2
    assert p.cost(1_000_000, 0, cached_input_tokens=400_000) == pytest.approx(1.4)
    no_cache_rate = ModelPrice(2.0, 8.0)
    assert no_cache_rate.cost(1_000_000, 0, cached_input_tokens=400_000) == pytest.approx(2.0)
    free_cache = ModelPrice(2.0, 8.0, cached_input_per_mtok=0.0)
    assert free_cache.cost(1_000_000, 0, cached_input_tokens=1_000_000) == 0.0


def test_cost_is_rounded_to_ten_decimals():
    p = ModelPrice(0.15, 0.6)
    assert p.cost(1, 1) == 7.5e-07
    assert ModelPrice(1 / 3, 0).cost(1, 0) == round(1 / 3 / 1_000_000, 10)


def test_cost_validates_token_counts():
    p = ModelPrice(1.0, 1.0)
    with pytest.raises(ValueError):
        p.cost(-1, 0)
    with pytest.raises(ValueError):
        p.cost(0, -1)
    with pytest.raises(ValueError):
        p.cost(10, 0, cached_input_tokens=-1)
    with pytest.raises(ValueError, match="exceed"):
        p.cost(10, 0, cached_input_tokens=11)
    assert p.cost(10, 0, cached_input_tokens=10) == pytest.approx(1e-5)


def _pricing():
    return Pricing.from_dict(
        {
            "models": {
                "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.6},
                "qwen2.5-*": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
                "qwen2.5-7b*": {"input_per_mtok": 0.2, "output_per_mtok": 0.2},
                "claude-?-haiku": {"input_per_mtok": 1, "output_per_mtok": 5},
            },
            "aliases": {"mini": "gpt-4o-mini"},
        }
    )


def test_lookup_exact_alias_and_glob():
    p = _pricing()
    assert p.lookup("gpt-4o-mini") == ModelPrice(0.15, 0.6)
    assert p.lookup("mini") == ModelPrice(0.15, 0.6)
    assert p.lookup("qwen2.5-0.5b-instruct") == ModelPrice(0.0, 0.0)
    assert p.lookup("claude-3-haiku") == ModelPrice(1.0, 5.0)


def test_lookup_most_specific_glob_wins():
    p = _pricing()
    assert p.lookup("qwen2.5-7b-instruct") == ModelPrice(0.2, 0.2)
    assert p.lookup("qwen2.5-14b") == ModelPrice(0.0, 0.0)


def test_lookup_glob_is_case_insensitive_and_exact_names_need_a_full_match():
    p = _pricing()
    assert p.lookup("Qwen2.5-0.5B-Instruct") == ModelPrice(0.0, 0.0)
    assert p.lookup("gpt-4o-mini-extra") is None  # no implicit prefix match
    assert p.lookup("gpt-4o-min") is None


def test_lookup_strips_paths_and_gguf_suffix():
    p = _pricing()
    assert p.lookup("/models/qwen2.5-0.5b-instruct-q4_k_m.gguf") == ModelPrice(0.0, 0.0)
    assert p.lookup("C:\\models\\gpt-4o-mini") == ModelPrice(0.15, 0.6)
    assert p.lookup("gpt-4o-mini.GGUF") == ModelPrice(0.15, 0.6)
    assert p.lookup("/x/unknown.gguf") is None


def test_lookup_unknown_and_empty():
    p = _pricing()
    assert p.lookup("") is None
    assert p.lookup("llama-3") is None
    assert p.cost("llama-3", 100, 100) is None
    assert "llama-3" not in p
    assert "mini" in p


def test_cost_for_model():
    p = _pricing()
    assert p.cost("gpt-4o-mini", input_tokens=2_000_000, output_tokens=1_000_000) == pytest.approx(0.9)
    assert p.cost("qwen2.5-0.5b", 10_000, 10_000) == 0.0
    assert p.cost("gpt-4o-mini") == 0.0


def test_from_dict_validation():
    with pytest.raises(PricingError, match="must be a table"):
        Pricing.from_dict({"models": []})
    with pytest.raises(PricingError, match="must be a table"):
        Pricing.from_dict({"models": {"m": 3}})
    with pytest.raises(PricingError, match="missing output_per_mtok"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": 1}}})
    with pytest.raises(PricingError, match="numbers"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": "x", "output_per_mtok": 1}}})
    with pytest.raises(PricingError, match="unknown keys"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": 1, "output_per_mtok": 1, "per_call": 1}}})
    with pytest.raises(PricingError, match=">= 0"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": -1, "output_per_mtok": 1}}})
    with pytest.raises(PricingError, match="output_per_mtok must be >= 0"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": 0, "output_per_mtok": -0.5}}})
    with pytest.raises(PricingError, match="cached_input_per_mtok must be >= 0"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": 0, "output_per_mtok": 0, "cached_input_per_mtok": -1}}})
    with pytest.raises(PricingError, match="unknown model"):
        Pricing.from_dict({"models": {}, "aliases": {"a": "missing"}})
    with pytest.raises(PricingError, match="aliases"):
        Pricing.from_dict({"aliases": {"a": 3}})
    with pytest.raises(PricingError, match="aliases"):
        Pricing.from_dict({"aliases": ["a"]})


def test_from_dict_zero_prices_and_cached_allowed():
    p = Pricing.from_dict({"models": {"m": {"input_per_mtok": 0, "output_per_mtok": 0, "cached_input_per_mtok": 0}}})
    assert p.lookup("m") == ModelPrice(0.0, 0.0, 0.0)
    assert len(Pricing.from_dict({})) == 0


def test_from_toml(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('[models."my-model"]\ninput_per_mtok = 1.5\noutput_per_mtok = 3\n[aliases]\nmine = "my-model"\n')
    p = Pricing.from_toml(f)
    assert p.cost("mine", 1_000_000, 1_000_000) == 4.5
    bad = tmp_path / "bad.toml"
    bad.write_text("[models\n")
    with pytest.raises(PricingError, match="bad.toml"):
        Pricing.from_toml(bad)


def test_default_table_loads_and_has_known_models():
    p = Pricing.default()
    assert len(p) >= 5
    assert p.lookup("gpt-4o-mini") is not None
    assert p.lookup("gpt-4o-mini-2024-07-18") == p.lookup("gpt-4o-mini")
    assert p.cost("qwen2.5-0.5b-instruct-q4_k_m", 1000, 1000) == 0.0
    assert p.cost("mock-v1", 1_000_000, 1_000_000) == 3.0


def test_merged_overrides():
    base = _pricing()
    override = Pricing({"gpt-4o-mini": ModelPrice(9, 9)}, {"tiny": "gpt-4o-mini"})
    m = base.merged(override)
    assert m.lookup("gpt-4o-mini") == ModelPrice(9, 9)
    assert m.lookup("tiny") == ModelPrice(9, 9)
    assert m.lookup("mini") == ModelPrice(9, 9)
    assert base.lookup("gpt-4o-mini") == ModelPrice(0.15, 0.6)  # original untouched
    assert len(m) == len(base)


def test_lookup_exact_and_alias_are_case_insensitive():
    p = _pricing()
    assert p.lookup("GPT-4o-Mini") == ModelPrice(0.15, 0.6)
    assert p.lookup("MINI") == ModelPrice(0.15, 0.6)


def test_exact_name_beats_glob_of_equal_specificity():
    p = Pricing({"abc*": ModelPrice(1, 1), "abc": ModelPrice(2, 2)})
    assert p.lookup("abc") == ModelPrice(2, 2)
    q = Pricing({"abc": ModelPrice(2, 2), "abc*": ModelPrice(1, 1)})
    assert q.lookup("abc") == ModelPrice(2, 2)
    assert q.lookup("abcd") == ModelPrice(1, 1)


def test_first_entry_wins_on_full_tie():
    p = Pricing({"abc?": ModelPrice(1, 1), "ab?d": ModelPrice(2, 2)})
    assert p.lookup("abcd") == ModelPrice(1, 1)
    q = Pricing({"ab?d": ModelPrice(2, 2), "abc?": ModelPrice(1, 1)})
    assert q.lookup("abcd") == ModelPrice(2, 2)


def test_more_specific_glob_wins_regardless_of_order():
    p = Pricing({"q-7b*": ModelPrice(7, 7), "q-*": ModelPrice(1, 1)})
    assert p.lookup("q-7b-instruct") == ModelPrice(7, 7)
    assert p.lookup("q-1b") == ModelPrice(1, 1)


def test_path_lookup_uses_last_component_only():
    p = Pricing({"m": ModelPrice(1, 1), "a/b/m": ModelPrice(9, 9)})
    assert p.lookup("/x/y/a/b/m.gguf") == ModelPrice(1, 1)
    assert p.lookup("dir/m") == ModelPrice(1, 1)


def test_alias_may_not_shadow_a_model():
    with pytest.raises(PricingError, match="shadows"):
        Pricing.from_dict({"models": {"m": {"input_per_mtok": 1, "output_per_mtok": 1}}, "aliases": {"M": "m"}})
