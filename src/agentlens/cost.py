"""Token cost accounting from a TOML pricing table.

Prices are USD per one million tokens::

    [models."gpt-4o-mini"]
    input_per_mtok = 0.15
    output_per_mtok = 0.60
    cached_input_per_mtok = 0.075   # optional; defaults to input_per_mtok

    [models."qwen2.5-*"]            # glob patterns are allowed; the most specific match wins
    input_per_mtok = 0.0
    output_per_mtok = 0.0

    [aliases]
    "gpt-4o-mini-2024-07-18" = "gpt-4o-mini"

A model with no matching entry has an *unknown* cost: :meth:`Pricing.cost` returns ``None`` and run
summaries carry ``cost_known = false`` instead of silently reporting $0.
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 in CI
    import tomli as tomllib

__all__ = ["ModelPrice", "Pricing", "PricingError"]

TOKENS_PER_UNIT = 1_000_000


class PricingError(ValueError):
    """Invalid pricing table."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
        """USD for one call. ``cached_input_tokens`` is the part of ``input_tokens`` served from cache."""
        if input_tokens < 0 or output_tokens < 0 or cached_input_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if cached_input_tokens > input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        cached_rate = self.input_per_mtok if self.cached_input_per_mtok is None else self.cached_input_per_mtok
        fresh = input_tokens - cached_input_tokens
        usd = (
            fresh * self.input_per_mtok + cached_input_tokens * cached_rate + output_tokens * self.output_per_mtok
        ) / TOKENS_PER_UNIT
        return round(usd, 10)


def _price_from_table(name: str, table: Any) -> ModelPrice:
    if not isinstance(table, dict):
        raise PricingError(f"models.{name!r} must be a table")
    unknown = set(table) - {"input_per_mtok", "output_per_mtok", "cached_input_per_mtok"}
    if unknown:
        raise PricingError(f"models.{name!r}: unknown keys {sorted(unknown)}")
    try:
        inp = float(table["input_per_mtok"])
        out = float(table["output_per_mtok"])
    except KeyError as exc:
        raise PricingError(f"models.{name!r} is missing {exc.args[0]}") from None
    except (TypeError, ValueError):
        raise PricingError(f"models.{name!r}: prices must be numbers") from None
    cached_raw = table.get("cached_input_per_mtok")
    cached = None if cached_raw is None else float(cached_raw)
    for label, value in (("input_per_mtok", inp), ("output_per_mtok", out), ("cached_input_per_mtok", cached)):
        if value is not None and value < 0:
            raise PricingError(f"models.{name!r}: {label} must be >= 0")
    return ModelPrice(inp, out, cached)


class Pricing:
    """Model name -> :class:`ModelPrice` lookup with exact names, aliases and glob patterns."""

    def __init__(self, models: dict[str, ModelPrice] | None = None, aliases: dict[str, str] | None = None) -> None:
        self.models: dict[str, ModelPrice] = dict(models or {})
        self.aliases: dict[str, str] = dict(aliases or {})

    # ---- loading -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pricing:
        models_raw = data.get("models", {})
        if not isinstance(models_raw, dict):
            raise PricingError("[models] must be a table")
        aliases = data.get("aliases", {})
        if not isinstance(aliases, dict) or not all(isinstance(v, str) for v in aliases.values()):
            raise PricingError("[aliases] must map model names to model names")
        models = {name: _price_from_table(name, t) for name, t in models_raw.items()}
        for alias, target in aliases.items():
            if target not in models:
                raise PricingError(f"alias {alias!r} points at unknown model {target!r}")
        return cls(models, aliases)

    @classmethod
    def from_toml(cls, path: str | Path) -> Pricing:
        with open(path, "rb") as fh:
            try:
                data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise PricingError(f"{path}: {exc}") from None
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> Pricing:
        """The bundled table (agentlens/data/pricing.toml). Check it against your provider's price page."""
        text = resources.files("agentlens").joinpath("data/pricing.toml").read_text(encoding="utf-8")
        return cls.from_dict(tomllib.loads(text))

    def merged(self, other: Pricing) -> Pricing:
        """A new table where entries of ``other`` override this one."""
        return Pricing({**self.models, **other.models}, {**self.aliases, **other.aliases})

    # ---- lookup --------------------------------------------------------------------
    def lookup(self, model: str) -> ModelPrice | None:
        """Exact name, then alias, then the longest matching glob pattern; ``None`` if nothing matches."""
        if not model:
            return None
        if model in self.models:
            return self.models[model]
        if model in self.aliases:
            return self.models[self.aliases[model]]
        # Local servers often report a file path ("/models/qwen2.5-0.5b-instruct-q4_k_m.gguf").
        name = model.replace("\\", "/").rsplit("/", 1)[-1]
        if name.lower().endswith(".gguf"):
            name = name[: -len(".gguf")]
        if name != model:
            return self.lookup(name)
        lowered = model.lower()
        best: tuple[int, str] | None = None
        for pattern in self.models:
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatchcase(lowered, pattern.lower()):
                specificity = len(pattern.replace("*", ""))
                if best is None or specificity > best[0]:
                    best = (specificity, pattern)
        return self.models[best[1]] if best is not None else None

    def cost(
        self, model: str, input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0
    ) -> float | None:
        """USD for a call to ``model``, or ``None`` when the model is not in the table."""
        price = self.lookup(model)
        if price is None:
            return None
        return price.cost(input_tokens, output_tokens, cached_input_tokens)

    def __contains__(self, model: str) -> bool:
        return self.lookup(model) is not None

    def __len__(self) -> int:
        return len(self.models)
