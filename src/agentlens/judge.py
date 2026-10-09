"""LLM-as-judge with an explicit rubric, pointwise or pairwise.

Pointwise::

    judge = RubricJudge(client, model="gpt-4o-mini", rubric=Rubric.load("rubric.yaml"))
    verdict = judge.score(question, answer)          # JudgeVerdict(score=4, passed=True, reasoning=...)

Pairwise (MT-Bench style, optional order swap to cancel position bias)::

    verdict = PairwiseJudge(client, model, rubric, swap=True).compare(question, answer_a, answer_b)
    verdict.winner                                   # "A" | "B" | "tie" | None (unparseable)

Calibrate any judge against human labels with :func:`agentlens.calibration.calibrate`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentlens.llm import ChatClient, LLMError
from agentlens.tracer import Tracer

__all__ = ["Rubric", "JudgeVerdict", "PairwiseVerdict", "RubricJudge", "PairwiseJudge", "parse_score", "parse_winner"]

DEFAULT_CRITERIA = [
    "Correctness: the answer is factually right and consistent with any tool results shown.",
    "Completeness: it addresses every part of the user's request.",
    "Helpfulness: it is clear, specific and directly usable; no padding.",
]


@dataclass
class Rubric:
    name: str = "default"
    instructions: str = "Judge how well the assistant's answer serves the user."
    criteria: list[str] = field(default_factory=lambda: list(DEFAULT_CRITERIA))
    scale_min: int = 1
    scale_max: int = 5
    pass_threshold: float = 4.0

    def __post_init__(self) -> None:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be below scale_max")
        if not self.scale_min <= self.pass_threshold <= self.scale_max:
            raise ValueError("pass_threshold must lie on the scale")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Rubric:
        known = {"name", "instructions", "criteria", "scale_min", "scale_max", "pass_threshold"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown rubric keys {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def render(self) -> str:
        lines = [self.instructions, "", "Criteria:"]
        lines += [f"- {c}" for c in self.criteria]
        return "\n".join(lines)


@dataclass
class JudgeVerdict:
    score: float | None
    passed: bool | None
    reasoning: str = ""
    raw: str = ""


@dataclass
class PairwiseVerdict:
    winner: str | None  # "A" | "B" | "tie" | None
    raw: list[str] = field(default_factory=list)


_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_PATTERNS = [
    re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]"),
    re.compile(r"\"?score\"?\s*[:=]\s*\"?(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"rating\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*\d+"),
]


def parse_score(text: str, scale_min: float, scale_max: float) -> tuple[float | None, str]:
    """Extract a score (and reasoning if JSON) from a judge reply; ``None`` if absent or off-scale."""
    reasoning = ""
    for m in _JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            reasoning = str(obj.get("reasoning", ""))
            try:
                value = float(obj["score"])
            except (TypeError, ValueError):
                break
            return (value if scale_min <= value <= scale_max else None), reasoning
    for pattern in _SCORE_PATTERNS:
        m2 = pattern.search(text)
        if m2:
            value = float(m2.group(1))
            return (value if scale_min <= value <= scale_max else None), reasoning or text.strip()[:500]
    return None, text.strip()[:500]


def parse_winner(text: str) -> str | None:
    """MT-Bench verdict format: ``[[A]]``, ``[[B]]`` or ``[[C]]`` (tie). Falls back to a bare letter
    on the last line."""
    m = re.findall(r"\[\[\s*([ABC])\s*\]\]", text)
    if m:
        letter = m[-1]
    else:
        last = text.strip().splitlines()[-1].strip().upper() if text.strip() else ""
        m2 = re.fullmatch(r"(?:VERDICT:?\s*)?([ABC])\.?", last)
        if not m2:
            return None
        letter = m2.group(1)
    return {"A": "A", "B": "B", "C": "tie"}[letter]


class _BaseJudge:
    def __init__(
        self,
        client: ChatClient,
        model: str,
        rubric: Rubric | None = None,
        tracer: Tracer | None = None,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.model = model
        self.rubric = rubric or Rubric()
        self.tracer = tracer
        self.max_tokens = max_tokens

    def _ask(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.tracer is None:
            return self.client.chat(messages, self.model, max_tokens=self.max_tokens).content or ""
        with self.tracer.llm_call(self.model, name=f"judge {self.model}") as span:
            span.set_attribute("agentlens.judge", True)
            resp = self.client.chat(messages, self.model, max_tokens=self.max_tokens)
            span.record_usage(resp.input_tokens, resp.output_tokens, model=resp.model or None)
            span.set_output(resp.content)
            return resp.content or ""


class RubricJudge(_BaseJudge):
    """Scores one answer on the rubric's scale."""

    def prompt(self, question: str, answer: str, reference: str | None = None) -> tuple[str, str]:
        r = self.rubric
        system = (
            "You are a strict, impartial evaluator. Grade only what is asked; do not reward length. "
            f'Reply with JSON only: {{"reasoning": "<one or two sentences>", "score": <integer {r.scale_min}-{r.scale_max}>}}'
        )
        parts = [
            r.render(),
            "",
            f"Scale: {r.scale_min} (worst) to {r.scale_max} (best).",
            "",
            f"[User question]\n{question}",
        ]
        if reference:
            parts += ["", f"[Reference answer]\n{reference}"]
        parts += ["", f"[Assistant answer]\n{answer}", "", "Return the JSON now."]
        return system, "\n".join(parts)

    def score(self, question: str, answer: str, reference: str | None = None) -> JudgeVerdict:
        system, user = self.prompt(question, answer, reference)
        try:
            raw = self._ask(system, user)
        except LLMError as exc:
            return JudgeVerdict(None, None, f"judge call failed: {exc}", "")
        value, reasoning = parse_score(raw, self.rubric.scale_min, self.rubric.scale_max)
        passed = None if value is None else value >= self.rubric.pass_threshold
        return JudgeVerdict(value, passed, reasoning, raw)


class PairwiseJudge(_BaseJudge):
    """Picks the better of two answers. With ``swap=True`` both orders are judged; a winner must win
    in both, otherwise the verdict is a tie (the MT-Bench recipe against position bias)."""

    def __init__(self, *args: Any, swap: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.swap = swap

    def prompt(self, question: str, first: str, second: str) -> tuple[str, str]:
        system = (
            "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
            "assistants to the user question displayed below. Avoid any position biases and do not let the "
            "length of the responses influence your evaluation. After a short explanation, output your final "
            'verdict strictly in this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, '
            'and "[[C]]" for a tie.'
        )
        user = (
            f"{self.rubric.render()}\n\n[User Question]\n{question}\n\n"
            f"[The Start of Assistant A's Answer]\n{first}\n[The End of Assistant A's Answer]\n\n"
            f"[The Start of Assistant B's Answer]\n{second}\n[The End of Assistant B's Answer]"
        )
        return system, user

    def compare(self, question: str, answer_a: str, answer_b: str) -> PairwiseVerdict:
        raws: list[str] = []
        try:
            raw = self._ask(*self.prompt(question, answer_a, answer_b))
        except LLMError as exc:
            return PairwiseVerdict(None, [f"judge call failed: {exc}"])
        raws.append(raw)
        first = parse_winner(raw)
        if not self.swap:
            return PairwiseVerdict(first, raws)
        try:
            raw2 = self._ask(*self.prompt(question, answer_b, answer_a))
        except LLMError as exc:
            return PairwiseVerdict(None, [*raws, f"judge call failed: {exc}"])
        raws.append(raw2)
        swapped = parse_winner(raw2)
        second = {"A": "B", "B": "A"}.get(swapped, swapped) if swapped is not None else None
        if first is None or second is None:
            return PairwiseVerdict(first if second is None else second, raws)
        return PairwiseVerdict(first if first == second else "tie", raws)
