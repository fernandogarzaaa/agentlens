"""Agreement statistics for calibrating an LLM judge against human labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["accuracy", "cohen_kappa", "confusion_matrix", "CalibrationReport", "calibrate", "ABSTAIN"]

ABSTAIN = "abstain"


def accuracy(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa: agreement corrected for the agreement expected by chance.

    1 = perfect agreement, 0 = chance level, < 0 = worse than chance. When both raters use a single
    identical label throughout (chance agreement is 1), kappa is defined here as 1.0.
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        raise ValueError("no labels")
    p_o = accuracy(a, b)
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def confusion_matrix(
    truth: Sequence[Hashable], predicted: Sequence[Hashable], labels: Sequence[Hashable] | None = None
) -> dict[str, dict[str, int]]:
    """``matrix[truth][predicted] = count`` with string keys (JSON friendly)."""
    if len(truth) != len(predicted):
        raise ValueError("label sequences differ in length")
    labels = list(labels) if labels is not None else sorted({*truth, *predicted}, key=str)
    matrix = {str(t): {str(p): 0 for p in labels} for t in labels}
    for t, p in zip(truth, predicted):
        matrix.setdefault(str(t), {str(x): 0 for x in labels})
        matrix[str(t)][str(p)] = matrix[str(t)].get(str(p), 0) + 1
    return matrix


@dataclass
class CalibrationReport:
    n: int
    accuracy: float
    kappa: float
    abstentions: int
    labels: list[str]
    confusion: dict[str, dict[str, int]]
    human_distribution: dict[str, int] = field(default_factory=dict)
    judge_distribution: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def calibrate(
    judge_labels: Sequence[Hashable | None], human_labels: Sequence[Hashable], labels: Sequence[str] | None = None
) -> CalibrationReport:
    """Compare judge labels with human labels. Unparseable judge outputs (``None``) count as an
    ``abstain`` label, so they lower both accuracy and kappa instead of being silently dropped."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("label sequences differ in length")
    judged = [ABSTAIN if j is None else str(j) for j in judge_labels]
    human = [str(h) for h in human_labels]
    label_set = list(labels) if labels is not None else sorted(set(human))
    if ABSTAIN in judged and ABSTAIN not in label_set:
        label_set.append(ABSTAIN)
    return CalibrationReport(
        n=len(human),
        accuracy=accuracy(judged, human),
        kappa=cohen_kappa(judged, human),
        abstentions=judged.count(ABSTAIN),
        labels=label_set,
        confusion=confusion_matrix(human, judged, label_set),
        human_distribution=dict(Counter(human)),
        judge_distribution=dict(Counter(judged)),
    )
