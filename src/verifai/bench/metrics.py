"""Three-class accuracy for one benchmark sample.

Macro-F1 is the unweighted mean of the three class F1 scores. A class with
an empty precision or recall denominator has F1 null. That class contributes
0 to the macro average. The per-class table still shows n/a, and the counts
are stored so the average can be recomputed.

Abstention is a prediction of unverified. Accuracy on non-abstained claims
ignores those rows. It is null when the system never said supported or refuted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verifai.bench.labels import BenchLabel

LABELS: tuple[BenchLabel, ...] = (
    BenchLabel.SUPPORTED,
    BenchLabel.REFUTED,
    BenchLabel.UNVERIFIED,
)


@dataclass(frozen=True)
class ScoredExample:
    gold: BenchLabel
    predicted: BenchLabel
    blocked: bool
    latency_s: float


@dataclass(frozen=True)
class ClassScore:
    label: str
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class Metrics:
    n: int
    correct: int
    accuracy: float | None
    abstentions: int
    abstention_rate: float | None
    non_abstained_n: int
    non_abstained_correct: int
    non_abstained_accuracy: float | None
    two_source_blocks: int
    two_source_block_rate: float | None
    total_latency_seconds: float
    mean_latency_seconds: float | None
    per_class: tuple[ClassScore, ...]
    macro_f1: float | None
    confusion: tuple[tuple[int, ...], ...]


def score_examples(rows: Sequence[ScoredExample]) -> Metrics:
    n = len(rows)
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    index = {label: position for position, label in enumerate(LABELS)}
    blocked = 0
    latency = 0.0
    for row in rows:
        matrix[index[row.gold]][index[row.predicted]] += 1
        if row.blocked:
            blocked += 1
        latency += row.latency_s

    per_class: list[ClassScore] = []
    for gold_index, label in enumerate(LABELS):
        true_positive = matrix[gold_index][gold_index]
        false_positive = sum(
            matrix[other][gold_index] for other in range(len(LABELS)) if other != gold_index
        )
        false_negative = sum(
            matrix[gold_index][other] for other in range(len(LABELS)) if other != gold_index
        )
        precision = _rate(true_positive, true_positive + false_positive)
        recall = _rate(true_positive, true_positive + false_negative)
        per_class.append(
            ClassScore(
                label=label.value,
                tp=true_positive,
                fp=false_positive,
                fn=false_negative,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
            )
        )

    correct = sum(matrix[position][position] for position in range(len(LABELS)))
    unverified = index[BenchLabel.UNVERIFIED]
    abstentions = sum(matrix[gold_index][unverified] for gold_index in range(len(LABELS)))
    non_abstained_n = n - abstentions
    non_abstained_correct = correct - matrix[unverified][unverified]
    return Metrics(
        n=n,
        correct=correct,
        accuracy=_rate(correct, n),
        abstentions=abstentions,
        abstention_rate=_rate(abstentions, n),
        non_abstained_n=non_abstained_n,
        non_abstained_correct=non_abstained_correct,
        non_abstained_accuracy=_rate(non_abstained_correct, non_abstained_n),
        two_source_blocks=blocked,
        two_source_block_rate=_rate(blocked, n),
        total_latency_seconds=latency,
        mean_latency_seconds=(latency / n) if n else None,
        per_class=tuple(per_class),
        macro_f1=None if n == 0 else sum(item.f1 or 0.0 for item in per_class) / len(LABELS),
        confusion=tuple(tuple(row) for row in matrix),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
