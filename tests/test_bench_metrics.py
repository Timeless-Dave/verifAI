import pytest

from verifai.bench.labels import BenchLabel
from verifai.bench.metrics import ScoredExample, score_examples
from verifai.bench.sampling import take_sample


def row(gold: str, predicted: str, *, blocked: bool = False, latency: float = 0.0) -> ScoredExample:
    return ScoredExample(
        gold=BenchLabel(gold),
        predicted=BenchLabel(predicted),
        blocked=blocked,
        latency_s=latency,
    )


def test_counts_match_a_hand_worked_matrix() -> None:
    rows = [
        row("supported", "supported", latency=0.1),
        row("supported", "refuted", blocked=True, latency=0.2),
        row("supported", "unverified", latency=0.3),
        row("refuted", "refuted", latency=0.4),
        row("refuted", "unverified", blocked=True, latency=0.5),
        row("unverified", "unverified", latency=0.6),
        row("unverified", "supported", latency=0.7),
        row("refuted", "refuted", latency=0.8),
    ]
    metrics = score_examples(rows)
    assert metrics.n == 8
    assert metrics.correct == 4
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.abstentions == 3
    assert metrics.abstention_rate == pytest.approx(0.375)
    assert metrics.non_abstained_n == 5
    assert metrics.non_abstained_correct == 3
    assert metrics.non_abstained_accuracy == pytest.approx(0.6)
    assert metrics.two_source_blocks == 2
    assert metrics.two_source_block_rate == pytest.approx(0.25)
    assert metrics.mean_latency_seconds == pytest.approx(0.45)
    assert metrics.confusion == ((1, 1, 1), (0, 2, 1), (1, 0, 1))
    by_label = {item.label: item for item in metrics.per_class}
    assert by_label["supported"].f1 == pytest.approx(0.4)
    assert by_label["refuted"].f1 == pytest.approx(2 / 3)
    assert by_label["unverified"].f1 == pytest.approx(0.4)
    assert metrics.macro_f1 == pytest.approx((0.4 + (2 / 3) + 0.4) / 3)


def test_undefined_f1_contributes_zero() -> None:
    metrics = score_examples([row("supported", "supported")])
    assert metrics.per_class[1].f1 is None
    assert metrics.macro_f1 == pytest.approx(1 / 3)
    assert metrics.non_abstained_accuracy == 1


def test_empty_sample_has_null_rates() -> None:
    metrics = score_examples([])
    assert metrics.accuracy is None
    assert metrics.macro_f1 is None
    assert metrics.abstention_rate is None
    assert metrics.two_source_block_rate is None
    assert metrics.mean_latency_seconds is None


def test_sample_is_stable_for_a_seed() -> None:
    rows = list(range(20))
    assert take_sample(rows, 5, 7) == take_sample(rows, 5, 7)
    assert take_sample(rows, 5, 7) != take_sample(rows, 5, 8)
    assert take_sample(rows, 0, 7) == rows
    assert take_sample(rows, None, 7) == rows


def test_blocked_lean_counts_only_blocked_rows_with_a_side() -> None:
    s, r, u = BenchLabel.SUPPORTED, BenchLabel.REFUTED, BenchLabel.UNVERIFIED
    rows = [
        ScoredExample(gold=s, predicted=u, blocked=True, latency_s=0.0, lean=s),
        ScoredExample(gold=r, predicted=u, blocked=True, latency_s=0.0, lean=s),
        ScoredExample(gold=r, predicted=u, blocked=True, latency_s=0.0, lean=None),
        ScoredExample(gold=s, predicted=s, blocked=False, latency_s=0.0, lean=r),
    ]
    metrics = score_examples(rows)
    assert metrics.two_source_blocks == 3
    assert metrics.blocked_leaning == 2
    assert metrics.blocked_lean_correct == 1
    assert metrics.blocked_lean_accuracy == 0.5
    assert score_examples([]).blocked_lean_accuracy is None
