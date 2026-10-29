"""Score labeled rows by calling :func:`verifai.orchestrate.run_check`.

The router, the experts, and ``corroborate`` are the ones the app uses.
This module only times that call and reads the verdict it already returned.

A two-source block is an unverifiable verdict whose evidence still contains
a supporting or refuting stance. The claim rule withheld a side. A mixed
verdict is not a block: both sides were corroborated. Citation rows do not
use that rule, so a citation verdict is not counted as a block.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from verifai.bench.datasets import Example, LoadedDataset
from verifai.bench.labels import VERDICT_TO_BENCH, BenchLabel
from verifai.bench.metrics import Metrics, ScoredExample, score_examples
from verifai.bench.sampling import take_sample
from verifai.experts.router import Router
from verifai.orchestrate import CheckBundle, run_check
from verifai.schemas import Verdict

_DIRECTIONAL = {"supports", "refutes"}


@dataclass(frozen=True)
class Outcome:
    example_id: str
    dataset: str
    domain: str
    gold: BenchLabel
    predicted: BenchLabel
    blocked: bool
    latency_s: float
    raw_label: str
    text: str
    verdict: str
    experts: tuple[str, ...]
    error: str | None
    detail: str


@dataclass(frozen=True)
class DatasetRun:
    dataset: LoadedDataset
    sampled_ids: tuple[str, ...]
    outcomes: tuple[Outcome, ...]
    metrics: Metrics


@dataclass(frozen=True)
class RunResult:
    datasets: tuple[DatasetRun, ...]
    overall: Metrics
    by_domain: dict[str, Metrics]


def evaluate(
    router: Router,
    datasets: Sequence[LoadedDataset],
    *,
    seed: int,
    limit: int | None,
) -> RunResult:
    runs: list[DatasetRun] = []
    flat: list[Outcome] = []
    for dataset in datasets:
        sample = take_sample(dataset.examples, limit, seed)
        outcomes = tuple(_run_one(router, example) for example in sample)
        runs.append(
            DatasetRun(
                dataset=dataset,
                sampled_ids=tuple(example.id for example in sample),
                outcomes=outcomes,
                metrics=score_examples(_scored(outcomes)),
            )
        )
        flat.extend(outcomes)
    by_domain: dict[str, Metrics] = {}
    seen: list[str] = []
    for run in runs:
        if run.dataset.domain not in seen:
            seen.append(run.dataset.domain)
    for domain in seen:
        rows = [outcome for outcome in flat if outcome.domain == domain]
        by_domain[domain] = score_examples(_scored(rows))
    return RunResult(
        datasets=tuple(runs), overall=score_examples(_scored(flat)), by_domain=by_domain
    )


def two_source_blocked(bundle: CheckBundle) -> bool:
    """True when claim evidence took a side and the verdict stayed unverifiable."""
    if bundle.verdict is not Verdict.UNVERIFIABLE:
        return False
    for result in bundle.results:
        if result.expert == "citation":
            continue
        for item in result.evidence:
            if item.stance in _DIRECTIONAL:
                return True
    for claim in bundle.combined:
        for item in claim.evidence:
            if item.stance in _DIRECTIONAL:
                return True
    return False


def _run_one(router: Router, example: Example) -> Outcome:
    started = time.perf_counter()
    try:
        bundle = run_check(router, example.text, limit=1)
        predicted = VERDICT_TO_BENCH[bundle.verdict]
        blocked = two_source_blocked(bundle)
        experts = tuple(dict.fromkeys(result.expert for result in bundle.results))
        detail = bundle.explanation
        error = None
        verdict = bundle.verdict.value
    except (RuntimeError, OSError, ValueError, httpx.HTTPError) as exc:
        predicted = BenchLabel.UNVERIFIED
        blocked = False
        experts = ()
        detail = ""
        error = f"{type(exc).__name__}: {exc}"
        verdict = Verdict.UNVERIFIABLE.value
    return Outcome(
        example_id=example.id,
        dataset=example.dataset,
        domain=example.domain,
        gold=example.gold,
        predicted=predicted,
        blocked=blocked,
        latency_s=time.perf_counter() - started,
        raw_label=example.raw_label,
        text=example.text[:400],
        verdict=verdict,
        experts=experts,
        error=error,
        detail=detail[:500],
    )


def _scored(outcomes: Sequence[Outcome]) -> list[ScoredExample]:
    return [
        ScoredExample(
            gold=outcome.gold,
            predicted=outcome.predicted,
            blocked=outcome.blocked,
            latency_s=outcome.latency_s,
        )
        for outcome in outcomes
    ]
