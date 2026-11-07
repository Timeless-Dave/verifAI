"""JSON results and a Markdown report.

Every report names the git SHA, the model id, the dataset version and file
hash, the sample size, the seed, and the date.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verifai import __version__
from verifai.bench.datasets import DOMAIN_DISPLAY
from verifai.bench.labels import BenchLabel
from verifai.bench.metrics import ClassScore, Metrics
from verifai.bench.scoring import DatasetRun, Outcome, RunResult

_CAVEATS = (
    "Sample size is whatever --limit kept. A 50-row sample is a noisy estimate.",
    "Claim rows go through run_check: the router, the existing experts, and corroborate.",
    "A supported or refuted claim needs two independent sources. Wikipedia is one source no matter how many articles were opened. The same paper in PubMed and Europe PMC is one source. Google Fact Check is one source.",
    "The none backend does not judge passages. Its claim accuracy is the share of gold unverified rows, and its abstention rate is 1, unless a retrieved page already carries a published rating.",
    "The default NLI model is the one in Settings, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli. It was trained on FEVER, so a FEVER score from that model is not an out-of-domain result.",
    "SciFact retrieval here is PubMed and Europe PMC, not the SciFact abstract corpus and not the gold cited documents.",
    "LIAR's half-true and barely-true labels are mapped to unverified. They are not FEVER's NOT ENOUGH INFO.",
    "Live pages change. The same seed can score differently later.",
    "A mixed verdict is scored unverified. It is an abstention, not a two-source block. A block is an unverifiable verdict that still has a supporting or refuting stance in the evidence.",
    "The citation checker asks whether a reference exists. It does not use the two-source claim rule, and one index can still verify a citation.",
    "Sentences shorter than 30 characters are not extracted as claims. If the router also does not see a health or news cue, the row is unverifiable because nothing was checked.",
)


@dataclass(frozen=True)
class RunMeta:
    date: str
    git_sha: str
    backend: str
    model_id: str
    judge_name: str
    seed: int
    limit: int | None
    offline: bool
    verifai_version: str = __version__


def git_sha(root: Path | None = None) -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        completed = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_report(
    result: RunResult,
    meta: RunMeta,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report_payload(result, meta), indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(result, meta), encoding="utf-8")


def report_payload(result: RunResult, meta: RunMeta) -> dict[str, Any]:
    return {
        "date": meta.date,
        "git_sha": meta.git_sha,
        "verifai_version": meta.verifai_version,
        "backend": meta.backend,
        "model_id": meta.model_id,
        "judge_name": meta.judge_name,
        "seed": meta.seed,
        "limit": meta.limit,
        "offline": meta.offline,
        "label_order": [label.value for label in BenchLabel],
        "macro_f1_note": (
            "Unweighted mean of the three class F1 scores. An undefined class F1 "
            "is null in per_class and contributes 0 to macro_f1."
        ),
        "datasets": [_dataset_payload(run) for run in result.datasets],
        "overall": metrics_payload(result.overall),
        "by_domain": {
            domain: metrics_payload(metrics) for domain, metrics in result.by_domain.items()
        },
        "examples": [
            _outcome_payload(outcome) for run in result.datasets for outcome in run.outcomes
        ],
        "caveats": list(_CAVEATS),
    }


def render_markdown(result: RunResult, meta: RunMeta) -> str:
    errors = [outcome for run in result.datasets for outcome in run.outcomes if outcome.error]
    lines = [
        "# VerifAI accuracy benchmark",
        "",
        f"- Date: {meta.date}",
        f"- Git SHA: `{meta.git_sha}`",
        f"- VerifAI: {meta.verifai_version}",
        f"- Backend: `{meta.backend}`",
        f"- Model id: `{meta.model_id}`",
        f"- Judge: `{meta.judge_name}`",
        f"- Seed: {meta.seed}",
        f"- Requested limit: {_limit_text(meta.limit)}",
        f"- Offline fixtures: {'yes' if meta.offline else 'no'}",
        f"- Examples scored: {result.overall.n}",
        "",
    ]
    if errors:
        lines.extend(
            [
                f"**{len(errors)} example(s) raised an error and were scored unverified.**",
                "Those rows keep the error string in the JSON. They are not silent successes.",
                "",
            ]
        )
    lines.extend(
        [
            "## How to read this",
            "",
            "Accuracy is correct labels over the sample. Abstention is how often the prediction is unverified.",
            "Accuracy on non-abstained claims ignores unverified predictions. Macro-F1 averages the three class F1 scores and treats an undefined F1 as 0.",
            "A two-source block is an unverifiable claim whose retrieved evidence still took a side. The existing rule withheld the verdict.",
            "",
            "## Datasets",
            "",
            "| Dataset | Domain | Split | Version | n | skipped while loading | sha256 |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for run in result.datasets:
        dataset = run.dataset
        domain = DOMAIN_DISPLAY.get(dataset.domain, dataset.domain)
        lines.append(
            f"| {dataset.name} | {domain} | {dataset.split} | {dataset.version} | "
            f"{len(run.outcomes)} | {dataset.skipped} | `{dataset.sha256}` |"
        )
    lines.extend(["", "Label mappings and licenses:", ""])
    for run in result.datasets:
        dataset = run.dataset
        lines.append(f"- **{dataset.name}** (`{dataset.source_url}`). {dataset.license_note}")
        lines.extend(f"  - {note}" for note in dataset.mapping_notes)
        if dataset.extra_sha256:
            lines.append(f"  - Corpus sha256: `{dataset.extra_sha256}`")
    lines.extend(["", "## Overall", ""])
    lines.extend(_metrics_lines(result.overall))
    lines.extend(["", "## By domain", ""])
    for domain, metrics in result.by_domain.items():
        lines.extend([f"### {DOMAIN_DISPLAY.get(domain, domain)}", ""])
        lines.extend(_metrics_lines(metrics))
        lines.append("")
    lines.extend(["## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in _CAVEATS)
    lines.append("")
    return "\n".join(lines)


def metrics_payload(metrics: Metrics) -> dict[str, Any]:
    return {
        "n": metrics.n,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "abstentions": metrics.abstentions,
        "abstention_rate": metrics.abstention_rate,
        "non_abstained_n": metrics.non_abstained_n,
        "non_abstained_correct": metrics.non_abstained_correct,
        "non_abstained_accuracy": metrics.non_abstained_accuracy,
        "two_source_blocks": metrics.two_source_blocks,
        "two_source_block_rate": metrics.two_source_block_rate,
        "blocked_leaning": metrics.blocked_leaning,
        "blocked_lean_correct": metrics.blocked_lean_correct,
        "blocked_lean_accuracy": metrics.blocked_lean_accuracy,
        "total_latency_seconds": metrics.total_latency_seconds,
        "mean_latency_seconds": metrics.mean_latency_seconds,
        "macro_f1": metrics.macro_f1,
        "per_class": [asdict(item) for item in metrics.per_class],
        "confusion_matrix": {
            "labels": [label.value for label in BenchLabel],
            "rows_are_gold": [list(row) for row in metrics.confusion],
        },
    }


def _dataset_payload(run: DatasetRun) -> dict[str, Any]:
    dataset = run.dataset
    return {
        "name": dataset.name,
        "domain": dataset.domain,
        "domain_display": DOMAIN_DISPLAY.get(dataset.domain, dataset.domain),
        "version": dataset.version,
        "split": dataset.split,
        "source_url": dataset.source_url,
        "sha256": dataset.sha256,
        "corpus_sha256": dataset.extra_sha256,
        "license_note": dataset.license_note,
        "mapping_notes": list(dataset.mapping_notes),
        "skipped_while_loading": dataset.skipped,
        "sampled_ids": list(run.sampled_ids),
        "metrics": metrics_payload(run.metrics),
    }


def _outcome_payload(outcome: Outcome) -> dict[str, Any]:
    return {
        "id": outcome.example_id,
        "dataset": outcome.dataset,
        "domain": outcome.domain,
        "gold": outcome.gold.value,
        "predicted": outcome.predicted.value,
        "pipeline_verdict": outcome.verdict,
        "raw_label": outcome.raw_label,
        "experts": list(outcome.experts),
        "blocked_by_two_source_rule": outcome.blocked,
        "blocked_evidence_lean": outcome.lean.value if outcome.lean else None,
        "latency_seconds": outcome.latency_s,
        "error": outcome.error,
        "detail": outcome.detail,
        "text": outcome.text,
    }


def _metrics_lines(metrics: Metrics) -> list[str]:
    lines = [
        f"- Accuracy: {_ratio(metrics.accuracy, metrics.correct, metrics.n)}",
        f"- Macro-F1: {_maybe(metrics.macro_f1)}",
        f"- Abstention rate: {_ratio(metrics.abstention_rate, metrics.abstentions, metrics.n)}",
        (
            "- Accuracy on non-abstained claims: "
            f"{_ratio(metrics.non_abstained_accuracy, metrics.non_abstained_correct, metrics.non_abstained_n)}"
        ),
        (
            "- Two-source blocks: "
            f"{_ratio(metrics.two_source_block_rate, metrics.two_source_blocks, metrics.n)}"
        ),
        (
            "- Blocked rows whose evidence leaned the gold way: "
            f"{_ratio(metrics.blocked_lean_accuracy, metrics.blocked_lean_correct, metrics.blocked_leaning)}"
        ),
        f"- Mean latency: {_seconds(metrics.mean_latency_seconds)}",
        "",
        "| Class | Precision | Recall | F1 | tp | fp | fn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_class_row(score) for score in metrics.per_class)
    lines.extend(
        [
            "",
            "Confusion matrix, rows are gold, columns are predicted in the order supported, refuted, unverified.",
            "",
            "| gold \\ predicted | supported | refuted | unverified |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    labels = (BenchLabel.SUPPORTED, BenchLabel.REFUTED, BenchLabel.UNVERIFIED)
    for label, row in zip(labels, metrics.confusion, strict=True):
        cells = " | ".join(str(value) for value in row)
        lines.append(f"| {label.value} | {cells} |")
    return lines


def _class_row(score: ClassScore) -> str:
    return (
        f"| {score.label} | {_maybe(score.precision)} | {_maybe(score.recall)} | "
        f"{_maybe(score.f1)} | {score.tp} | {score.fp} | {score.fn} |"
    )


def _ratio(value: float | None, numerator: int, denominator: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} ({numerator}/{denominator})"


def _maybe(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} s"


def _limit_text(limit: int | None) -> str:
    if limit is None or limit <= 0:
        return "all rows"
    return str(limit)
