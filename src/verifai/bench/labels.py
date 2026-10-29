"""Gold labels and the map from VerifAI's verdicts.

The scored classes are supported, refuted, and unverified. ``unsupported``
on a claim is the pipeline's refuted verdict. ``unverifiable`` is unverified.
``mixed`` is scored as unverified: the pipeline did not pick one side.
That row counts as an abstention, and it is not a two-source block. A block
is an unverifiable verdict that still has a supporting or refuting stance
in the evidence the experts retrieved.
"""

from __future__ import annotations

from enum import Enum

from verifai.schemas import Verdict

FEVER_LABELS = {
    "SUPPORTS": "supported",
    "REFUTES": "refuted",
    "NOT ENOUGH INFO": "unverified",
}

# LIAR is a six-way scale. The middle labels are mixed truth values, not
# "not enough information". They stay unverified so the benchmark does not
# invent a binary the annotators refused to make.
LIAR_LABELS = {
    "true": "supported",
    "mostly-true": "supported",
    "false": "refuted",
    "pants-fire": "refuted",
    "half-true": "unverified",
    "barely-true": "unverified",
}

FEVER_MAPPING_NOTES = (
    "SUPPORTS → supported",
    "REFUTES → refuted",
    "NOT ENOUGH INFO → unverified",
)

SCIFACT_MAPPING_NOTES = (
    "evidence labels are only SUPPORT → supported",
    "evidence labels are only CONTRADICT → refuted",
    "no labeled evidence, or both SUPPORT and CONTRADICT → unverified",
)

LIAR_MAPPING_NOTES = (
    "true and mostly-true → supported",
    "false and pants-fire → refuted",
    "half-true and barely-true → unverified",
    "Half-true and barely-true are mixed truth values, not FEVER's NOT ENOUGH INFO.",
)

CITATION_MAPPING_NOTES = (
    "The citation expert's verdict is scored: supported, unsupported → refuted, unverifiable → unverified.",
    "A mixed citation verdict (a partial match, or mixed rows) is scored unverified.",
    "Gold supported: a real work or a real page. A URL is supported when the page exists.",
    "Gold refuted: a citation written to be fake.",
    "Gold unverified: a real work cited with a wrong detail on purpose.",
    "The citation checker does not use the two-source claim rule. One index can still verify a citation.",
)


class BenchLabel(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNVERIFIED = "unverified"


VERDICT_TO_BENCH = {
    Verdict.SUPPORTED: BenchLabel.SUPPORTED,
    Verdict.UNSUPPORTED: BenchLabel.REFUTED,
    Verdict.UNVERIFIABLE: BenchLabel.UNVERIFIED,
    Verdict.MIXED: BenchLabel.UNVERIFIED,
}


def bench_label(raw: str) -> BenchLabel:
    try:
        return BenchLabel(raw.strip().casefold())
    except ValueError as exc:
        raise ValueError(f"Label {raw!r} is not supported, refuted, or unverified.") from exc


def scifact_gold(evidence: object) -> BenchLabel:
    """Collapse SciFact's per-rationale labels to one claim label."""
    if not isinstance(evidence, dict) or not evidence:
        return BenchLabel.UNVERIFIED
    found: set[str] = set()
    for rationales in evidence.values():
        if not isinstance(rationales, list):
            continue
        for rationale in rationales:
            if not isinstance(rationale, dict):
                continue
            label = rationale.get("label")
            if isinstance(label, str) and label:
                found.add(label)
    if found == {"SUPPORT"}:
        return BenchLabel.SUPPORTED
    if found == {"CONTRADICT"}:
        return BenchLabel.REFUTED
    return BenchLabel.UNVERIFIED
