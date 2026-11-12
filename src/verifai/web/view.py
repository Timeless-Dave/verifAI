"""Turn a check into what the page shows: tones, icons, short sentences, bars.

The templates stay simple, and the wording lives here where it is tested.
Every tone travels with an icon and a label, so color never carries meaning
alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from verifai.experts.citation.extract import dangling_markers, unresolvable_mentions
from verifai.orchestrate import CheckBundle
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    CombinedClaim,
    EvidenceItem,
    Verdict,
    source_label,
)

_LITERATURE = {"pubmed", "europepmc"}
_AMBIGUOUS_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")


@dataclass(frozen=True)
class Badge:
    tone: str  # good, critical, warning, serious, neutral
    icon: str  # an id in the base.html icon sprite
    label: str


@dataclass(frozen=True)
class Segment:
    key: str
    label: str
    count: int
    tone: str
    icon: str


@dataclass(frozen=True)
class Bar:
    title: str
    total: int
    segments: list[Segment]


@dataclass(frozen=True)
class EvidenceGroup:
    label: str
    tone: str
    items: list[EvidenceItem]


@dataclass(frozen=True)
class ClaimCard:
    text: str
    badge: Badge
    headline: str
    confidence: float | None
    support: int
    refute: int
    unrelated: int
    experts: list[str]
    flags: list[str]
    explanation: str
    groups: list[EvidenceGroup]
    health: bool
    judged: bool = True
    found: int = 0
    # The English the sources were searched with, for a claim in another language.
    translation: str | None = None


@dataclass(frozen=True)
class CitationCard:
    badge: Badge
    kind: str
    raw: str
    matched_title: str | None
    matched_meta: str
    matched_url: str | None
    mismatches: list[str]
    occurrences: int
    explanation: str
    sources: list[str]
    sources_down: list[str]
    confidence: float


@dataclass(frozen=True)
class PageView:
    badge: Badge
    headline: str
    confidence: float | None
    claims_found: int
    claims_checked: int
    citation_total: int
    citation_problems: int
    claim_bar: Bar | None
    citation_bar: Bar | None
    claims: list[ClaimCard]
    citations: list[CitationCard]
    skipped: list[str] = field(default_factory=list)
    judged: bool = True
    confidence_label: str = "Confidence"
    ignored: list[str] = field(default_factory=list)
    unchecked_pointers: list[str] = field(default_factory=list)


VERDICT_BADGES = {
    Verdict.SUPPORTED: Badge("good", "i-check", "Supported"),
    Verdict.UNSUPPORTED: Badge("critical", "i-cross", "Refuted"),
    Verdict.MIXED: Badge("warning", "i-split", "Mixed"),
    Verdict.UNVERIFIABLE: Badge("neutral", "i-question", "Unverified"),
}

CITATION_BADGES = {
    CitationStatus.VERIFIED: Badge("good", "i-check", "Verified"),
    CitationStatus.PARTIALLY_MATCHES: Badge("warning", "i-split", "Partial match"),
    CitationStatus.NOT_FOUND: Badge("critical", "i-cross", "Not found"),
    CitationStatus.INVALID: Badge("critical", "i-ban", "Invalid"),
    CitationStatus.WRONG_WORK: Badge("critical", "i-cross", "Wrong paper"),
    CitationStatus.RETRACTED: Badge("serious", "i-alert", "Retracted"),
    CitationStatus.UNREACHABLE: Badge("neutral", "i-cloud", "Couldn't check"),
}

KIND_LABELS = {
    "doi": "DOI",
    "url": "Link",
    "reference": "Reference",
    "isbn": "ISBN",
    "pmid": "PMID",
    "pmc": "PMC",
}
_ARXIV = re.compile(r"\b(?:arxiv[:./]|10\.48550/arxiv\.)", re.IGNORECASE)
EXPERT_LABELS = {"health": "Health", "news": "News", "general": "General", "citation": "Citations"}
_SIDES = {"supports", "refutes"}
_PROBLEMS = {
    CitationStatus.NOT_FOUND,
    CitationStatus.INVALID,
    CitationStatus.WRONG_WORK,
    CitationStatus.RETRACTED,
    CitationStatus.PARTIALLY_MATCHES,
}


def build_view(bundle: CheckBundle, *, judged: bool = True, text: str = "") -> PageView:
    citations = [item for result in bundle.results for item in result.citations]
    claims = [claim_card(claim, judged=judged) for claim in bundle.combined]
    problems = sum(1 for item in citations if item.status in _PROBLEMS)
    resolved = sum(1 for claim in bundle.combined if claim.verdict is not Verdict.UNVERIFIABLE)
    # A confidence number only means something for what was resolved. With
    # claims and none resolved, show none; with only citations, say so.
    confidence: float | None = bundle.confidence
    label = "Confidence"
    if bundle.verdict is Verdict.UNVERIFIABLE or (bundle.combined and not resolved):
        confidence = None
    elif not resolved and citations:
        label = "Citation check confidence"
    pointers = dangling_markers(text) + unresolvable_mentions(text) if text else []
    badge = VERDICT_BADGES[bundle.verdict]
    # One supported claim beside two unverified ones is not a supported paste.
    if bundle.combined and 0 < resolved < len(bundle.combined):
        if bundle.verdict is Verdict.SUPPORTED:
            badge = Badge("good", "i-check", "Partly supported")
        elif bundle.verdict is Verdict.UNSUPPORTED:
            badge = Badge("critical", "i-cross", "Partly refuted")
    return PageView(
        badge=badge,
        headline=_page_headline(bundle, citations, problems),
        confidence=confidence,
        confidence_label=label,
        judged=judged,
        ignored=list(bundle.ignored),
        unchecked_pointers=pointers,
        claims_found=bundle.claims_found,
        claims_checked=bundle.claims_checked,
        citation_total=len(citations),
        citation_problems=problems,
        claim_bar=_claim_bar(bundle.combined),
        citation_bar=_citation_bar(citations),
        claims=claims,
        citations=[citation_card(item) for item in citations],
        skipped=list(bundle.claims_skipped),
    )


def claim_card(claim: CombinedClaim, *, judged: bool = True) -> ClaimCard:
    # The same article can come back from several checkers or reviews. Show
    # it once, keeping the copy that took a side.
    unique: dict[str, EvidenceItem] = {}
    for item in claim.evidence:
        key = item.url or f"{item.source}:{item.title}"
        kept = unique.get(key)
        if kept is None or (kept.stance not in _SIDES and item.stance in _SIDES):
            unique[key] = item
    claim = claim.model_copy(update={"evidence": list(unique.values())})
    support = _sources(claim.evidence, "supports")
    refute = _sources(claim.evidence, "refutes")
    unrelated = sum(1 for item in claim.evidence if item.stance not in {"supports", "refutes"})
    flags = []
    date = _AMBIGUOUS_DATE.search(claim.claim)
    if (
        date
        and date.group(1) != date.group(2)
        and max(int(date.group(1)), int(date.group(2))) <= 12
    ):
        flags.append(f"Ambiguous date {date.group(0)}: day and month can be read either way")
    if "is retracted" in claim.explanation:
        flags.append("Cites a retracted paper")
    if "may be made up" in claim.explanation:
        flags.append("Cites a source that could not be found")
    unreadable = "could not translate this claim reliably" in claim.explanation
    if unreadable:
        flags.append("Translation uncertain: try it in English")
    if claim.verdict is Verdict.UNVERIFIABLE and refute and not support:
        flags.append("Every source that took a side contradicts it")
    elif claim.verdict is Verdict.UNVERIFIABLE and support and not refute:
        flags.append("Leans supported: one source so far")
    by_stance = {
        "supports": [item for item in claim.evidence if item.stance == "supports"],
        "refutes": [item for item in claim.evidence if item.stance == "refutes"],
        "other": [item for item in claim.evidence if item.stance not in {"supports", "refutes"}],
    }
    groups = [
        EvidenceGroup(label, tone, by_stance[key])
        for key, label, tone in (
            ("supports", "Supports the claim", "good"),
            ("refutes", "Contradicts the claim", "critical"),
            ("other", "Not about it, or no clear answer", "neutral"),
        )
        if by_stance[key]
    ]
    decided = claim.verdict is not Verdict.UNVERIFIABLE
    if not judged and not decided:
        badge = Badge("neutral", "i-question", "Not judged")
        headline = (
            f"{_count(len(claim.evidence), 'source')} found. Claim verdicts are paused for the "
            "moment, so they were not weighed."
            if claim.evidence
            else "No source was found for this claim."
        )
    else:
        badge = VERDICT_BADGES[claim.verdict]
        headline = claim_headline(claim.verdict, support, refute, claim.evidence)
        if unreadable:
            headline = "Two translations disagreed about what this says, so no source was counted."
    return ClaimCard(
        text=claim.claim,
        badge=badge,
        headline=headline,
        judged=judged or decided,
        found=len(claim.evidence),
        confidence=claim.confidence if decided else None,
        support=len(support),
        refute=len(refute),
        unrelated=unrelated,
        experts=[EXPERT_LABELS.get(name, name.title()) for name in claim.experts],
        flags=flags,
        explanation=claim.explanation,
        groups=groups,
        health="health" in claim.experts,
        translation=_translation(claim.explanation),
    )


_TRANSLATED = re.compile(r"searched with the English translation “([^”]+)”")


def _translation(explanation: str) -> str | None:
    match = _TRANSLATED.search(explanation)
    return match.group(1) if match else None


def claim_headline(
    verdict: Verdict, support: list[str], refute: list[str], evidence: list[EvidenceItem]
) -> str:
    if verdict is Verdict.SUPPORTED:
        return f"{_count(len(support), 'independent source')} back this up."
    if verdict is Verdict.UNSUPPORTED:
        return f"{_count(len(refute), 'independent source')} contradict this."
    if verdict is Verdict.MIXED:
        return "Independent sources disagree about this."
    if support and refute:
        return "The sources that answered disagree, and neither side has two."
    if len(support) == 1 and not refute:
        return f"Only {support[0]} backs this up. A verdict needs two independent sources."
    if len(refute) == 1 and not support:
        return f"Only {refute[0]} contradicts this. A verdict needs two independent sources."
    if not evidence:
        return "No source was found for this claim."
    return "No source clearly addressed this claim."


def citation_card(item: CitationResult) -> CitationCard:
    meta = " · ".join(
        str(part) for part in (item.matched_authors, item.matched_year, item.matched_venue) if part
    )
    raw = item.raw if len(item.raw) <= 220 else item.raw[:217].rstrip() + "…"
    return CitationCard(
        badge=CITATION_BADGES[item.status],
        kind="arXiv" if _ARXIV.search(item.raw) else KIND_LABELS.get(item.kind, item.kind.upper()),
        raw=raw,
        matched_title=item.matched_title,
        matched_meta=meta,
        matched_url=item.matched_url,
        mismatches=item.mismatches,
        occurrences=item.occurrences,
        explanation=item.explanation,
        sources=item.sources_contributed,
        sources_down=item.sources_unavailable,
        confidence=item.confidence,
    )


def _sources(evidence: Iterable[EvidenceItem], stance: str) -> list[str]:
    """Independent sources that took this side, named for display.

    Papers count one per document. Anything else counts one per publisher,
    so three Wikipedia articles are one source.
    """
    names: dict[str, str] = {}
    for item in evidence:
        if item.stance != stance:
            continue
        family = item.family or item.source
        if family in _LITERATURE:
            key = item.document_ids[0] if item.document_ids else f"title:{item.title}"
        else:
            key = f"family:{family}"
        names.setdefault(key, _source_name(item))
    return list(names.values())


def _source_name(item: EvidenceItem) -> str:
    return item.speaker or source_label(item.source)


def _claim_bar(claims: list[CombinedClaim]) -> Bar | None:
    if not claims:
        return None
    order = (Verdict.SUPPORTED, Verdict.UNSUPPORTED, Verdict.MIXED, Verdict.UNVERIFIABLE)
    segments = [
        Segment(verdict.value, badge.label, count, badge.tone, badge.icon)
        for verdict in order
        if (count := sum(1 for claim in claims if claim.verdict is verdict))
        and (badge := VERDICT_BADGES[verdict])
    ]
    return Bar("Claims", len(claims), segments)


def _citation_bar(citations: list[CitationResult]) -> Bar | None:
    if not citations:
        return None
    groups = (
        ("verified", "Verified", "good", "i-check", {CitationStatus.VERIFIED}),
        ("partial", "Partial match", "warning", "i-split", {CitationStatus.PARTIALLY_MATCHES}),
        # Retracted sits between partial and fake so the two closest inks
        # (ochre and red) never touch.
        ("retracted", "Retracted", "serious", "i-alert", {CitationStatus.RETRACTED}),
        (
            "fake",
            "Fake, invalid, or wrong paper",
            "critical",
            "i-cross",
            {CitationStatus.NOT_FOUND, CitationStatus.INVALID, CitationStatus.WRONG_WORK},
        ),
        ("unchecked", "Couldn't check", "neutral", "i-cloud", {CitationStatus.UNREACHABLE}),
    )
    segments = [
        Segment(key, label, count, tone, icon)
        for key, label, tone, icon, statuses in groups
        if (count := sum(1 for item in citations if item.status in statuses))
    ]
    return Bar("Citations", len(citations), segments)


def _page_headline(bundle: CheckBundle, citations: list[CitationResult], problems: int) -> str:
    parts: list[str] = []
    if bundle.combined:
        decided = sum(1 for claim in bundle.combined if claim.verdict is not Verdict.UNVERIFIABLE)
        parts.append(
            f"{decided} of {_count(len(bundle.combined), 'claim')} got a verdict"
            if decided
            else f"None of the {_count(len(bundle.combined), 'claim')} could be confirmed or refuted"
        )
    if citations:
        if problems:
            verb = "needs" if problems == 1 else "need"
            parts.append(f"{problems} of {_count(len(citations), 'citation')} {verb} attention")
        else:
            parts.append(f"all {_count(len(citations), 'citation')} check out")
    if not parts:
        return "Nothing checkable was found in this paste."
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
