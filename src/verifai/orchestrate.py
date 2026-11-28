"""Run the router and fold expert results into one verdict per claim."""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from verifai.citelink import Link as CitationLink
from verifai.citelink import link_sources
from verifai.claims import only_links, opinions, screen_with_citations, without_references
from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.experts.base import Expert
from verifai.experts.claim_check import NOT_MEDICAL_ADVICE
from verifai.experts.router import Router, domains_for
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    CombinedClaim,
    EvidenceItem,
    ExpertResult,
    SourceCheck,
    Verdict,
)
from verifai.source_check import as_evidence, claim_content


@dataclass(frozen=True)
class CheckBundle:
    results: list[ExpertResult]
    combined: list[CombinedClaim]
    verdict: Verdict
    confidence: float
    explanation: str
    claims_found: int = 0
    claims_checked: int = 0
    claims_skipped: tuple[str, ...] = ()
    # Pieces of the paste set aside because they address the checker.
    ignored: tuple[str, ...] = ()
    # The paste's reference list, for the next batch of its claims.
    sources: str = ""


def run_check(router: Router, text: str, *, limit: int = 5, sources: str = "") -> CheckBundle:
    """Check a paste. ``sources`` is the reference list of an earlier batch of it."""
    searched = f"{text}\n\n{sources}" if sources.strip() else text
    results = [
        expert.check(searched) for expert in router.select(searched) if expert.name == "citation"
    ]
    citations = [
        item for result in results if result.expert == "citation" for item in result.citations
    ]
    references = _numbered_references(searched)
    subjects, ignored, written = _subjects(text)
    checked = subjects[:limit]
    checker = getattr(router, "source_checker", None)

    def check_one(claim: str) -> tuple[CombinedClaim | None, list[ExpertResult]]:
        as_written = written.get(claim, claim)
        links = link_sources(as_written, citations, references, single_claim=len(subjects) == 1)
        # A paste with references is academic writing: every claim in it is research.
        experts = _experts_for(router, claim, cites=bool(links or citations))
        # The checkers and the reading of the cited work run side by side.
        with ThreadPoolExecutor(max_workers=len(experts) + 1) as pool:
            reading = (
                pool.submit(checker.check, claim, links) if links and checker is not None else None
            )
            claim_results = list(pool.map(lambda expert: _check(expert, claim), experts))
            checks = reading.result() if reading is not None else []
        present = [result for result in claim_results if result is not None]
        if not present and not links:
            return None, present
        one = (
            combine_claim(claim, present)
            if present
            else CombinedClaim(
                claim=claim, verdict=Verdict.UNVERIFIABLE, confidence=0.0, explanation=""
            )
        )
        if as_written != claim:
            one = one.model_copy(update={"written": as_written})
        if links and checker is not None:
            one = _with_source_checks(one, checks, links)
        return one, present

    # A few claims at once: each waits mostly on sources and the model.
    with ThreadPoolExecutor(max_workers=max(1, min(_CLAIMS_AT_ONCE, len(checked)))) as pool:
        outcomes = list(pool.map(check_one, checked))
    combined: list[CombinedClaim] = []
    for one, present in outcomes:
        results.extend(present)
        if one is not None:
            combined.append(one)
    combined = [_with_citation_notes(claim, citations, references) for claim in combined]
    bundle_sources = _sources_text(citations, references)
    verdict, confidence, explanation = rollup(combined, results)
    if not subjects and not ignored and (taste := opinions(text)):
        quoted = "; ".join(f"“{item}”" for item in taste[:2])
        explanation = (
            f"That reads as opinion, not fact: {quoted}. Taste and belief cannot be "
            "checked against sources, so no verdict is given."
        )
    if len(subjects) > len(checked):
        explanation = (
            f"{explanation} Only the first {len(checked)} of {len(subjects)} claims found were "
            "checked. The others are listed below and can be checked next."
        )
    return CheckBundle(
        results,
        combined,
        verdict,
        confidence,
        explanation,
        claims_found=len(subjects),
        claims_checked=len(checked),
        claims_skipped=tuple(written.get(claim, claim) for claim in subjects[len(checked) :]),
        ignored=tuple(ignored),
        sources=bundle_sources,
    )


_CLAIMS_AT_ONCE = 3


def _experts_for(router: Router, claim: str, *, cites: bool) -> list[Expert]:
    """The claim's checkers. A claim that cites a work is research, whatever its words."""
    experts = [expert for expert in router.select(claim) if expert.name != "citation"]
    names = {expert.name for expert in experts}
    if cites and not names & {"research", "health"}:
        experts.extend(expert for expert in router.experts if expert.name == "research")
    return experts


def _with_source_checks(
    claim: CombinedClaim, checks: list[SourceCheck], links: list[CitationLink]
) -> CombinedClaim:
    """Add what the cited sources say, and count each one that took a side as a source."""
    cited = [
        item
        for check, link in zip(checks, links, strict=False)
        if (item := as_evidence(check, link.citation)) is not None
    ]
    update: dict[str, object] = {"source_checks": checks}
    if cited:
        evidence = [*claim.evidence, *cited]
        decision = corroborate(_votes(evidence))
        update.update(
            evidence=evidence,
            verdict=decision.verdict,
            confidence=decision.confidence,
            explanation=decision.explanation,
        )
    decided = _attributed_verdict(claim.claim, checks)
    if decided is not None:
        update.update(decided)
    notes = [_source_note(check) for check in checks]
    text = " ".join(note for note in notes if note)
    base = str(update.get("explanation", claim.explanation))
    if text:
        update["explanation"] = f"{base} {text}".strip()
    return claim.model_copy(update=update)


def _attributed_verdict(claim: str, checks: list[SourceCheck]) -> dict[str, object] | None:
    """A claim about what a source says ("Northey et al. found that …") is that source's to settle.

    Read against the work it names, a clear answer decides it: no second
    source can say better what Northey et al. found.
    """
    if claim_content(claim) == claim:
        return None
    named = [
        check
        for check in checks
        if check.cited_as
        and check.status in {"supports", "contradicts"}
        and check.confidence >= 0.7
    ]
    if len(named) != 1:
        return None
    check = named[0]
    says = "says this" if check.status == "supports" else "contradicts this"
    return {
        "verdict": Verdict.SUPPORTED if check.status == "supports" else Verdict.UNSUPPORTED,
        "confidence": round(0.9 * check.confidence, 2),
        "explanation": (
            f"This claim is about what {_short_title(check)} says, so that source decides it, "
            f"and its {check.read or 'text'} {says}."
        ),
    }


def _source_note(check: SourceCheck) -> str:
    name = _short_title(check)
    return {
        "supports": f"The cited source ({name}) says this.",
        "contradicts": f"The cited source ({name}) contradicts this.",
        "not_found": f"The cited source ({name}) does not say this.",
        "missing": f"The cited source ({name}) could not be found.",
    }.get(check.status, "")


def _short_title(check: SourceCheck) -> str:
    title = check.title or check.citation
    return f"“{title[:77]}…”" if len(title) > 80 else f"“{title}”"


def _sources_text(citations: list[CitationResult], references: dict[str, str]) -> str:
    """The paste's reference list, carried to later batches so their claims can cite it."""
    lines = [
        f"[{number}] {entry.split(']', 1)[-1].strip()}" for number, entry in references.items()
    ]
    numbered = " ".join(lines).casefold()
    lines.extend(item.raw for item in citations if item.raw.casefold() not in numbered)
    return "\n".join(dict.fromkeys(lines))[:20_000]


_CITATION_WARNINGS = {
    CitationStatus.RETRACTED: "is retracted",
    CitationStatus.NOT_FOUND: "could not be found, so it may be made up",
    CitationStatus.INVALID: "is not a valid identifier",
    CitationStatus.WRONG_WORK: "points to a different paper than the one named",
    CitationStatus.PARTIALLY_MATCHES: "does not match the record it points to",
}


def _numbered_references(text: str) -> dict[str, str]:
    """Reference-list entries by number: "[15] Mehra, M. R. …", "15. Mehra MR …" under "15"."""
    entries: dict[str, str] = {}
    for line in (text or "").splitlines():
        # "1." and "1)" lists (Vancouver) start with an author: "1. Doll R, Hill AB."
        match = re.match(r"\s*\[(\d{1,3})\]\s+(\S.*)", line) or re.match(
            r"\s*(\d{1,3})[.)]\s+([A-Z][\w'’-]+,?\s+[A-Z].*)", line
        )
        if match:
            entries.setdefault(match.group(1), line.strip())
    return entries


def _with_citation_notes(
    claim: CombinedClaim,
    citations: list[CitationResult],
    references: dict[str, str] | None = None,
) -> CombinedClaim:
    """Say so when the claim's own line cites a bad source.

    "This study proved that the MMR vaccine causes autism" sits after a
    retracted paper. The claim is still judged on retrieved evidence, but a
    reader should know what it leans on.
    """
    folded = " ".join(claim.claim.casefold().split())
    # The claim's own line, plus any reference it points to by number: a
    # claim ending "… as shown by Mehra et al. [15]" leans on entry [15].
    pointed = [
        " ".join(references[number].casefold().split())
        for number in re.findall(r"\[(\d{1,3})\]", claim.claim)
        if references and number in references
    ]

    def leans_on(item: CitationResult) -> bool:
        raw = " ".join(item.raw.casefold().split())
        return folded in raw or any(raw and (raw in entry or entry in raw) for entry in pointed)

    notes = [
        f"The source cited for this claim ({_short(item)}) {_CITATION_WARNINGS[item.status]}."
        for item in citations
        if item.status in _CITATION_WARNINGS and leans_on(item)
    ]
    if not notes:
        return claim
    return claim.model_copy(update={"explanation": f"{claim.explanation} {' '.join(notes)}"})


def _short(item: CitationResult) -> str:
    if item.matched_title:
        title = item.matched_title
        return f"“{title[:77]}…”" if len(title) > 80 else f"“{title}”"
    raw = item.raw
    return raw[:60] + "…" if len(raw) > 60 else raw


def combine_claim(claim: str, results: list[ExpertResult]) -> CombinedClaim:
    evidence = [item for result in results for item in result.evidence]
    if len(results) == 1:
        result = results[0]
        return CombinedClaim(
            claim=claim,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            evidence=evidence,
            experts=[result.expert],
            correction=result.correction,
        )
    decision = corroborate(_votes(evidence))
    explanation = decision.explanation
    if (
        any(result.expert == "health" for result in results)
        and NOT_MEDICAL_ADVICE not in explanation
    ):
        explanation = f"{explanation} {NOT_MEDICAL_ADVICE}"
    return CombinedClaim(
        claim=claim,
        verdict=decision.verdict,
        confidence=decision.confidence,
        explanation=explanation,
        evidence=evidence,
        experts=[result.expert for result in results],
        correction=_correction(results, decision.verdict),
    )


def _correction(results: list[ExpertResult], verdict: Verdict) -> str | None:
    """The correction from an expert that reached the same verdict, else any."""
    agreeing = [result.correction for result in results if result.verdict is verdict]
    others = [result.correction for result in results]
    return next((text for text in [*agreeing, *others] if text), None)


def rollup(
    combined: list[CombinedClaim], results: list[ExpertResult]
) -> tuple[Verdict, float, str]:
    """One verdict for the paste. Citations count beside the claims.

    An unverified side has no say. A paste whose claims are unverified but
    whose citations include a fabricated one is mixed, not unverified.
    """
    claim_side = _rollup_claims(combined, results)
    citation = next(
        (result for result in results if result.expert == "citation" and result.citations), None
    )
    if not combined or citation is None:
        return claim_side
    claim_verdict, claim_confidence, claim_text = claim_side
    cited = f"Citations: {citation.explanation.split('. ')[0].rstrip('.')}."
    misread = _misattributed(combined)
    if misread:
        cited = f"{cited} {misread}"
        if claim_verdict is Verdict.SUPPORTED or claim_verdict is Verdict.UNVERIFIABLE:
            return Verdict.MIXED, claim_confidence or 0.5, f"Claims: {claim_text} {cited}"
    if citation.verdict is Verdict.UNVERIFIABLE or citation.verdict is claim_verdict:
        return claim_verdict, claim_confidence, f"Claims: {claim_text} {cited}"
    serious = any(item.status in _SERIOUS for item in citation.citations)
    if not serious and claim_verdict is not Verdict.UNVERIFIABLE:
        # Real citations say nothing about whether the claims are true; the claims decide.
        return claim_verdict, claim_confidence, f"Claims: {claim_text} {cited}"
    if claim_verdict is Verdict.UNVERIFIABLE:
        if citation.verdict is Verdict.SUPPORTED:
            # Real papers do not make the claims true: they still have to say them.
            return (
                Verdict.UNVERIFIABLE,
                0.0,
                (
                    f"Claims: {claim_text} {cited} The cited works exist, but the claims "
                    "themselves are not yet confirmed."
                ),
            )
        return citation.verdict, citation.confidence, f"Claims: {claim_text} {cited}"
    return (
        Verdict.MIXED,
        min(claim_confidence, citation.confidence),
        f"Claims: {claim_text} {cited} The claims and the citations do not agree.",
    )


# Citation faults that matter for a paste's verdict. A year off by one for a
# preprint, or a venue nickname, does not.
_SERIOUS = frozenset(
    {
        CitationStatus.NOT_FOUND,
        CitationStatus.INVALID,
        CitationStatus.RETRACTED,
        CitationStatus.WRONG_WORK,
    }
)


def _misattributed(combined: list[CombinedClaim]) -> str:
    """Say how many claims lean on a source that contradicts them or does not say them."""
    wrong = sum(
        1
        for claim in combined
        if any(check.status in {"contradicts", "not_found"} for check in claim.source_checks)
    )
    if not wrong:
        return ""
    noun = "claim is" if wrong == 1 else "claims are"
    return f"{wrong} {noun} not backed by the source cited for {'it' if wrong == 1 else 'them'}."


def _rollup_claims(
    combined: list[CombinedClaim], results: list[ExpertResult]
) -> tuple[Verdict, float, str]:
    if len(combined) == 1:
        claim = combined[0]
        return claim.verdict, claim.confidence, claim.explanation
    if len(combined) > 1:
        return _rollup_many(combined)
    if results:
        first = results[0]
        return first.verdict, first.confidence, first.explanation
    return (
        Verdict.UNVERIFIABLE,
        0.0,
        "No checkable claim was found in that paste.",
    )


def _subjects(text: str) -> tuple[list[str], list[str], dict[str, str]]:
    claims, ignored, written = screen_with_citations(text)
    if claims:
        return claims, ignored, written
    if ignored:
        return [], ignored, {}  # never fall back to checking text that addressed the checker
    cleaned = " ".join(without_references(text).split())
    if only_links(cleaned):
        return [], ignored, {}
    if cleaned and domains_for(cleaned):
        return [cleaned[:800]], ignored, {}
    return [], ignored, {}


def _check(expert: Expert, claim: str) -> ExpertResult | None:
    return expert.check(claim)


def _votes(evidence: list[EvidenceItem]) -> list[SourceVote]:
    votes: list[SourceVote] = []
    for item in evidence:
        if item.stance not in {Stance.SUPPORTS.value, Stance.REFUTES.value}:
            continue
        try:
            stance = Stance(item.stance)
        except ValueError:
            continue
        votes.append(
            SourceVote(
                source=item.source,
                family=item.family or item.source,
                stance=stance,
                confidence=item.score if item.score is not None else 0.5,
                document_ids=tuple(item.document_ids),
                url=item.url,
                title=item.title,
                detail=item.snippet or "",
            )
        )
    return votes


def _rollup_many(combined: list[CombinedClaim]) -> tuple[Verdict, float, str]:
    kinds = {claim.verdict for claim in combined}
    if kinds == {Verdict.UNVERIFIABLE}:
        return (
            Verdict.UNVERIFIABLE,
            0.0,
            "None of the claims had two independent sources that agree.",
        )
    decided = [claim for claim in combined if claim.verdict is not Verdict.UNVERIFIABLE]
    decided_kinds = {claim.verdict for claim in decided}
    if decided_kinds == {Verdict.SUPPORTED}:
        return (
            Verdict.SUPPORTED,
            _mean(claim.confidence for claim in decided),
            f"{len(decided)} of {len(combined)} claims are supported by agreeing sources.",
        )
    if decided_kinds == {Verdict.UNSUPPORTED}:
        return (
            Verdict.UNSUPPORTED,
            _mean(claim.confidence for claim in decided),
            f"{len(decided)} of {len(combined)} claims are refuted by agreeing sources.",
        )
    return (
        Verdict.MIXED,
        _mean(claim.confidence for claim in decided) if decided else 0.0,
        "The claims do not all land on the same verdict.",
    )


def _mean(values: Iterable[float]) -> float:
    """The typical claim's confidence. The weakest one would sink eight sound ones."""
    numbers = list(values)
    return round(sum(numbers) / len(numbers), 2) if numbers else 0.0
