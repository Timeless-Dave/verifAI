"""Run the router and fold expert results into one verdict per claim."""

from __future__ import annotations

from dataclasses import dataclass

from verifai.claims import find_claims, only_links, without_references
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
    Verdict,
)


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


def run_check(router: Router, text: str, *, limit: int = 5) -> CheckBundle:
    results = [expert.check(text) for expert in router.select(text) if expert.name == "citation"]
    combined: list[CombinedClaim] = []
    subjects = _subjects(text)
    checked = subjects[:limit]
    for claim in checked:
        experts = [expert for expert in router.select(claim) if expert.name != "citation"]
        claim_results = [_check(expert, claim) for expert in experts]
        results.extend(result for result in claim_results if result is not None)
        present = [result for result in claim_results if result is not None]
        if present:
            combined.append(combine_claim(claim, present))
    citations = [
        item for result in results if result.expert == "citation" for item in result.citations
    ]
    combined = [_with_citation_notes(claim, citations) for claim in combined]
    verdict, confidence, explanation = rollup(combined, results)
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
        claims_skipped=tuple(subjects[len(checked) :]),
    )


_CITATION_WARNINGS = {
    CitationStatus.RETRACTED: "is retracted",
    CitationStatus.NOT_FOUND: "could not be found, so it may be made up",
    CitationStatus.INVALID: "is not a valid identifier",
    CitationStatus.PARTIALLY_MATCHES: "does not match the record it points to",
}


def _with_citation_notes(claim: CombinedClaim, citations: list[CitationResult]) -> CombinedClaim:
    """Say so when the claim's own line cites a bad source.

    "This study proved that the MMR vaccine causes autism" sits after a
    retracted paper. The claim is still judged on retrieved evidence, but a
    reader should know what it leans on.
    """
    folded = " ".join(claim.claim.casefold().split())
    notes = [
        f"The source cited for this claim ({_short(item)}) {_CITATION_WARNINGS[item.status]}."
        for item in citations
        if item.status in _CITATION_WARNINGS and folded in " ".join(item.raw.casefold().split())
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
    )


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
    if citation.verdict is Verdict.UNVERIFIABLE or citation.verdict is claim_verdict:
        return claim_verdict, claim_confidence, f"Claims: {claim_text} {cited}"
    if claim_verdict is Verdict.UNVERIFIABLE:
        return citation.verdict, citation.confidence, f"Claims: {claim_text} {cited}"
    return (
        Verdict.MIXED,
        min(claim_confidence, citation.confidence),
        f"Claims: {claim_text} {cited} The claims and the citations do not agree.",
    )


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


def _subjects(text: str) -> list[str]:
    claims = find_claims(text)
    if claims:
        return claims
    cleaned = " ".join(without_references(text).split())
    if only_links(cleaned):
        return []
    if cleaned and domains_for(cleaned):
        return [cleaned[:800]]
    return []


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
            min(claim.confidence for claim in decided),
            f"{len(decided)} of {len(combined)} claims are supported by agreeing sources.",
        )
    if decided_kinds == {Verdict.UNSUPPORTED}:
        return (
            Verdict.UNSUPPORTED,
            min(claim.confidence for claim in decided),
            f"{len(decided)} of {len(combined)} claims are refuted by agreeing sources.",
        )
    return (
        Verdict.MIXED,
        min(claim.confidence for claim in decided) if decided else 0.0,
        "The claims do not all land on the same verdict.",
    )
