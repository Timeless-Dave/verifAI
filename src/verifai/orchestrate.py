"""Run the router and fold expert results into one verdict per claim."""

from __future__ import annotations

from dataclasses import dataclass

from verifai.claims import find_claims, only_links, without_references
from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.experts.base import Expert
from verifai.experts.claim_check import NOT_MEDICAL_ADVICE
from verifai.experts.router import Router, domains_for
from verifai.schemas import CombinedClaim, EvidenceItem, ExpertResult, Verdict


@dataclass(frozen=True)
class CheckBundle:
    results: list[ExpertResult]
    combined: list[CombinedClaim]
    verdict: Verdict
    confidence: float
    explanation: str
    claims_found: int = 0
    claims_checked: int = 0


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
    verdict, confidence, explanation = rollup(combined, results)
    if len(subjects) > len(checked):
        explanation = (
            f"{explanation} Only the first {len(checked)} of {len(subjects)} claims found were "
            "checked. The rest were not looked at, so paste fewer claims at a time to check them."
        )
    return CheckBundle(
        results,
        combined,
        verdict,
        confidence,
        explanation,
        claims_found=len(subjects),
        claims_checked=len(checked),
    )


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
