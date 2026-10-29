"""Run the router and fold expert results into one verdict per claim."""

from __future__ import annotations

from dataclasses import dataclass

from verifai.claims import extract_claims
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


def run_check(router: Router, text: str, *, limit: int = 5) -> CheckBundle:
    results = [expert.check(text) for expert in router.select(text) if expert.name == "citation"]
    combined: list[CombinedClaim] = []
    for claim in _subjects(text, limit=limit):
        experts = [expert for expert in router.select(claim) if expert.name != "citation"]
        claim_results = [_check(expert, claim) for expert in experts]
        results.extend(result for result in claim_results if result is not None)
        present = [result for result in claim_results if result is not None]
        if present:
            combined.append(combine_claim(claim, present))
    verdict, confidence, explanation = rollup(combined, results)
    return CheckBundle(results, combined, verdict, confidence, explanation)


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


def _subjects(text: str, *, limit: int) -> list[str]:
    claims = extract_claims(text, limit=limit)
    if claims:
        return claims
    cleaned = " ".join((text or "").split())
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
