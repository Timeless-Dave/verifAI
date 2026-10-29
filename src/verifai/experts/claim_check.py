"""Run retrievers and the corroboration rule for one claim."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.models.judge import Judge
from verifai.models.parsing import Judgment
from verifai.retrieval.types import Passage, RetrievalResult, Retriever
from verifai.schemas import EvidenceItem, ExpertResult, ordered_sources, source_label

NOT_MEDICAL_ADVICE = (
    "This is not medical advice. It does not diagnose, treat, or recommend care. "
    "It only reports what the retrieved studies say."
)


def check_claim(
    *,
    expert: str,
    scope: str,
    claim: str,
    retrievers: Sequence[Retriever],
    judge: Judge,
    workers: int,
    disclaimer: str | None = None,
    note: str | None = None,
) -> ExpertResult:
    outcomes = _retrieve_all(retrievers, claim, workers)
    evidence: list[EvidenceItem] = []
    votes: list[SourceVote] = []
    consulted: list[str] = []
    contributed: list[str] = []
    unavailable: list[str] = []
    saw_review = False
    unjudged = False
    for outcome in outcomes:
        consulted.append(outcome.source)
        if outcome.unavailable:
            unavailable.append(outcome.source)
            continue
        contributed.append(outcome.source)
        for passage in outcome.passages:
            judgment = _judge(judge, claim, passage)
            if passage.publication_type and "review" in passage.publication_type:
                saw_review = True
            if passage.native_stance is None and passage.text and not judge.available():
                unjudged = True
            evidence.append(
                _item(passage, judgment.stance, judgment.confidence, judgment.explanation)
            )
            if judgment.stance is not Stance.NEUTRAL:
                votes.append(
                    SourceVote(
                        source=passage.source,
                        family=passage.family or passage.source,
                        stance=judgment.stance,
                        confidence=judgment.confidence,
                        document_ids=passage.document_ids,
                        url=passage.url,
                        title=passage.title,
                        detail=judgment.explanation,
                    )
                )
    decision = corroborate(votes)
    pieces = [decision.explanation]
    if unjudged:
        pieces.append(
            "No local model is loaded, so retrieved pages were not judged for support "
            "or contradiction. Fact-check ratings are still read."
        )
    if saw_review:
        pieces.append("Systematic reviews and meta-analyses were preferred over single studies.")
    if unavailable:
        labels = ", ".join(source_label(name) for name in ordered_sources(unavailable))
        pieces.append(f"No answer from {labels}.")
    if note:
        pieces.append(note)
    if disclaimer:
        pieces.append(disclaimer)
    return ExpertResult(
        expert=expert,
        verdict=decision.verdict,
        verdict_scope=scope,
        confidence=decision.confidence,
        explanation=" ".join(pieces),
        evidence=evidence,
        sources_consulted=ordered_sources(consulted),
        sources_contributed=ordered_sources(contributed),
        sources_unavailable=ordered_sources(unavailable),
        subject=claim,
    )


def _judge(judge: Judge, claim: str, passage: Passage) -> Judgment:
    if passage.native_stance is not None:
        return Judgment(
            passage.native_stance,
            passage.native_confidence if passage.native_confidence is not None else 0.8,
            passage.rating or "The source published this rating.",
        )
    if not passage.text.strip() or not judge.available():
        return Judgment(Stance.NEUTRAL, 0.0, "")
    return judge.judge(claim, passage.text[:3500])


def _item(passage: Passage, stance: Stance, confidence: float, explanation: str) -> EvidenceItem:
    snippet = passage.text[:400]
    if explanation:
        snippet = f"{snippet} {explanation}".strip()[:500]
    return EvidenceItem(
        source=passage.source,
        title=passage.title,
        url=passage.url,
        authors=list(passage.authors),
        year=passage.year,
        venue=passage.venue,
        snippet=snippet or None,
        score=confidence,
        stance=stance.value,
        family=passage.family,
        document_ids=list(passage.document_ids),
        speaker=passage.speaker,
    )


def _retrieve_all(
    retrievers: Sequence[Retriever], query: str, workers: int
) -> list[RetrievalResult]:
    if not retrievers:
        return []
    if len(retrievers) == 1 or workers <= 1:
        return [retriever.retrieve(query) for retriever in retrievers]
    with ThreadPoolExecutor(max_workers=min(workers, len(retrievers))) as pool:
        return list(pool.map(_call, retrievers, [query] * len(retrievers)))


def _call(retriever: Retriever, query: str) -> RetrievalResult:
    return retriever.retrieve(query)


def google_note(*, configured: bool) -> str | None:
    if configured:
        return None
    return (
        "Google Fact Check is not configured, so it was not consulted. "
        "When a key is set, its reviews still count as one source."
    )
