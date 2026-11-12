"""Run retrievers and the corroboration rule for one claim."""

from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.models.judge import Judge
from verifai.models.parsing import Judgment
from verifai.retrieval.textutil import detect_language, search_terms_for
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
            judgment = screen(claim, passage, _judge(judge, claim, passage))
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
    if passage.native_stance is not None and passage.reviewed_claim is None:
        # A rating published for this exact claim, as in benchmark fixtures.
        return Judgment(
            passage.native_stance,
            passage.native_confidence if passage.native_confidence is not None else 0.8,
            passage.rating or "The source published this rating.",
        )
    if passage.native_stance is not None:
        return _judge_review(judge, claim, passage)
    if not passage.text.strip() or not judge.available():
        return Judgment(Stance.NEUTRAL, 0.0, "")
    return judge.judge(claim, passage.text[:3500])


def _judge_review(judge: Judge, claim: str, passage: Passage) -> Judgment:
    """A fact check rates the statement it reviewed, not necessarily this claim.

    A review of "vaccines cause autism" rated False refutes that statement,
    which supports the claim "vaccines do not cause autism". With a model
    loaded, the judge reads the review and decides what it means for the
    claim. Without one, the rating counts only when the reviewed statement
    is this claim, worded nearly the same way and negated the same way.
    """
    native = passage.native_stance or Stance.NEUTRAL
    native_confidence = passage.native_confidence if passage.native_confidence is not None else 0.8
    if native is Stance.NEUTRAL:
        return Judgment(Stance.NEUTRAL, 0.0, passage.rating or "")
    if judge.available():
        judged = judge.judge(claim, passage.text[:3500])
        if judged.stance is Stance.NEUTRAL:
            return judged
        return Judgment(
            judged.stance, min(judged.confidence, native_confidence), judged.explanation
        )
    if same_statement(claim, passage.reviewed_claim or ""):
        return Judgment(
            native, native_confidence, passage.rating or "The fact checker rated this claim."
        )
    return Judgment(
        Stance.NEUTRAL,
        0.0,
        "The fact check reviewed a different statement, and no model is loaded to compare them.",
    )


_NEGATION = re.compile(
    r"\b(?:not|no|never|n't|cannot|isn't|doesn't|don't|didn't|wasn't|won't)\b", re.IGNORECASE
)


def same_statement(left: str, right: str) -> bool:
    """Nearly the same words, and negated the same way."""
    if not left.strip() or not right.strip():
        return False
    if bool(_NEGATION.search(left)) != bool(_NEGATION.search(right)):
        return False
    return fuzz.token_set_ratio(left.casefold(), right.casefold()) >= 90 and (
        fuzz.ratio(left.casefold(), right.casefold()) >= 75
    )


# Words that appear in many claims and say nothing about the topic.
_GENERIC = frozenset(
    [
        "study",
        "studies",
        "proved",
        "prove",
        "proves",
        "shows",
        "show",
        "showed",
        "found",
        "claim",
        "claims",
        "research",
        "evidence",
        "people",
        "general",
        "population",
        "approximately",
        "official",
        "greatly",
        "contrary",
        "popular",
        "belief",
        "first",
        "described",
        "really",
        "actually",
        "known",
        "widely",
        "said",
        "says",
        "according",
        "true",
        "false",
    ]
)
_LITERATURE = frozenset({"pubmed", "europepmc"})
_NOT_ABOUT = re.compile(
    r"\b(?:does not|doesn't|do not|did not|never|no)\s+(?:\w+\s+){0,2}"
    r"(?:mention|address|discuss|refer to|relate to|concern|cover|examine|study|test)",
    re.IGNORECASE,
)


def key_terms(claim: str) -> list[str]:
    lang = detect_language(claim)
    return [term for term in search_terms_for(claim, lang, limit=12) if term not in _GENERIC]


def on_topic(claim: str, passage: Passage) -> bool:
    """The passage names enough of the claim's key terms to be about it.

    A heart-muscle paper is not evidence about MMR and autism, whatever a
    small model says. Terms match on their first five letters, so
    "vaccine" also finds "vaccination".
    """
    terms = key_terms(claim)
    if not terms:
        return True
    haystack = f"{passage.title} {passage.text}".casefold()

    def present(term: str) -> bool:
        stem = term[:5] if len(term) > 5 else term
        return bool(re.search(rf"\b{re.escape(stem)}", haystack))

    stems = {term[:5] if len(term) > 5 else term for term in terms}
    if sum(1 for term in set(terms) if present(term)) < min(2, len(stems)):
        return False
    # "Smoking cigarettes increases lung cancer risk" is about smoking. A
    # salt-and-cancer paper shares "cancer" and "risk" but not the subject.
    subject = [term for term in key_terms(_subject(claim)) if term in terms]
    title = passage.title.casefold()

    def central(term: str) -> bool:
        """In the title, or more than a passing mention in the text."""
        stem = re.escape(term[:5] if len(term) > 5 else term)
        return bool(re.search(rf"\b{stem}", title)) or len(re.findall(rf"\b{stem}", haystack)) >= 2

    # Only for research papers: an encyclopedia item for "the 2016 election"
    # rightly answers a claim about who won it without naming the claimant.
    if passage.family not in _LITERATURE:
        return True
    return not subject or any(central(term) for term in subject)


_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|may|might|"
    r"causes?|caused|prevents?|increases?|reduces?|lowers?|raises?|cures?|leads?|"
    r"won|said|says|created|published|described|proved)\b",
    re.IGNORECASE,
)


def _subject(claim: str) -> str:
    """The words before the first verb: "Smoking cigarettes" in "Smoking cigarettes increases…"."""
    match = _VERB.search(claim)
    return claim[: match.start()] if match else ""


def screen(claim: str, passage: Passage, judgment: Judgment) -> Judgment:
    """Drop a stance the passage cannot carry: off topic, or contradicted by its own reason."""
    if judgment.stance is Stance.NEUTRAL:
        return judgment
    if passage.native_stance is not None and passage.reviewed_claim is None:
        return judgment  # a rating published for this exact claim
    if not on_topic(claim, passage):
        return Judgment(
            Stance.NEUTRAL, 0.0, "Not counted: this source is not about the claim's main subject."
        )
    if _NOT_ABOUT.search(judgment.explanation):
        return Judgment(
            Stance.NEUTRAL,
            0.0,
            "Not counted: the model's own reason says the source does not address the claim.",
        )
    return judgment


def clip_words(text: str, limit: int) -> str:
    """Shorten at a word boundary and mark the cut."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


def _item(passage: Passage, stance: Stance, confidence: float, explanation: str) -> EvidenceItem:
    snippet = clip_words(passage.text, 320)
    return EvidenceItem(
        source=passage.source,
        title=passage.title,
        url=passage.url,
        authors=list(passage.authors),
        year=passage.year,
        venue=passage.venue,
        snippet=snippet or None,
        reason=clip_words(explanation, 300) or None,
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
