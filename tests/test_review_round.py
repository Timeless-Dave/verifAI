"""Fixes from the second stress-test review."""

from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.experts.base import Expert
from verifai.experts.citation.extract import extract_citations
from verifai.experts.claim_check import clip_words, on_topic, screen
from verifai.experts.router import Router
from verifai.models.parsing import Judgment
from verifai.orchestrate import run_check
from verifai.retrieval.types import Passage
from verifai.retrieval.wikidata import entity_names
from verifai.schemas import CitationResult, CitationStatus, ExpertResult, Verdict


def _passage(title: str, text: str) -> Passage:
    return Passage(source="pubmed", family="pubmed", title=title, url=None, text=text)


def test_off_topic_papers_are_not_evidence() -> None:
    claim = "This study proved that the MMR vaccine causes autism."
    heart = _passage(
        "Direct reprogramming induces vascular regeneration", "Ischemic muscle injury."
    )
    on = _passage("MMR vaccination and autism", "No link between MMR vaccine and autism.")
    assert not on_topic(claim, heart)
    assert on_topic(claim, on)
    refutes = Judgment(Stance.REFUTES, 0.9, "It finds no link.")
    assert screen(claim, heart, refutes).stance is Stance.NEUTRAL
    assert screen(claim, on, refutes).stance is Stance.REFUTES
    self_denied = Judgment(Stance.REFUTES, 0.9, "The paper never mentions the MMR vaccine.")
    assert screen(claim, on, self_denied).stance is Stance.NEUTRAL


def test_snippets_stop_at_a_word() -> None:
    assert clip_words("alpha beta gamma delta", 12) == "alpha beta…"
    assert clip_words("short", 12) == "short"


def test_a_source_that_points_both_ways_is_named() -> None:
    def vote(source: str, stance: Stance, title: str) -> SourceVote:
        return SourceVote(source=source, family=source, stance=stance, confidence=0.9, title=title)

    result = corroborate(
        [
            vote("wikipedia", Stance.SUPPORTS, "Mount Everest"),
            vote("wikipedia", Stance.REFUTES, "Timeline"),
            vote("wikidata", Stance.SUPPORTS, "Q513"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "Wikipedia also answered, but its results pointed both ways" in result.explanation


class _Stub:
    def __init__(self, name: str, result: ExpertResult) -> None:
        self.name = name
        self.description = name
        self.result = result

    def check(self, text: str) -> ExpertResult:
        del text
        return self.result


def test_claims_beside_a_retracted_citation_say_so_and_skipped_claims_are_listed() -> None:
    line = "Wakefield AJ et al. Lancet. 1998. doi:10.1/x. This study proved that the MMR vaccine causes autism."
    retracted = CitationResult(
        raw=line,
        kind="doi",
        status=CitationStatus.RETRACTED,
        confidence=0.97,
        explanation="Retracted.",
        matched_title="Ileal-lymphoid-nodular hyperplasia",
    )
    citation = _Stub(
        "citation",
        ExpertResult(
            expert="citation",
            verdict=Verdict.MIXED,
            verdict_scope="citations",
            confidence=0.97,
            explanation="Checked 1 citation.",
            citations=[retracted],
        ),
    )
    health = _Stub(
        "health",
        ExpertResult(
            expert="health",
            verdict=Verdict.UNSUPPORTED,
            verdict_scope="health",
            confidence=0.8,
            explanation="Refuted.",
        ),
    )
    router = Router(list[Expert]([citation, health]))
    text = (
        line
        + "\nVaccines cause autism in most children today.\nVaccine doses cause autism in toddlers."
    )
    bundle = run_check(router, text, limit=1)
    assert "is retracted" in bundle.combined[0].explanation
    assert "Ileal-lymphoid-nodular hyperplasia" in bundle.combined[0].explanation
    assert bundle.claims_skipped == (
        "Vaccines cause autism in most children today.",
        "Vaccine doses cause autism in toddlers.",
    )


def test_repeated_citations_are_counted() -> None:
    found = extract_citations("See 10.1038/nature14539.\nAgain: doi.org/10.1038/nature14539")
    assert len(found) == 1
    assert found[0].occurrences == 2


def test_lowercase_subjects_are_looked_up() -> None:
    assert entity_names("The speed of light is approximately 300,000 km/h.") == ["speed of light"]
