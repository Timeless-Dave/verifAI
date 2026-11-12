from verifai.orchestrate import CheckBundle
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    CombinedClaim,
    EvidenceItem,
    ExpertResult,
    Verdict,
)
from verifai.web.view import build_view, claim_headline


def _item(source: str, stance: str, *, family: str | None = None, doc: str = "") -> EvidenceItem:
    return EvidenceItem(
        source=source,
        title=f"{source} {doc}",
        stance=stance,
        family=family or source,
        document_ids=[doc] if doc else [],
    )


def test_headlines_say_what_happened_in_one_line() -> None:
    wiki = _item("wikipedia", "supports")
    assert claim_headline(Verdict.SUPPORTED, ["Wikipedia", "Wikidata"], [], [wiki]) == (
        "2 independent sources back this up."
    )
    assert claim_headline(Verdict.UNVERIFIABLE, ["Wikipedia"], [], [wiki]).startswith(
        "Only Wikipedia backs this up."
    )
    assert claim_headline(Verdict.UNVERIFIABLE, [], [], []) == "No source was found for this claim."
    assert claim_headline(Verdict.UNVERIFIABLE, ["A"], ["B"], [wiki]).startswith("The sources")


def test_page_view_counts_independent_sources_and_problems() -> None:
    claim = CombinedClaim(
        claim="MMR causes autism.",
        verdict=Verdict.UNSUPPORTED,
        confidence=0.88,
        explanation="Refuted. The source cited for this claim (x) is retracted.",
        evidence=[
            _item("pubmed", "refutes", doc="pmid:1"),
            _item("europepmc", "refutes", doc="pmid:1"),
            _item("europepmc", "refutes", doc="pmid:2"),
            _item("wikipedia", "neutral"),
        ],
        experts=["health"],
    )
    citations = ExpertResult(
        expert="citation",
        verdict=Verdict.MIXED,
        verdict_scope="citations",
        confidence=0.9,
        explanation="Checked 2.",
        citations=[
            CitationResult(
                raw="a", kind="doi", status=CitationStatus.VERIFIED, confidence=0.9, explanation="x"
            ),
            CitationResult(
                raw="b", kind="isbn", status=CitationStatus.INVALID, confidence=0.9, explanation="x"
            ),
        ],
    )
    bundle = CheckBundle(
        results=[citations],
        combined=[claim],
        verdict=Verdict.MIXED,
        confidence=0.88,
        explanation="e",
        claims_found=3,
        claims_checked=1,
    )
    view = build_view(bundle)
    card = view.claims[0]
    assert (card.support, card.refute, card.unrelated) == (0, 2, 1)  # pmid:1 counts once
    assert card.headline == "2 independent sources contradict this."
    assert card.flags == ["Cites a retracted paper"]
    assert card.health is True
    assert view.citation_problems == 1
    assert view.citation_bar is not None
    assert [(s.key, s.count) for s in view.citation_bar.segments] == [("verified", 1), ("fake", 1)]
    assert view.headline == "1 of 1 claim got a verdict; 1 of 2 citations needs attention."
