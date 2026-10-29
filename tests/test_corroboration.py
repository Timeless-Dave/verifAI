from verifai.corroboration import GOOGLE_FAMILY, SourceVote, Stance, corroborate
from verifai.schemas import Verdict


def _vote(
    source: str,
    stance: Stance,
    *,
    family: str | None = None,
    confidence: float = 0.9,
    document_ids: tuple[str, ...] = (),
    title: str | None = None,
) -> SourceVote:
    return SourceVote(
        source=source,
        family=family if family is not None else source,
        stance=stance,
        confidence=confidence,
        document_ids=document_ids,
        title=title,
    )


def test_no_votes_stay_unverified() -> None:
    result = corroborate([])
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.confidence == 0.0
    assert "unverified" in result.explanation
    assert "insufficient evidence" in result.explanation


def test_neutral_votes_do_not_count() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.NEUTRAL, title="Page"),
            _vote("wikidata", Stance.NEUTRAL, title="Item"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.support == ()
    assert result.refute == ()


def test_one_source_cannot_decide() -> None:
    result = corroborate([_vote("wikipedia", Stance.SUPPORTS, title="Eiffel Tower")])
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "Wikipedia" in result.explanation
    assert "two independent" in result.explanation


def test_two_independent_sources_can_support() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, title="Eiffel Tower", confidence=0.8),
            _vote("wikidata", Stance.SUPPORTS, title="Eiffel Tower", confidence=0.8),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.confidence == 0.72
    assert "Wikipedia" in result.explanation
    assert "Wikidata" in result.explanation


def test_two_independent_sources_can_refute() -> None:
    result = corroborate(
        [
            _vote("pubmed", Stance.REFUTES, document_ids=("pmid:1",), title="Review A"),
            _vote("europepmc", Stance.REFUTES, document_ids=("pmid:2",), title="Review B"),
        ]
    )
    assert result.verdict is Verdict.UNSUPPORTED
    assert "refuted" in result.explanation


def test_google_reviews_count_as_one_source() -> None:
    reviews = [
        _vote(
            "google_factcheck",
            Stance.REFUTES,
            family=GOOGLE_FAMILY,
            title=f"Review {index}",
        )
        for index in range(5)
    ]
    alone = corroborate(reviews)
    assert alone.verdict is Verdict.UNVERIFIABLE
    assert "one source" in alone.explanation
    assert len(alone.refute) == 1

    with_wiki = corroborate([*reviews, _vote("wikipedia", Stance.REFUTES, title="Article")])
    assert with_wiki.verdict is Verdict.UNSUPPORTED
    assert len(with_wiki.refute) == 2


def test_google_disagreement_is_not_a_vote() -> None:
    result = corroborate(
        [
            _vote("google_factcheck", Stance.SUPPORTS, family=GOOGLE_FAMILY, title="True"),
            _vote("google_factcheck", Stance.REFUTES, family=GOOGLE_FAMILY, title="False"),
            _vote("wikipedia", Stance.SUPPORTS, title="Page"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert len(result.support) == 1
    assert "Wikipedia" in result.explanation


def test_google_internal_conflict_without_anyone_else() -> None:
    result = corroborate(
        [
            _vote("google_factcheck", Stance.SUPPORTS, family=GOOGLE_FAMILY),
            _vote("google_factcheck", Stance.REFUTES, family=GOOGLE_FAMILY),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "do not agree" in result.explanation


def test_same_paper_from_two_indexes_is_one_source() -> None:
    result = corroborate(
        [
            _vote(
                "pubmed",
                Stance.SUPPORTS,
                document_ids=("pmid:42", "doi:10.1000/same"),
                title="One review",
            ),
            _vote(
                "europepmc",
                Stance.SUPPORTS,
                document_ids=("doi:10.1000/same",),
                title="One review",
            ),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert len(result.support) == 1
    assert "PubMed and Europe PMC" in result.explanation


def test_same_paper_conflict_drops_the_vote() -> None:
    result = corroborate(
        [
            _vote("pubmed", Stance.SUPPORTS, document_ids=("pmid:7",), title="A"),
            _vote("europepmc", Stance.REFUTES, document_ids=("pmid:7",), title="A"),
            _vote("wikipedia", Stance.SUPPORTS, title="B"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert [group.sources for group in result.support] == [("wikipedia",)]


def test_two_different_papers_can_agree() -> None:
    result = corroborate(
        [
            _vote("pubmed", Stance.SUPPORTS, document_ids=("pmid:1",), title="Review A"),
            _vote("pubmed", Stance.SUPPORTS, document_ids=("pmid:2",), title="Review B"),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
    assert len(result.support) == 2


def test_both_sides_corroborated_is_mixed() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, title="For"),
            _vote("wikidata", Stance.SUPPORTS, title="For"),
            _vote("pubmed", Stance.REFUTES, document_ids=("pmid:1",), title="Against A"),
            _vote("europepmc", Stance.REFUTES, document_ids=("pmid:2",), title="Against B"),
        ]
    )
    assert result.verdict is Verdict.MIXED
    assert "both sides" in result.explanation
    assert result.confidence <= 0.6


def test_two_against_one_keeps_the_majority_and_names_the_dissent() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, confidence=0.9, title="A"),
            _vote("wikidata", Stance.SUPPORTS, confidence=0.9, title="B"),
            _vote("pubmed", Stance.REFUTES, document_ids=("pmid:9",), confidence=0.9, title="C"),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
    assert "disagrees" in result.explanation
    assert result.confidence < 0.72


def test_one_against_one_is_not_agreement() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, title="For"),
            _vote("wikidata", Stance.REFUTES, title="Against"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "do not agree" in result.explanation


def test_three_sources_can_raise_the_cap() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, confidence=0.95),
            _vote("wikidata", Stance.SUPPORTS, confidence=0.95),
            _vote("page", Stance.SUPPORTS, family="page:example.com", confidence=0.95),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
    assert result.confidence == 0.88


def test_confidence_is_clamped() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, confidence=5),
            _vote("wikidata", Stance.SUPPORTS, confidence=-2),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
    assert 0 <= result.confidence <= 1


def test_long_titles_are_shortened() -> None:
    result = corroborate([_vote("wikipedia", Stance.SUPPORTS, title="A" * 120)])
    assert "..." in result.explanation


def test_pages_on_one_host_are_one_source() -> None:
    result = corroborate(
        [
            _vote("page", Stance.REFUTES, family="page:example.com", title="One"),
            _vote("page", Stance.REFUTES, family="page:example.com", title="Two"),
            _vote("wikipedia", Stance.REFUTES, title="Wiki"),
        ]
    )
    assert result.verdict is Verdict.UNSUPPORTED
    assert len(result.refute) == 2


def test_refute_majority_names_the_single_dissent() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.REFUTES, confidence=0.9, title="A"),
            _vote("wikidata", Stance.REFUTES, confidence=0.9, title="B"),
            _vote("pubmed", Stance.SUPPORTS, document_ids=("pmid:3",), title="C"),
        ]
    )
    assert result.verdict is Verdict.UNSUPPORTED
    assert "disagrees" in result.explanation


def test_internal_conflict_on_one_family_drops_that_family() -> None:
    result = corroborate(
        [
            _vote("wikipedia", Stance.SUPPORTS, title="Yes"),
            _vote("wikipedia", Stance.REFUTES, title="No"),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "conflict internally" in result.explanation


def test_three_indexes_of_one_paper_are_named_together() -> None:
    result = corroborate(
        [
            _vote("pubmed", Stance.SUPPORTS, document_ids=("pmid:5",), title="Shared"),
            _vote("europepmc", Stance.SUPPORTS, document_ids=("pmid:5",), title="Shared"),
            _vote("page", Stance.SUPPORTS, family="page:example.com", document_ids=("pmid:5",)),
        ]
    )
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "PubMed, Europe PMC, and Linked page" in result.explanation


def test_document_ids_merge_transitively() -> None:
    result = corroborate(
        [
            _vote("pubmed", Stance.REFUTES, document_ids=("doi:10.1000/a", ""), title="A"),
            _vote(
                "page",
                Stance.REFUTES,
                family="page:journal.test",
                document_ids=("pmid:8",),
                title="A",
            ),
            _vote(
                "europepmc",
                Stance.REFUTES,
                document_ids=("doi:10.1000/a", "pmid:8"),
                title="A",
            ),
            _vote("wikipedia", Stance.REFUTES, title="Other"),
        ]
    )
    assert result.verdict is Verdict.UNSUPPORTED
    assert len(result.refute) == 2


def test_family_fallback_uses_the_source_name() -> None:
    result = corroborate(
        [
            SourceVote("alpha", "", Stance.SUPPORTS, 0.8, title="A"),
            SourceVote("beta", "", Stance.SUPPORTS, 0.8, title="B"),
        ]
    )
    assert result.verdict is Verdict.SUPPORTED
