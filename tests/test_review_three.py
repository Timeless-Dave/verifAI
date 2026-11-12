"""Regressions from the third outside review of the live site."""

import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings
from tests.fakes import install

from verifai.claims import screen_paste
from verifai.corroboration import Stance
from verifai.experts.citation.decide import decide_isbn
from verifai.experts.citation.extract import ExtractedCitation, extract_citations
from verifai.experts.citation.match import Query, compare_authors
from verifai.experts.citation.providers.base import ProviderOutcome, Record
from verifai.experts.claim_check import _NOT_ABOUT
from verifai.main import create_app
from verifai.orchestrate import CheckBundle, _rollup_many
from verifai.schemas import CitationResult, CitationStatus, CombinedClaim, EvidenceItem, Verdict
from verifai.web.view import build_view, citation_card, claim_card


def test_a_reason_that_says_not_about_it_is_not_a_contradiction() -> None:
    for reason in (
        "The evidence does not support the claim, as it only describes the Celsius unit.",
        "It provides no information about the boiling point of water.",
        "This page is about a different campus of the university system.",
        "The entry says nothing about where the tower stands.",
    ):
        assert _NOT_ABOUT.search(reason), reason
    for reason in (
        "The evidence states the Eiffel Tower is in Paris, a different city from Rome.",
        "The review found no evidence that vaccines cause autism.",
        "Water boils at 100 °C at sea level, not 50 °C.",
    ):
        assert not _NOT_ABOUT.search(reason), reason


def _item(stance: str, url: str, source: str = "wikipedia") -> EvidenceItem:
    return EvidenceItem(source=source, title=url, url=url, stance=stance, family=source)


def _claim(
    verdict: Verdict, evidence: list[EvidenceItem], confidence: float = 0.0
) -> CombinedClaim:
    return CombinedClaim(
        claim="Drinking bleach cures COVID-19.",
        verdict=verdict,
        confidence=confidence,
        explanation="e",
        evidence=evidence,
        experts=["health"],
    )


def test_one_article_is_listed_once_and_a_false_lean_is_flagged() -> None:
    same = "https://www.factcheck.org/mms"
    evidence = [
        _item("neutral", same, "google_factcheck"),
        _item("refutes", same, "google_factcheck"),
        _item("refutes", same, "google_factcheck"),
        _item("neutral", "https://en.wikipedia.org/wiki/Bleach"),
    ]
    card = claim_card(_claim(Verdict.UNVERIFIABLE, evidence))
    assert card.found == 2
    assert card.refute == 1
    assert "Every source that took a side contradicts it" in card.flags


def test_a_paste_with_unverified_claims_is_partly_supported() -> None:
    supported = _claim(Verdict.SUPPORTED, [], 0.8)
    unverified = _claim(Verdict.UNVERIFIABLE, [])
    bundle = CheckBundle([], [supported, unverified, unverified], Verdict.SUPPORTED, 0.8, "x")
    assert build_view(bundle).badge.label == "Partly supported"
    every = CheckBundle([], [supported, supported], Verdict.SUPPORTED, 0.8, "x")
    assert build_view(every).badge.label == "Supported"


def test_the_paste_confidence_is_the_typical_claim_not_the_weakest() -> None:
    claims = [_claim(Verdict.SUPPORTED, [], value) for value in (0.88, 0.88, 0.4)]
    verdict, confidence, _text = _rollup_many(claims)
    assert verdict is Verdict.SUPPORTED
    assert confidence == 0.72


def test_arxiv_identifiers_are_labelled_arxiv() -> None:
    item = CitationResult(
        raw="10.48550/arXiv.1706.03762",
        kind="doi",
        status=CitationStatus.VERIFIED,
        confidence=0.9,
        explanation="x",
    )
    assert citation_card(item).kind == "arXiv"


def test_taste_is_set_aside_but_superlative_facts_are_kept() -> None:
    claims, _ignored = screen_paste(
        "Pizza is objectively the best food.\n"
        "Thriller is the best-selling album of all time.\n"
        "Mount Everest is the highest mountain on Earth.\n"
        "The cheetah is the fastest land animal."
    )
    assert claims == [
        "Thriller is the best-selling album of all time.",
        "Mount Everest is the highest mountain on Earth.",
        "The cheetah is the fastest land animal.",
    ]


def test_an_opinion_alone_says_so(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    with TestClient(create_app(make_settings())) as client:
        page = client.post(
            "/check",
            data={"text": "Pizza is objectively the best food."},
            headers={"HX-Request": "true"},
        )
        assert "That reads as opinion, not fact" in page.text
        emoji = client.post("/check", data={"text": "🍕🍕🍕"}, headers={"HX-Request": "true"})
        assert "No checker is registered" not in emoji.text


def test_a_negative_offset_is_refused(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    with TestClient(create_app(make_settings())) as client:
        refused = client.post("/check/more", data={"text": "x", "offset": "-5", "total": "2"})
        assert refused.status_code == 422


def test_later_batches_update_the_counts(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    with TestClient(create_app(make_settings(max_claims=1))) as client:
        page = client.post(
            "/check",
            data={"text": "The Eiffel Tower is in Paris.\nThe Moon is made of cheese."},
            headers={"HX-Request": "true"},
        )
        assert 'id="claims-progress"' in page.text and "Checking 1 more claim" in page.text
        assert "1 checked of 2" in page.text
        more = client.post(
            "/check/more",
            data={"text": "The Moon is made of cheese.", "offset": "1", "total": "2"},
            headers={"HX-Request": "true"},
        )
        assert 'id="claims-count" class="count" hx-swap-oob="true">2 checked<' in more.text
        assert 'id="claims-progress" class="batch-note" role="status" hx-swap-oob="true"></p>' in (
            more.text
        )


def _book(title: str, authors: list[str]) -> ProviderOutcome:
    record = Record(
        source="openlibrary",
        title=title,
        authors=authors,
        surnames=[name.split()[-1] for name in authors],
        year=2022,
        years=[2022],
        venue="MIT Press",
        doi=None,
        url="https://openlibrary.org/books/OL1M",
    )
    return ProviderOutcome("openlibrary", unavailable=False, records=[record])


def _isbn(context: str | None) -> ExtractedCitation:
    return ExtractedCitation(
        kind="isbn",
        raw="ISBN 9780262033848",
        start=0,
        end=18,
        isbn="9780262033848",
        context=context,
    )


def test_an_isbn_record_must_be_a_real_book_and_the_book_named() -> None:
    spam = decide_isbn(_isbn("Introduction to Algorithms"), _book("Fomo App Referral Code", []))
    assert spam.status is CitationStatus.UNREACHABLE
    other = decide_isbn(
        _isbn("Cormen. Introduction to Algorithms. MIT Press"),
        _book("The Art of Computer Programming", ["Donald Knuth"]),
    )
    assert other.status is CitationStatus.PARTIALLY_MATCHES
    right = decide_isbn(
        _isbn("Cormen. Introduction to Algorithms. MIT Press"),
        _book("Introduction to Algorithms, Fourth Edition", ["Thomas H. Cormen"]),
    )
    assert right.status is CitationStatus.VERIFIED
    bare = decide_isbn(_isbn(None), _book("Introduction to Algorithms", ["Thomas H. Cormen"]))
    assert bare.status is CitationStatus.VERIFIED


def test_isbn_context_is_the_words_on_its_line() -> None:
    found = extract_citations(
        "Intro line.\nCormen et al. Introduction to Algorithms. MIT Press. ISBN 978-0-262-04630-5\n"
    )
    isbn = next(item for item in found if item.kind == "isbn")
    assert isbn.context and "Introduction to Algorithms" in isbn.context


def test_vancouver_initials_are_not_a_surname() -> None:
    found = extract_citations(
        "Wakefield AJ et al. (1998). Ileal-lymphoid-nodular hyperplasia. The Lancet. "
        "doi:10.1016/S0140-6736(97)11096-0"
    )
    assert found[0].surnames == ["Wakefield"]


def test_a_record_with_fewer_authors_is_short_not_wrong() -> None:
    query = Query(
        title=None,
        surnames=["Krizhevsky", "Sutskever", "Hinton"],
        year=2012,
        venue=None,
        et_al=False,
    )
    record = Record(
        source="openalex",
        title="ImageNet classification",
        authors=["Alex Krizhevsky"],
        surnames=["Krizhevsky"],
        year=2012,
        years=[2012],
        venue=None,
        doi=None,
        url=None,
    )
    assert compare_authors(query, record) == "match"
    record.surnames = ["Smith"]
    assert compare_authors(query, record) == "mismatch"


def test_refutes_stance_keeps_its_meaning() -> None:
    assert Stance.REFUTES.value == "refutes"
