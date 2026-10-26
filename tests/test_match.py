from verifai.experts.citation.match import assess, compare_authors, compare_venue, title_scores
from verifai.experts.citation.match import Query
from verifai.experts.citation.providers.base import Record
from verifai.schemas import CitationStatus


def _record(**overrides) -> Record:
    data = dict(
        source="crossref",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        surnames=["Vaswani", "Shazeer"],
        year=2017,
        years=[2017],
        venue="Advances in Neural Information Processing Systems",
        doi="10.48550/arxiv.1706.03762",
        url="https://doi.org/10.48550/arxiv.1706.03762",
    )
    data.update(overrides)
    return Record(**data)


def test_exact_title_authors_and_year_verify():
    query = Query(
        title="Attention is all you need",
        surnames=["Vaswani", "Shazeer"],
        year=2017,
        venue="Advances in Neural Information Processing Systems",
        et_al=False,
    )
    result = assess(query, _record())
    assert result is not None
    assert result.status == CitationStatus.VERIFIED
    assert result.mismatches == []


def test_year_off_by_one_is_partial():
    query = Query(
        title="Attention is all you need",
        surnames=["Vaswani"],
        year=2016,
        venue=None,
        et_al=True,
    )
    result = assess(query, _record())
    assert result is not None
    assert result.status == CitationStatus.PARTIALLY_MATCHES
    assert any("off by one" in item for item in result.mismatches)


def test_wrong_authors_on_the_same_title_are_partial():
    query = Query(
        title="Attention is all you need",
        surnames=["Smith"],
        year=2021,
        venue="Nature",
        et_al=True,
    )
    result = assess(query, _record())
    assert result is not None
    assert result.status == CitationStatus.PARTIALLY_MATCHES
    assert any(item.startswith("Author mismatch") for item in result.mismatches)
    assert any(item.startswith("Year mismatch") for item in result.mismatches)


def test_unrelated_title_is_not_a_candidate():
    query = Query(
        title="Quantum entanglement in household sourdough starters",
        surnames=["Smith", "Doe"],
        year=2019,
        venue="Journal of Improbable Culinary Physics",
        et_al=False,
    )
    record = _record(
        title="Scaling is Like Making Sourdough: Finding Sourdough Starters to Help Your Research Scale",
        authors=["Karyn Georgilis"],
        surnames=["Georgilis"],
        year=2020,
        years=[2020],
        venue="Epic Journal",
    )
    assert assess(query, record) is None
    assert title_scores(query.title, record.title).blend < 0.8


def test_short_title_fragment_does_not_verify():
    query = Query(title="all you need", surnames=["Vaswani"], year=2017, venue=None, et_al=True)
    assert assess(query, _record()) is None


def test_accent_folded_surnames_match():
    query = Query(title="A programmable dual RNA guided DNA endonuclease", surnames=["Jinek"], year=2012, venue="Science", et_al=True)
    record = _record(
        title="A programmable dual-RNA-guided DNA endonuclease in adaptive bacterial immunity",
        authors=["Martin Jínek"],
        surnames=["Jínek"],
        year=2012,
        years=[2012],
        venue="Science",
    )
    assert compare_authors(query, record) == "match"


def test_venue_abbreviation_and_contradiction():
    assert compare_venue("PNAS", "Proceedings of the National Academy of Sciences") == "match"
    assert compare_venue("Science", "Nature") == "mismatch"
    assert compare_venue("Science", "Science") == "match"
