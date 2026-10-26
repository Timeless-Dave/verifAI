import httpx
import respx
from tests.conftest import make_settings

from verifai.experts.citation import build_citation_expert
from verifai.experts.citation.providers.semantic_scholar import SemanticScholarProvider
from verifai.http import PoliteHttp
from verifai.sample import EXAMPLE_TEXT

_PAPER = {
    "title": "A programmable dual RNA guided DNA endonuclease",
    "year": 2012,
    "authors": [{"name": "Martin Jinek"}, "not-an-author", {"name": " "}],
    "venue": "Science",
    "externalIds": {"DOI": "10.1126/science.1225829"},
    "url": "https://www.science.org/doi/10.1126/science.1225829",
    "paperId": "paper-1",
}


def test_lookup_search_and_failures() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r"https://api\.semanticscholar\.org/graph/v1/paper/DOI:.*").mock(
            side_effect=[
                httpx.Response(200, json=_PAPER),
                httpx.Response(404, json={"error": "missing"}),
                httpx.Response(503, text="slow"),
                httpx.Response(200, json={"title": "   "}),
                httpx.Response(200, json=["not", "an", "object"]),
            ]
        )
        router.get(url__startswith="https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=[
                httpx.Response(200, json={"data": [_PAPER, {"title": ""}]}),
                httpx.Response(200, json={"data": "nope"}),
                httpx.Response(429, json={"message": "rate limit"}),
            ]
        )
        http = PoliteHttp(make_settings())
        provider = SemanticScholarProvider(http, api_key="test-key")
        try:
            found = provider.lookup_doi("10.1126/science.1225829")
            missing = provider.lookup_doi("10.1234/missing")
            down = provider.lookup_doi("10.1234/down")
            blank = provider.lookup_doi("10.1234/blank")
            garbage = provider.lookup_doi("10.1234/garbage")
            searched = provider.search(
                title="A programmable dual RNA guided DNA endonuclease",
                authors=["Jinek"],
                year=2012,
                venue="Science",
                raw="unused when a title is present",
            )
            bad_list = provider.search(
                title=None,
                authors=[],
                year=None,
                venue=None,
                raw="Jinek 2012 Science",
            )
            limited = provider.search(
                title="again",
                authors=[],
                year=None,
                venue=None,
                raw="again",
            )
            api_key_header = router.calls[0].request.headers["x-api-key"]
        finally:
            http.close()

    assert found.unavailable is False
    assert found.records[0].doi == "10.1126/science.1225829"
    assert found.records[0].surnames == ["Jinek"]
    assert api_key_header == "test-key"
    assert missing.unavailable is False and missing.records == []
    assert down.unavailable is True
    assert blank.unavailable is False and blank.records == []
    assert garbage.unavailable is True
    assert searched.unavailable is False and searched.records[0].venue == "Science"
    assert bad_list.unavailable is True
    assert limited.unavailable is True


def test_expert_asks_semantic_scholar_when_a_key_is_set() -> None:
    with respx.mock(assert_all_called=False) as router:
        from tests.fakes import install

        install(router)
        router.route(url__startswith="https://api.semanticscholar.org/").mock(
            side_effect=lambda request: (
                httpx.Response(200, json={"data": [_PAPER]})
                if request.url.path.endswith("/search")
                else httpx.Response(200, json=_PAPER)
            )
        )
        settings = make_settings(semantic_scholar_api_key="test-key", max_concurrency=4)
        http = PoliteHttp(settings)
        expert = build_citation_expert(http, settings)
        try:
            result = expert.check(EXAMPLE_TEXT)
        finally:
            http.close()
    assert "semantic_scholar" in result.sources_consulted
    assert result.citations
