import httpx
import respx
from tests.conftest import make_settings

from verifai.experts.citation import CitationExpert
from verifai.experts.citation.decide import decide_pmid
from verifai.experts.citation.extract import extract_citations
from verifai.experts.citation.providers.pubmed_id import PubMedIdProvider
from verifai.experts.citation.providers.url_check import parse_page_meta
from verifai.http import PoliteHttp
from verifai.retrieval.textutil import detect_language, search_terms
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import CitationStatus

NATURE_HEAD = """<html><head><title>Mastering the game of Go | Nature</title>
<meta name="citation_reference" content="citation_title=Deep learning; citation_doi=10.1038/nature14539"/>
<meta name="citation_title" content="Mastering the game of Go without human knowledge"/>
<meta name="dc.identifier" content="doi:10.1038/nature24270"/>
</head></html>"""


def test_page_meta_reads_the_page_doi_not_a_reference_doi() -> None:
    meta = parse_page_meta(NATURE_HEAD)
    assert meta.doi == "10.1038/nature24270"
    assert meta.title == "Mastering the game of Go without human knowledge"
    plain = parse_page_meta("<html><head><title> A  plain page </title></head></html>")
    assert plain.doi is None
    assert plain.title == "A plain page"


def test_a_redirect_to_a_private_address_is_refused() -> None:
    http = PoliteHttp(make_settings())
    try:
        with respx.mock(assert_all_called=False) as router:
            router.get("https://example.org/go").mock(
                return_value=httpx.Response(
                    302, headers={"location": "http://127.0.0.1:11434/api/tags"}
                )
            )
            local = router.get("http://127.0.0.1:11434/api/tags").mock(
                return_value=httpx.Response(200, json={"models": []})
            )
            result = http.get("https://example.org/go")
        assert result.error is not None
        assert result.error.startswith("BlockedRedirect")
        assert local.called is False
    finally:
        http.close()


def test_a_live_link_is_checked_as_the_doi_it_declares(
    http_mock: respx.MockRouter, expert: CitationExpert
) -> None:
    from tests.fakes import install

    install(http_mock)
    http_mock.route(method="HEAD", url="https://journal.example/paper").mock(
        return_value=httpx.Response(200)
    )
    http_mock.get("https://journal.example/paper").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<meta name="citation_doi" content="10.1234/verifai.fabricated.2021">',
        )
    )
    result = expert.check("See https://journal.example/paper for details.")
    row = result.citations[0]
    assert row.kind == "url"
    assert "identifies itself as DOI 10.1234/verifai.fabricated.2021" in row.explanation
    assert row.status is CitationStatus.NOT_FOUND


def test_pmid_is_skipped_when_its_line_has_a_doi() -> None:
    with_doi = extract_citations("Lancet. 1998. doi:10.1016/S0140-6736(97)11096-0. PMID: 9500320.")
    assert [item.kind for item in with_doi] == ["doi"]
    alone = extract_citations("Wakefield AJ et al. Lancet 1998. PMID: 9500320.")
    assert [(item.kind, item.pmid) for item in alone] == [("pmid", "9500320")]


def test_pubmed_marks_a_retracted_publication() -> None:
    http = PoliteHttp(make_settings())
    summary = {
        "result": {
            "uids": ["9500320"],
            "9500320": {
                "title": "Ileal-lymphoid-nodular hyperplasia.",
                "pubdate": "1998 Feb 28",
                "fulljournalname": "Lancet (London, England)",
                "authors": [{"name": "Wakefield AJ"}],
                "pubtype": ["Journal Article", "Retracted Publication"],
                "articleids": [{"idtype": "doi", "value": "10.1016/s0140-6736(97)11096-0"}],
            },
        }
    }
    try:
        with respx.mock() as router:
            router.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/").mock(
                return_value=httpx.Response(200, json=summary)
            )
            outcome = PubMedIdProvider(http, email="t@example.com").lookup_pmid("9500320")
        citation = extract_citations("Wakefield AJ et al. Lancet 1998. PMID: 9500320.")[0]
        result = decide_pmid(citation, outcome)
        assert result.status is CitationStatus.RETRACTED
        assert "Retracted Publication" in result.explanation
        with respx.mock() as router:
            router.get(url__startswith="https://eutils.ncbi.nlm.nih.gov/").mock(
                return_value=httpx.Response(
                    200,
                    json={"result": {"uids": ["1"], "1": {"error": "cannot get document summary"}}},
                )
            )
            missing = PubMedIdProvider(http, email="t@example.com").lookup_pmid("1")
        assert decide_pmid(citation, missing).status is CitationStatus.NOT_FOUND
    finally:
        http.close()


def test_claims_are_searched_in_their_own_language() -> None:
    assert detect_language("La tour Eiffel se trouve à Berlin.") == "fr"
    assert detect_language("El agua hierve a 100 °C al nivel del mar.") == "es"
    assert detect_language("The Eiffel Tower is in Paris.") == "en"
    assert detect_language("Paris") == "en"
    assert "gödel" in search_terms("Kurt Gödel published his incompleteness theorems in 1931.")
    http = PoliteHttp(make_settings())
    try:
        with respx.mock() as router:
            french = router.get(url__startswith="https://fr.wikipedia.org/w/api.php").mock(
                return_value=httpx.Response(200, json={"query": {"search": []}})
            )
            WikipediaRetriever(http).retrieve("La tour Eiffel se trouve à Berlin.")
        assert french.called
        assert "eiffel" in str(french.calls[0].request.url).casefold()
    finally:
        http.close()


def test_non_latin_scripts_pick_their_wikipedia() -> None:
    from verifai.retrieval.textutil import search_terms_for

    assert detect_language("长城位于中国。") == "zh"
    assert detect_language("富士山は日本で一番高い山です。") == "ja"
    assert detect_language("القاهرة هي عاصمة مصر.") == "ar"
    assert search_terms_for("长城位于中国。", "zh") == ["长城位于中国"]
