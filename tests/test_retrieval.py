from datetime import date

import httpx

from verifai.corroboration import Stance
from verifai.retrieval.current_events import CurrentEventsRetriever
from verifai.retrieval.europepmc import EuropePMCRetriever
from verifai.retrieval.gdelt import GdeltRetriever
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.pages import PageRetriever
from verifai.retrieval.pubmed import PubMedRetriever
from verifai.retrieval.textutil import (
    document_ids,
    extract_urls,
    html_to_text,
    search_terms,
    stance_from_rating,
    year_of,
)
from verifai.retrieval.wikidata import WikidataRetriever
from verifai.retrieval.wikipedia import WikipediaRetriever

_PUBMED_XML = b"""
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article>
        <ArticleTitle>Vitamin D and fractures</ArticleTitle>
        <Abstract><AbstractText>It does not prevent fractures.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
        <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue>
          <Title>The Lancet</Title></Journal>
        <PublicationTypeList><PublicationType>Systematic Review</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="doi">10.1000/vitd</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle><MedlineCitation><Article></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>
"""


def test_text_helpers():
    page = html_to_text(
        b"<html><style>skip</style><p>Hello <b>there</b></p><script>no</script></html>"
    )
    assert "Hello" in page and "there" in page and "skip" not in page and "no" not in page
    assert extract_urls("see https://doi.org/10.1/abc and https://example.com/a).") == [
        "https://example.com/a"
    ]
    assert "that" not in search_terms("that vitamin helps people")
    assert search_terms("") == []
    assert year_of(2020) == 2020
    assert year_of(12) is None
    assert year_of("published 1999") == 1999
    assert year_of(None) is None
    assert document_ids(doi="https://doi.org/10.1/X", pmid="5") == ("doi:10.1/x", "pmid:5")
    assert document_ids(doi="  ", pmid="abc") == ()
    assert stance_from_rating("  ") == (Stance.NEUTRAL, 0.0)
    assert stance_from_rating("Mostly false")[0] is Stance.REFUTES
    assert stance_from_rating("Pants on Fire")[0] is Stance.REFUTES
    assert stance_from_rating("Mostly true")[0] is Stance.SUPPORTS
    assert stance_from_rating("Half true")[0] is Stance.NEUTRAL
    assert stance_from_rating("Misleading")[1] == 0.65
    assert stance_from_rating("True")[0] is Stance.SUPPORTS
    assert stance_from_rating("unclear rating")[0] is Stance.NEUTRAL


def test_wikipedia_extract_and_snippet_fallback(polite_http, http_mock):
    http_mock.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "query": {"search": [{"title": "Eiffel Tower", "pageid": 1, "snippet": "iron"}]}
                },
            ),
            httpx.Response(
                200,
                json={
                    "query": {
                        "pages": {
                            "1": {
                                "extract": "The tower is in Paris.",
                                "fullurl": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                            }
                        }
                    }
                },
            ),
            httpx.Response(
                200, json={"query": {"search": [{"title": "Bare", "snippet": "only snippet"}]}}
            ),
            httpx.Response(503, text="down"),
            httpx.Response(200, json={"query": {}}),
            httpx.Response(200, text="nope"),
        ]
    )
    wiki = WikipediaRetriever(polite_http)
    found = wiki.retrieve("How tall is the Eiffel Tower in Paris?")
    assert found.passages[0].text.startswith("The tower")
    assert found.passages[0].family == "wikipedia"
    fallback = wiki.retrieve("Bare page about nothing much")
    assert "only snippet" in fallback.passages[0].text
    down = wiki.retrieve("anything here please")
    assert down.unavailable
    empty = wiki.retrieve("another query with words")
    assert empty.passages == []
    bad = wiki.retrieve("json failed for this query")
    assert bad.unavailable
    assert wiki.retrieve("   ").passages == []


def test_wikidata(polite_http, http_mock):
    http_mock.get("https://www.wikidata.org/w/api.php").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "search": [
                        {
                            "id": "Q243",
                            "label": "Eiffel Tower",
                            "description": "tower in Paris",
                            "concepturi": "http://www.wikidata.org/entity/Q243",
                        },
                        {"id": "Q1", "label": "", "description": ""},
                    ]
                },
            ),
            httpx.Response(200, json={"search": [{"id": "Q2", "label": "Item"}]}),
            httpx.Response(500, text="down"),
            httpx.Response(200, json={"search": "nope"}),
        ]
    )
    wiki = WikidataRetriever(polite_http)
    found = wiki.retrieve("Eiffel Tower")
    assert len(found.passages) == 1
    assert found.passages[0].family == "wikidata"
    bare = wiki.retrieve("Item only")
    bare_url = bare.passages[0].url
    assert bare_url is not None and bare_url.endswith("Q2")
    assert wiki.retrieve("down please").unavailable
    assert wiki.retrieve("bad shape").passages == []
    assert wiki.retrieve(" ").passages == []


def test_google_factcheck(polite_http, http_mock):
    http_mock.get(url__startswith="https://factchecktools.googleapis.com/").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "claims": [
                        {
                            "text": "The bridge opened in 1889.",
                            "claimant": "A speaker",
                            "claimReview": [
                                {
                                    "textualRating": "False",
                                    "title": "Not that year",
                                    "url": "https://example.com/check",
                                    "publisher": {"name": "Example Desk"},
                                },
                                "skip",
                            ],
                        }
                    ]
                },
            ),
            httpx.Response(200, json={}),
            httpx.Response(429, text="slow"),
        ]
    )
    checker = GoogleFactCheckRetriever(polite_http, "test-key")
    found = checker.retrieve("The bridge opened in 1889.")
    assert found.passages[0].native_stance is Stance.REFUTES
    assert found.passages[0].speaker == "Example Desk"
    assert found.passages[0].family == "google_factcheck"
    assert checker.retrieve("no claims here at all").passages == []
    assert checker.retrieve("rate limited query").unavailable
    assert checker.retrieve(" ").passages == []


def test_pages_skip_private_and_failed_fetches(polite_http, http_mock):
    http_mock.get("https://example.com/story").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><p>The city council voted to fund the bridge repair this year.</p></html>",
        )
    )
    http_mock.get("https://example.com/missing").mock(return_value=httpx.Response(404, text="no"))
    http_mock.get("https://example.com/file.pdf").mock(
        return_value=httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF"
        )
    )
    http_mock.get("https://example.com/tiny").mock(
        return_value=httpx.Response(200, text="<p>Hi</p>")
    )
    pages = PageRetriever(polite_http)
    found = pages.retrieve("Read https://example.com/story today.")
    assert found.passages[0].family == "page:example.com"
    assert "council" in found.passages[0].text
    blocked = pages.retrieve("http://127.0.0.1/secret")
    assert blocked.passages == [] and not blocked.unavailable
    missing = pages.retrieve("https://example.com/missing")
    assert missing.unavailable
    assert pages.retrieve("https://example.com/file.pdf").passages == []
    assert pages.retrieve("https://example.com/tiny").passages == []
    assert pages.retrieve("no links").passages == []


def test_pubmed_prefers_reviews_then_falls_back(polite_http, http_mock):
    calls: list[str] = []

    def search(request: httpx.Request) -> httpx.Response:
        term = request.url.params.get("term", "")
        calls.append(term)
        if "systematic review" in term:
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        return httpx.Response(200, json={"esearchresult": {"idlist": ["111"]}})

    http_mock.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        side_effect=search
    )
    http_mock.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        side_effect=[
            httpx.Response(200, content=_PUBMED_XML),
            httpx.Response(200, text="<not-xml"),
            httpx.Response(503, text="down"),
        ]
    )
    pubmed = PubMedRetriever(polite_http, email="tests@example.com")
    found = pubmed.retrieve("Does vitamin D prevent fractures in older adults?")
    assert calls[0].startswith("(") and "systematic review" in calls[0]
    assert len(calls) == 2
    assert found.passages[0].publication_type == "systematic review"
    assert found.passages[0].document_ids == ("doi:10.1000/vitd", "pmid:111")
    assert found.passages[0].year == 2020
    bad_xml = pubmed.retrieve("Does vitamin D prevent fractures in older adults?")
    assert bad_xml.unavailable
    http_mock.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": ["111"]}})
    )
    down = pubmed.retrieve("Does vitamin D prevent fractures in older adults?")
    assert down.unavailable
    assert pubmed.retrieve(" ").passages == []


def test_pubmed_keeps_review_hits_and_reports_search_errors(polite_http, http_mock):
    http_mock.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        side_effect=[
            httpx.Response(200, json={"esearchresult": {"idlist": ["111"]}}),
            httpx.Response(503, text="down"),
            httpx.Response(200, json={"esearchresult": {}}),
            httpx.Response(200, text="nope"),
        ]
    )
    http_mock.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, content=_PUBMED_XML)
    )
    pubmed = PubMedRetriever(polite_http, email="tests@example.com")
    found = pubmed.retrieve("vitamin D fractures older adults")
    assert found.passages and "systematic review" in (found.passages[0].publication_type or "")
    assert pubmed.retrieve("vitamin D fractures older adults").unavailable
    assert pubmed.retrieve("vitamin D fractures older adults").unavailable
    assert pubmed.retrieve("vitamin D fractures older adults").unavailable


def test_europepmc_reviews_and_errors(polite_http, http_mock):
    def search(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("query", "")
        if "PUB_TYPE" in query:
            return httpx.Response(200, json={"resultList": {"result": []}})
        return httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "pmid": "222",
                            "doi": "10.2000/abc",
                            "title": "A review of vitamin D",
                            "abstractText": "No fracture benefit.",
                            "pubType": "review",
                            "pubYear": "2019",
                            "authorString": "Ada Lovelace, Alan Turing",
                            "journalTitle": "BMJ",
                        },
                        {"title": ""},
                    ]
                }
            },
        )

    http_mock.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        side_effect=[
            search,
            search,
            httpx.Response(200, json={"resultList": {"result": {"nope": True}}}),
            httpx.Response(502, text="down"),
        ]
    )
    europe = EuropePMCRetriever(polite_http)
    found = europe.retrieve("vitamin D fractures in adults")
    assert found.passages[0].publication_type == "review"
    assert "doi:10.2000/abc" in found.passages[0].document_ids
    assert europe.retrieve("vitamin D fractures in adults").unavailable
    assert europe.retrieve("vitamin D fractures in adults").unavailable
    assert europe.retrieve("a").passages == []


def test_gdelt_outlets(polite_http, http_mock):
    http_mock.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "articles": [
                        {
                            "title": "Council funds the bridge",
                            "url": "https://news.example/story",
                            "domain": "news.example",
                            "seendate": "20261029",
                            "sourcecountry": "UK",
                        },
                        {"title": "", "url": "https://news.example/empty"},
                        "skip",
                    ]
                },
            ),
            httpx.Response(200, json={}),
            httpx.Response(200, json={"articles": {"bad": True}}),
            httpx.Response(500, text="down"),
        ]
    )
    gdelt = GdeltRetriever(polite_http)
    found = gdelt.retrieve("The council funded the bridge repair")
    assert found.passages[0].family == "outlet:news.example"
    assert found.passages[0].speaker == "news.example"
    assert gdelt.retrieve("The council funded the bridge repair").passages == []
    assert gdelt.retrieve("The council funded the bridge repair").unavailable
    assert gdelt.retrieve("The council funded the bridge repair").unavailable
    assert gdelt.retrieve("a").passages == []


def test_current_events_filters_stories(polite_http, http_mock):
    http_mock.get(url__startswith="https://en.wikipedia.org/api/rest_v1/feed/featured/").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "news": [
                        {
                            "story": "Voters went to the polls in the national election.",
                            "links": [
                                {
                                    "title": "National election",
                                    "content_urls": {
                                        "desktop": {
                                            "page": "https://en.wikipedia.org/wiki/National_election"
                                        }
                                    },
                                }
                            ],
                        },
                        {"story": "A sports team won a match.", "links": ["skip"]},
                        "skip",
                    ]
                },
            ),
            httpx.Response(200, json={"news": [{"story": "Election results are in."}]}),
            httpx.Response(200, json={"onthisday": []}),
            httpx.Response(404, text="missing"),
        ]
    )
    events = CurrentEventsRetriever(polite_http, today=lambda: date(2026, 10, 29))
    found = events.retrieve("What happened in the national election?")
    found_url = found.passages[0].url
    assert found_url is not None and found_url.endswith("National_election")
    assert found.passages[0].family == "wikipedia"
    bare = events.retrieve("election results today")
    bare_url = bare.passages[0].url
    assert bare_url is not None and bare_url.endswith("Portal:Current_events")
    assert events.retrieve("election results today").passages == []
    assert events.retrieve("election results today").unavailable
    assert events.retrieve("a").passages == []
