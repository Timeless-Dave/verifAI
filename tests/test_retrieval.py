from datetime import date

import httpx

from verifai.corroboration import Stance
from verifai.retrieval.current_events import CurrentEventsRetriever
from verifai.retrieval.europepmc import EuropePMCRetriever
from verifai.retrieval.federal_register import FederalRegisterRetriever
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.news_search import build_major_news, build_us_government
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
    def api(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if "titles" in params:
            return httpx.Response(200, json={"query": {"pages": {"-1": {"missing": ""}}}})
        if "pageids" in params:
            page = {
                "extract": "The tower is in Paris.",
                "fullurl": "https://en.wikipedia.org/wiki/Eiffel_Tower",
            }
            return httpx.Response(200, json={"query": {"pages": {"1": page}}})
        search = params["srsearch"].casefold()
        if "eiffel" in search:
            hit = {"title": "Eiffel Tower", "pageid": 1, "snippet": "iron"}
            return httpx.Response(200, json={"query": {"search": [hit]}})
        if "bare" in search:
            hit = {"title": "Bare", "snippet": "only snippet"}
            return httpx.Response(200, json={"query": {"search": [hit]}})
        if "anything" in search:
            return httpx.Response(503, text="down")
        if "another" in search:
            return httpx.Response(200, json={"query": {}})
        return httpx.Response(200, text="nope")

    http_mock.get("https://en.wikipedia.org/w/api.php").mock(side_effect=api)
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


def test_wikipedia_puts_the_article_the_claim_names_first(polite_http, http_mock):
    def api(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if "titles" in params:
            assert params["titles"] == "Napoleon Bonaparte|Corsica"
            pages = {
                "5": {"title": "Corsica", "extract": "Corsica is an island.", "fullurl": "u5"},
                "6": {"title": "Napoleon", "extract": "Napoleon was born in Ajaccio."},
                "7": {"title": "Mercury", "extract": "Mercury may refer to: a planet"},
            }
            redirects = [{"from": "Napoleon Bonaparte", "to": "Napoleon"}]
            return httpx.Response(200, json={"query": {"pages": pages, "redirects": redirects}})
        if "pageids" in params:
            page = {"title": "Letizia Bonaparte", "extract": "His mother."}
            return httpx.Response(200, json={"query": {"pages": {"9": page}}})
        hits = [
            {"title": "Napoleon", "pageid": 6},
            {"title": "Letizia Bonaparte", "pageid": 9},
        ]
        return httpx.Response(200, json={"query": {"search": hits}})

    http_mock.get("https://en.wikipedia.org/w/api.php").mock(side_effect=api)
    found = WikipediaRetriever(polite_http).retrieve("Napoleon Bonaparte was born in Corsica.")
    assert [passage.title for passage in found.passages] == [
        "Napoleon",
        "Corsica",
        "Letizia Bonaparte",
    ]
    assert found.passages[0].url == "https://en.wikipedia.org/wiki/Napoleon"


def test_wikidata(polite_http, http_mock):
    election = {
        "labels": {"en": {"value": "2016 United States presidential election"}},
        "descriptions": {"en": {"value": "58th quadrennial U.S. presidential election"}},
        "claims": {
            "P585": [_time_statement("+2016-11-08T00:00:00Z", 11)],
            "P991": [_item_statement("Q22686")],
            "P2044": [_quantity_statement("+12.5", "Q11573")],
            "P726": [_item_statement("Q6294", rank="deprecated")],
        },
    }

    def api(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params["action"] == "wbsearchentities":
            hits = {"2016 United States presidential election": "Q699872", "Joe Biden": None}
            found = hits.get(params["search"])
            return httpx.Response(200, json={"search": [{"id": found}] if found else []})
        if params["props"] == "labels":
            labels = {"Q22686": "Donald Trump", "Q11573": "metre"}
            return httpx.Response(
                200,
                json={
                    "entities": {
                        qid: {"labels": {"en": {"value": labels[qid]}}}
                        for qid in params["ids"].split("|")
                        if qid in labels
                    }
                },
            )
        return httpx.Response(200, json={"entities": {"Q699872": election}})

    http_mock.get("https://www.wikidata.org/w/api.php").mock(side_effect=api)
    wiki = WikidataRetriever(polite_http)
    found = wiki.retrieve("Joe Biden won the 2016 U.S. presidential election.")
    assert len(found.passages) == 1
    passage = found.passages[0]
    assert passage.family == "wikidata"
    assert passage.url == "https://www.wikidata.org/wiki/Q699872"
    assert passage.document_ids == ("wikidata:Q699872",)
    assert passage.text == (
        "2016 United States presidential election (58th quadrennial U.S. presidential election). "
        "point in time: 8 November 2016; successful candidate: Donald Trump; "
        "elevation above sea level: 12.5 metre."
    )
    assert wiki.retrieve(" ").passages == []
    assert wiki.retrieve("nothing capitalised here at all").passages == []


def test_wikidata_outage(polite_http, http_mock):
    http_mock.get("https://www.wikidata.org/w/api.php").mock(
        return_value=httpx.Response(500, text="down")
    )
    assert WikidataRetriever(polite_http).retrieve("Barack Obama was born in Hawaii.").unavailable


def test_wikidata_names_and_dates():
    from verifai.retrieval.wikidata import _time_text, entity_names

    assert entity_names("Joe Biden won the 2016 U.S. presidential election.") == [
        "2016 United States presidential election",
        "Joe Biden",
    ]
    assert entity_names("Mount Everest's official height is 8,849 m.") == ["Mount Everest"]
    assert entity_names("Taking megadoses of vitamin C prevents colds.") == []
    assert entity_names("Python was created by Guido van Rossum.") == ["Python", "Guido van Rossum"]
    assert _time_text({"time": "+1961-08-04T00:00:00Z", "precision": 11}) == "4 August 1961"
    assert _time_text({"time": "+1991-02-00T00:00:00Z", "precision": 10}) == "February 1991"
    assert _time_text({"time": "-0480-00-00T00:00:00Z", "precision": 9}) == "480 BCE"


def _item_statement(qid: str, *, rank: str = "normal") -> dict[str, object]:
    return {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"type": "wikibase-entityid", "value": {"id": qid}},
        },
    }


def _time_statement(time: str, precision: int) -> dict[str, object]:
    return {
        "rank": "normal",
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {"type": "time", "value": {"time": time, "precision": precision}},
        },
    }


def _quantity_statement(amount: str, unit: str) -> dict[str, object]:
    return {
        "rank": "normal",
        "mainsnak": {
            "snaktype": "value",
            "datavalue": {
                "type": "quantity",
                "value": {"amount": amount, "unit": f"http://www.wikidata.org/entity/{unit}"},
            },
        },
    }


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


_RSS = """<rss><channel>
<item><title>Democrats win full control of Congress - NPR</title>
<link>https://news.google.com/rss/articles/a</link><pubDate>Wed, 04 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://www.npr.org">NPR</source></item>
<item><title>Democrats win full control of Congress, again - NPR</title>
<link>https://news.google.com/rss/articles/b</link><pubDate>Wed, 04 Nov 2026 09:00:00 GMT</pubDate>
<source url="https://www.npr.org">NPR</source></item>
<item><title>Opinion | What the Democrats should do now - The New York Times</title>
<link>https://news.google.com/rss/articles/c</link><pubDate>Thu, 05 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://www.nytimes.com">The New York Times</source></item>
<item><title>Democrats sweep Congress in a blue wave - Some Blog</title>
<link>https://news.google.com/rss/articles/d</link><pubDate>Thu, 05 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://someblog.example">Some Blog</source></item>
<item><title>Republicans win control of the House, AP projects - AP News</title>
<link>https://news.google.com/rss/articles/e</link><pubDate>Wed, 13 Nov 2024 08:00:00 GMT</pubDate>
<source url="https://apnews.com">AP News</source></item>
</channel></rss>"""


def test_major_news_keeps_newsrooms_once_each(polite_http, http_mock):
    route = http_mock.get("https://news.google.com/rss/search").mock(
        side_effect=[
            httpx.Response(200, text=_RSS),
            httpx.Response(200, text="<rss><channel></channel></rss>"),
            httpx.Response(500, text="down"),
            httpx.Response(200, text="not xml <"),
        ]
    )
    news = build_major_news(polite_http)
    found = news.retrieve("Democrats win full control of Congress")
    speakers = [passage.speaker for passage in found.passages]
    # One vote per newsroom; a blog and an opinion column are left out.
    assert speakers == ["NPR", "AP News"]
    assert found.passages[0].family == "outlet:npr.org"
    assert "Headline from NPR, published 4 November 2026" in found.passages[0].text
    # The two-year-old headline says how old it is.
    assert "years ago: it shows what was true then" in found.passages[1].text
    # The last month first: the first request asks for it.
    assert route.calls[0].request.url.params["q"].endswith("when:30d")
    assert news.retrieve("Democrats win full control of Congress").unavailable
    assert news.retrieve("a").passages == []


def test_government_sites_count_each_agency(polite_http, http_mock):
    rss = """<rss><channel>
<item><title>FDA Approves First Drug - U.S. Food and Drug Administration (.gov)</title>
<link>https://news.google.com/rss/articles/x</link><pubDate>Mon, 09 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://www.fda.gov">U.S. Food and Drug Administration (.gov)</source></item>
<item><title>Statement on the new drug approval - Some Agency</title>
<link>https://news.google.com/rss/articles/y</link><pubDate>Mon, 09 Nov 2026 08:00:00 GMT</pubDate>
<source url="https://www.newagency.gov">Some Agency</source></item>
</channel></rss>"""
    route = http_mock.get("https://news.google.com/rss/search").mock(
        return_value=httpx.Response(200, text=rss)
    )
    found = build_us_government(polite_http).retrieve("The FDA approved the first drug")
    assert [passage.family for passage in found.passages] == ["gov:fda.gov", "gov:newagency.gov"]
    assert found.passages[0].speaker == "U.S. Food and Drug Administration"
    assert "site:whitehouse.gov" in str(route.calls[0].request.url).replace("%3A", ":")


def test_federal_register_documents(polite_http, http_mock):
    http_mock.get("https://www.federalregister.gov/api/v1/documents.json").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Menthol in Cigarettes",
                            "abstract": "FDA proposes a product standard.",
                            "html_url": "https://www.federalregister.gov/d/1",
                            "agencies": [{"name": "Food and Drug Administration"}],
                            "publication_date": "2022-05-04",
                            "type": "Proposed Rule",
                        },
                        {"title": "", "html_url": "x"},
                        "skip",
                    ]
                },
            ),
            httpx.Response(200, json={"results": "bad"}),
            httpx.Response(500, text="down"),
        ]
    )
    register = FederalRegisterRetriever(polite_http)
    found = register.retrieve("FDA bans menthol cigarettes")
    assert len(found.passages) == 1
    passage = found.passages[0]
    assert passage.family == "federal_register" and passage.year == 2022
    assert "Proposed Rule by Food and Drug Administration" in passage.text
    assert register.retrieve("FDA bans menthol cigarettes").passages == []
    assert register.retrieve("FDA bans menthol cigarettes").unavailable
    assert register.retrieve("a").passages == []


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


def test_fact_check_ratings_map_to_stances():
    from verifai.corroboration import Stance
    from verifai.retrieval.textutil import stance_from_rating

    expected = {
        "Not true": Stance.REFUTES,
        "No es cierto": Stance.REFUTES,
        "Faux": Stance.REFUTES,
        "Falso": Stance.REFUTES,
        "Pants on Fire!": Stance.REFUTES,
        "Verdadero": Stance.SUPPORTS,
        "Mostly True": Stance.SUPPORTS,
        "Half True": Stance.NEUTRAL,
        "Partly true": Stance.NEUTRAL,
        "Missing context": Stance.NEUTRAL,
        "Satire": Stance.NEUTRAL,
    }
    for rating, stance in expected.items():
        assert stance_from_rating(rating)[0] is stance, rating
    assert stance_from_rating("Partly false") == (Stance.REFUTES, 0.65)
    assert stance_from_rating("False") == (Stance.REFUTES, 0.9)
