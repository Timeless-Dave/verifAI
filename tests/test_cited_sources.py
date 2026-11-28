"""A claim read against the source the paste cites for it, and research search."""

from __future__ import annotations

import io

import httpx
import respx
from pypdf import PdfWriter
from tests.conftest import make_settings

from verifai.citelink import Link, link_sources, markers_in
from verifai.claims import screen_with_citations
from verifai.corroboration import Stance
from verifai.experts.router import domains_for
from verifai.http import PoliteHttp
from verifai.models.parsing import Judgment
from verifai.orchestrate import CheckBundle, _with_source_checks, rollup
from verifai.retrieval.scholarly import (
    ArxivSearchRetriever,
    OpenAlexSearchRetriever,
    SemanticScholarSearchRetriever,
    topic_terms,
)
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    CombinedClaim,
    EvidenceItem,
    ExpertResult,
    SourceCheck,
    Verdict,
)
from verifai.source_check import (
    SourceChecker,
    as_evidence,
    best_passages,
    best_sentence,
    claim_content,
    search_links,
)
from verifai.source_reader import SourceReader, SourceText, abstract_from_index, identifiers
from verifai.web.view import build_view, citation_card, claim_card

NORTHEY = (
    "Northey, J. M., Cherbuin, N., Pumpa, K. L., Smee, D. J., & Rattray, B. (2018). Exercise "
    "interventions for cognitive function in adults older than 50. British Journal of Sports "
    "Medicine, 52(3), 154-160. https://doi.org/10.1136/bjsports-2016-096587"
)


def _citation(
    raw: str, *, status: CitationStatus = CitationStatus.VERIFIED, **extra: object
) -> CitationResult:
    return CitationResult(
        raw=raw,
        kind=extra.pop("kind", "reference"),  # type: ignore[arg-type]
        status=status,
        confidence=0.9,
        explanation="x",
        **extra,  # type: ignore[arg-type]
    )


# Claims keep what they cite ------------------------------------------------


def test_cited_sentences_in_the_present_tense_are_claims() -> None:
    claims, _ignored, written = screen_with_citations(
        "Regular exercise improves cognitive function in adults over 50 (Northey et al., 2018).\n"
        "According to Vaswani et al. (2017), the Transformer relies on recurrent networks.\n"
        "The Transformer dispenses with recurrence entirely [1].\n"
        "Photosynthesis stores light energy in sugars. https://en.wikipedia.org/wiki/Photosynthesis"
    )
    assert claims == [
        "Regular exercise improves cognitive function in adults over 50.",
        "According to Vaswani et al., the Transformer relies on recurrent networks.",
        "The Transformer dispenses with recurrence entirely.",
        "Photosynthesis stores light energy in sugars.",
    ]
    assert written[claims[0]].endswith("(Northey et al., 2018).")
    assert written[claims[1]].startswith("According to Vaswani et al. (2017)")
    assert written[claims[2]].endswith("[1].")
    assert "wikipedia.org" in written[claims[3]]


def test_markers_of_every_common_style() -> None:
    found = markers_in(
        "A (Northey et al., 2018; Lee & Park, 2019) and Vaswani et al. (2017) said it [2, 4-5] "
        "doi:10.1038/nature14539 arXiv:1706.03762 https://example.org/paper"
    )
    surnames = {(item.surname, item.year) for item in found if item.surname}
    assert {("Northey", 2018), ("Lee", 2019), ("Vaswani", 2017)} <= surnames
    assert {item.number for item in found if item.number} == {2, 4, 5}
    identifiers_found = {item.identifier for item in found if item.identifier}
    assert {"10.1038/nature14539", "1706.03762", "https://example.org/paper"} <= identifiers_found
    # "(see Table 2, 2019)" is not an author.
    assert not [item for item in markers_in("as shown (see 2019)") if item.surname]


def test_markers_link_to_their_reference_entries() -> None:
    northey = _citation(NORTHEY)
    vaswani = _citation("arXiv:1706.03762", kind="doi")
    bert = _citation("Devlin, J. (2019). BERT. NAACL. arXiv:1810.04805")
    references = {"1": "[1] Devlin, J. (2019). BERT. NAACL. arXiv:1810.04805"}
    citations = [northey, vaswani, bert]
    assert [
        link.citation
        for link in link_sources(
            "X (Northey et al., 2018).", citations, references, single_claim=False
        )
    ] == [northey]
    assert [
        link.citation for link in link_sources("X [1].", citations, references, single_claim=False)
    ] == [bert]
    assert [
        link.citation
        for link in link_sources("X arXiv:1706.03762.", citations, references, single_claim=False)
    ] == [vaswani]
    # No marker: one source in the paste is the claim's source.
    only = link_sources("X.", [northey], {}, single_claim=False)
    assert only == [Link(northey, None)]
    # No marker and several sources: a single claim is read against each.
    assert len(link_sources("X.", citations, {}, single_claim=True)) == 3
    assert link_sources("X.", citations, {}, single_claim=False) == []
    # A wrong year is not a match.
    assert link_sources("X (Northey et al., 2011).", [northey, bert], {}, single_claim=False) == []


# Reading the cited work ----------------------------------------------------


def test_identifiers_come_from_the_paste_and_the_records() -> None:
    ids = identifiers(_citation(NORTHEY))
    assert ids.doi == "10.1136/bjsports-2016-096587"
    arxiv = identifiers(_citation("10.48550/arXiv.1706.03762", kind="doi"))
    assert arxiv.arxiv == "1706.03762"
    assert identifiers(_citation("29897866", kind="pmid")).pmid == "29897866"
    assert identifiers(_citation("example.org/x", kind="url")).url == "https://example.org/x"
    found = _citation(
        "Smith (2020). A paper.",
        evidence=[EvidenceItem(source="crossref", doi="10.1/abc")],
    )
    assert identifiers(found).doi == "10.1/abc"


def test_abstracts_are_put_back_in_order() -> None:
    assert abstract_from_index({"world": [1], "Hello": [0], "again": [2, 3]}) == (
        "Hello world again again"
    )
    assert abstract_from_index(None) == ""


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@respx.mock
def test_the_reader_tries_each_open_route() -> None:
    settings = make_settings()
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.core.ac.uk/v3/search/works/").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    body = (
        "<html><body>"
        + "<p>The Transformer relies solely on attention.</p>" * 120
        + "</body></html>"
    )
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            text="<feed><entry><title>Attention</title><summary>We propose it.</summary></entry></feed>",
        )
    )
    respx.get("https://arxiv.org/html/1706.03762").mock(return_value=httpx.Response(200, text=body))
    arxiv = reader.read(_citation("arXiv:1706.03762", kind="doi"))
    assert arxiv and arxiv.read == "full text" and arxiv.via == "arXiv"
    assert arxiv.abstract == "We propose it."

    respx.get("https://api.openalex.org/works/doi:10.1000/oa").mock(
        return_value=httpx.Response(
            200,
            json={
                "display_name": "A study",
                "abstract_inverted_index": {"Exercise": [0], "helps.": [1]},
                "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456"},
            },
        )
    )
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fullTextXML").mock(
        return_value=httpx.Response(
            200, text="<article><body>" + "<p>Exercise helps.</p>" * 200 + "</body></article>"
        )
    )
    pmc = reader.read(_citation("doi:10.1000/oa", kind="doi"))
    assert pmc and pmc.read == "full text" and pmc.via == "Europe PMC"
    assert pmc.abstract == "Exercise helps."

    respx.get("https://api.openalex.org/works/doi:10.1000/closed").mock(
        return_value=httpx.Response(
            200, json={"display_name": "Closed", "abstract_inverted_index": None}
        )
    )
    respx.get("https://api.crossref.org/works/10.1000/closed").mock(
        return_value=httpx.Response(
            200, json={"message": {"abstract": "<jats:p>Only this.</jats:p>"}}
        )
    )
    closed = reader.read(_citation("doi:10.1000/closed", kind="doi"))
    assert closed and closed.read == "abstract" and closed.abstract == "Only this."

    respx.get("https://api.openalex.org/works/doi:10.1000/pdf").mock(
        return_value=httpx.Response(
            200, json={"best_oa_location": {"pdf_url": "https://example.org/paper.pdf"}}
        )
    )
    respx.get("https://example.org/paper.pdf").mock(
        return_value=httpx.Response(
            200, content=_pdf_bytes(), headers={"content-type": "application/pdf"}
        )
    )
    respx.get("https://api.crossref.org/works/10.1000/pdf").mock(return_value=httpx.Response(404))
    # A blank PDF has no text, and with no abstract there is nothing to read.
    assert reader.read(_citation("doi:10.1000/pdf", kind="doi")) is None

    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [{"title": "Diet", "abstractText": "<b>Fewer</b> events."}]
                }
            },
        )
    )
    pmid = reader.read(_citation("29897866", kind="pmid"))
    assert pmid and pmid.abstract == "Fewer events." and "pubmed" in (pmid.url or "")

    respx.get("https://example.org/page").mock(
        return_value=httpx.Response(
            200, text="<html><body><p>" + "Sugar stores energy. " * 30 + "</p></body></html>"
        )
    )
    page = reader.read(_citation("https://example.org/page", kind="url"))
    assert page and page.read == "web page" and page.via == "example.org"

    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "Dropout: a simple way to prevent overfitting",
                        "abstract_inverted_index": {"Drop": [0], "units.": [1]},
                    }
                ]
            },
        )
    )
    by_title = reader.read(
        _citation(
            "Srivastava (2014). Dropout. JMLR.",
            matched_title="Dropout: A simple way to prevent overfitting",
        )
    )
    assert by_title and by_title.abstract == "Drop units."


# Judging the claim against it -----------------------------------------------


class _Judge:
    name = "test"
    concurrent = True

    def __init__(
        self, stance: Stance, confidence: float = 0.9, reason: str = "It says so."
    ) -> None:
        self.stance = stance
        self.confidence = confidence
        self.reason = reason

    def available(self) -> bool:
        return True

    def judge(self, claim: str, passage: str) -> Judgment:
        del claim
        if "unrelated" in passage:
            return Judgment(Stance.NEUTRAL, 0.0, "Not about it.")
        return Judgment(self.stance, self.confidence, self.reason)


class _Reader:
    def __init__(self, source: SourceText | None) -> None:
        self.source = source

    def read(self, citation: CitationResult) -> SourceText | None:
        del citation
        return self.source


_FULL = SourceText(
    title="Exercise interventions",
    url="https://doi.org/10.1136/bjsports-2016-096587",
    read="full text",
    abstract="Exercise improved cognitive function in the over 50s, regardless of status.",
    body="Methods.\nWe searched databases.\nResults.\nExercise improved cognitive function.",
    via="an open-access PDF",
)


def _check(
    stance: Stance,
    source: SourceText | None = _FULL,
    status: CitationStatus = CitationStatus.VERIFIED,
) -> SourceCheck:
    checker = SourceChecker(_Reader(source), _Judge(stance), workers=4)  # type: ignore[arg-type]
    link = Link(_citation(NORTHEY, status=status), "(Northey et al., 2018)")
    return checker.check("Exercise improves cognitive function in adults over 50.", [link])[0]


def test_the_cited_source_says_it_contradicts_it_or_is_silent() -> None:
    says = _check(Stance.SUPPORTS)
    assert says.status == "supports" and says.read == "full text"
    assert says.quote and "Exercise improved cognitive function" in says.quote
    assert [link.label for link in says.links] == ["Google Scholar", "ResearchGate"]
    assert _check(Stance.REFUTES).status == "contradicts"
    silent = _check(Stance.NEUTRAL)
    assert silent.status == "not_found" and "full text was read" in (silent.reason or "")
    abstract_only = _check(
        Stance.NEUTRAL,
        SourceText("T", None, "abstract", "unrelated words", "", "OpenAlex"),
    )
    assert "full text is not open" in (abstract_only.reason or "")
    assert _check(Stance.SUPPORTS, None).status == "unreadable"
    assert _check(Stance.SUPPORTS, status=CitationStatus.NOT_FOUND).status == "missing"
    assert _check(Stance.SUPPORTS, status=CitationStatus.UNREACHABLE).status == "unreadable"


def test_who_said_it_is_not_what_is_checked() -> None:
    assert claim_content("According to Vaswani et al., the Transformer relies on attention.") == (
        "The Transformer relies on attention."
    )
    assert claim_content("Northey et al. (2018) found that exercise helps the over 50s.") == (
        "Exercise helps the over 50s."
    )
    assert claim_content("The study showed that sleep matters for memory.") == (
        "Sleep matters for memory."
    )
    plain = "Exercise improves cognition in adults."
    assert claim_content(plain) == plain


def test_passages_and_quotes_follow_the_claim() -> None:
    source = SourceText(
        "T",
        None,
        "full text",
        "",
        "Contents\nReferences\n"
        + "Background on many unrelated topics. " * 40
        + "\nExercise improved memory and attention in older adults.\n"
        + "\nReferences\n[1] Exercise memory attention older adults cited paper.",
        "x",
    )
    picked = best_passages("Exercise improves memory in older adults.", source)
    assert any("Exercise improved memory" in passage for passage in picked)
    assert not any("cited paper" in passage for passage in picked)
    quote = best_sentence(
        "Transformers rely on recurrence.",
        "Earlier models were based on recurrent networks. The Transformer relies solely on "
        "attention, dispensing with recurrence.",
        reason="The evidence says the Transformer relies solely on attention, dispensing with recurrence.",
    )
    assert quote.startswith("The Transformer relies solely")
    assert search_links("") == []


def test_a_side_taken_counts_as_one_source() -> None:
    check = _check(Stance.SUPPORTS)
    item = as_evidence(check, _citation(NORTHEY))
    assert (
        item
        and item.stance == "supports"
        and item.document_ids == ["doi:10.1136/bjsports-2016-096587"]
    )
    assert as_evidence(_check(Stance.NEUTRAL), _citation(NORTHEY)) is None


def _claim(text: str, verdict: Verdict = Verdict.UNVERIFIABLE) -> CombinedClaim:
    return CombinedClaim(claim=text, verdict=verdict, confidence=0.0, explanation="thin")


def test_a_claim_about_what_a_source_found_is_that_sources_to_settle() -> None:
    claim = _claim("Northey et al. found that exercise helped only impaired adults.")
    link = Link(_citation(NORTHEY), "Northey et al. (2018)")
    settled = _with_source_checks(claim, [_check(Stance.REFUTES)], [link])
    assert settled.verdict is Verdict.UNSUPPORTED
    assert "so that source decides it" in settled.explanation
    # A plain claim still needs a second source.
    plain = _with_source_checks(
        _claim("Exercise improves cognition in adults over 50."), [_check(Stance.SUPPORTS)], [link]
    )
    assert plain.verdict is Verdict.UNVERIFIABLE
    assert "The cited source" in plain.explanation and "says this" in plain.explanation


def test_the_paste_verdict_weighs_misattribution_and_real_citations() -> None:
    cited = ExpertResult(
        expert="citation",
        verdict=Verdict.SUPPORTED,
        verdict_scope="citations",
        confidence=0.9,
        explanation="Checked 1 citation: 1 verified.",
        citations=[_citation(NORTHEY)],
    )
    unverified = _claim("Exercise helps.")
    verdict, _confidence, text = rollup([unverified], [cited])
    assert verdict is Verdict.UNVERIFIABLE and "cited works exist" in text
    refuted = _claim("Exercise harms.", Verdict.UNSUPPORTED)
    assert rollup([refuted], [cited])[0] is Verdict.UNSUPPORTED
    misread = _claim("Exercise helps.", Verdict.SUPPORTED).model_copy(
        update={"source_checks": [_check(Stance.NEUTRAL)]}
    )
    verdict, _confidence, text = rollup([misread], [cited])
    assert verdict is Verdict.MIXED and "not backed by the source cited" in text


def test_the_card_shows_what_the_cited_source_says() -> None:
    claim = _claim("Exercise helps.").model_copy(
        update={"source_checks": [_check(Stance.REFUTES), _check(Stance.NEUTRAL)]}
    )
    card = claim_card(claim)
    assert [source.badge.label for source in card.sources] == [
        "Contradicts this",
        "Doesn't say this",
    ]
    assert "Its cited source contradicts it" in card.flags
    assert card.sources[0].read == "Read: its full text, via an open-access PDF."
    silent = claim_card(claim.model_copy(update={"source_checks": [_check(Stance.NEUTRAL)]}))
    assert "Not in its cited source" in silent.flags
    labels = [link.label for link in citation_card(_citation(NORTHEY)).search_links]
    assert labels == ["Google Scholar", "ResearchGate"]
    view = build_view(CheckBundle([], [claim], Verdict.MIXED, 0.5, "x", sources="[1] A ref"))
    assert view.sources == "[1] A ref"


# Searching the literature ----------------------------------------------------


def test_research_claims_go_to_the_literature() -> None:
    for claim in (
        "Growth mindset interventions improve academic achievement.",
        "The Transformer relies on attention mechanisms.",
        "Spaced repetition improves long-term retention compared with massed practice.",
    ):
        assert "research" in domains_for(claim), claim
    # Health claims keep their own literature search.
    assert "research" not in domains_for("Vaccines cause autism in children.")
    assert topic_terms("Spaced repetition improves long-term retention since 2010.") == [
        "spaced",
        "repetition",
        "long-term",
        "retention",
    ]


@respx.mock
def test_literature_search_results_become_papers() -> None:
    settings = make_settings()
    http = PoliteHttp(settings)
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "doi": "https://doi.org/10.1/x",
                        "display_name": "Spaced practice",
                        "publication_year": 2019,
                        "abstract_inverted_index": {"Spacing": [0], "helps.": [1]},
                        "primary_location": {"source": {"display_name": "Memory"}},
                    },
                    {"display_name": "No abstract"},
                    "junk",
                ]
            },
        )
    )
    found = OpenAlexSearchRetriever(http, settings).retrieve(
        "Spaced repetition improves retention."
    )
    assert len(found.passages) == 1
    paper = found.passages[0]
    assert paper.family == "openalex" and paper.document_ids == ("doi:10.1/x",)
    assert paper.venue == "Memory" and "Spacing helps." in paper.text

    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            text=(
                "<feed><entry><id>http://arxiv.org/abs/1706.03762v5</id><title>Attention</title>"
                "<summary>Only attention.</summary><published>2017-06-12</published></entry>"
                "<entry><title>Broken</title></entry></feed>"
            ),
        )
    )
    arxiv = ArxivSearchRetriever(http).retrieve("Transformer attention recurrence architecture")
    assert [passage.url for passage in arxiv.passages] == ["https://arxiv.org/abs/1706.03762"]
    assert arxiv.passages[0].year == 2017

    respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "title": "Mindset",
                        "tldr": {"text": "It helps some."},
                        "externalIds": {"DOI": "10.2/y"},
                        "year": 2019,
                    },
                    {"title": "Nothing"},
                ]
            },
        )
    )
    s2 = SemanticScholarSearchRetriever(http, "key").retrieve("Growth mindset achievement")
    assert [passage.text for passage in s2.passages] == ["Mindset\nIt helps some."]
    assert OpenAlexSearchRetriever(http, settings).retrieve("a").passages == []


def test_bundle_sources_reach_later_batches(http_mock: respx.MockRouter) -> None:
    from fastapi.testclient import TestClient
    from tests.fakes import install

    from verifai.main import create_app

    install(http_mock)
    with TestClient(create_app(make_settings(max_claims=1))) as client:
        page = client.post(
            "/check",
            data={
                "text": "Exercise improves cognition (Northey et al., 2018).\n"
                "Sleep improves memory consolidation (Walker, 2017).\n\n" + NORTHEY
            },
            headers={"HX-Request": "true"},
        )
        assert 'name="sources"' in page.text
        assert "(Walker, 2017)" in page.text
