"""CORE, and a paper the user attaches as a PDF."""

from __future__ import annotations

import base64
import zlib

import httpx
import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings
from tests.fakes import install

from verifai.citelink import Link
from verifai.http import PoliteHttp
from verifai.main import create_app
from verifai.orchestrate import _attached_citation, _with_paper
from verifai.retrieval.scholarly import CoreSearchRetriever
from verifai.schemas import CitationResult, CitationStatus
from verifai.source_check import ATTACHED
from verifai.source_reader import (
    SourceReader,
    SourceText,
    attached_paper,
    pack_paper,
    unpack_paper,
)

CORE = "https://api.core.ac.uk/v3/search/works/"


def _text_pdf(lines: list[str], title: str = "Sleep and memory consolidation") -> bytes:
    """A one-page PDF whose text pypdf can read back."""
    shown = " ".join(f"({line}) Tj T*" for line in lines)
    stream = f"BT /F1 9 Tf 30 770 Td 11 TL {shown} ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title ({title}) >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return out


_LINES = [
    "doi: 10.1000/attached",
    "Abstract",
    "Sleep after learning strengthened memory consolidation in all forty participants.",
    *["Participants slept eight hours and recalled more word pairs the next morning."] * 40,
    "Introduction",
]


def _citation(raw: str, **extra: object) -> CitationResult:
    return CitationResult(
        raw=raw,
        kind=extra.pop("kind", "reference"),  # type: ignore[arg-type]
        status=CitationStatus.VERIFIED,
        confidence=0.9,
        explanation="x",
        **extra,  # type: ignore[arg-type]
    )


# CORE ------------------------------------------------------------------------


@respx.mock
def test_core_gives_an_abstract_no_other_route_had() -> None:
    settings = make_settings(core_api_key=None)
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.openalex.org/works/doi:10.1000/closed").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.crossref.org/works/10.1000/closed").mock(
        return_value=httpx.Response(404)
    )
    core = respx.get(CORE).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"doi": "10.1000/other", "abstract": "Another paper."},
                    {
                        "id": 7,
                        "doi": "10.1000/closed",
                        "title": "Closed",
                        "abstract": "Sleep helps memory.",
                        "fullText": "Not available for public API users.",
                    },
                ]
            },
        )
    )
    found = reader.read(_citation("doi:10.1000/closed", kind="doi"))
    assert found is not None and found.read == "abstract" and found.via == "CORE"
    assert found.abstract == "Sleep helps memory."
    request = core.calls.last.request
    assert 'doi:"10.1000/closed"' in request.url.params["q"]
    assert "authorization" not in request.headers


@respx.mock
def test_without_a_key_core_is_spared_when_an_abstract_exists() -> None:
    settings = make_settings(core_api_key=None)
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.openalex.org/works/doi:10.1000/abs").mock(
        return_value=httpx.Response(
            200, json={"display_name": "Abs", "abstract_inverted_index": {"Short.": [0]}}
        )
    )
    core = respx.get(CORE).mock(return_value=httpx.Response(200, json={"results": []}))
    found = reader.read(_citation("doi:10.1000/abs", kind="doi"))
    assert found is not None and found.via == "OpenAlex"
    assert not core.called


@respx.mock
def test_a_core_key_turns_an_abstract_into_full_text() -> None:
    settings = make_settings(core_api_key="core-key")
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.openalex.org/works/doi:10.1000/abs").mock(
        return_value=httpx.Response(
            200, json={"display_name": "Abs", "abstract_inverted_index": {"Short.": [0]}}
        )
    )
    core = respx.get(CORE).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"doi": "10.1000/abs", "abstract": "", "fullText": "Results. " * 400}]
            },
        )
    )
    found = reader.read(_citation("doi:10.1000/abs", kind="doi"))
    assert found is not None and found.read == "full text" and found.via == "CORE"
    assert found.abstract == "Short." and found.title == "Abs"
    assert core.calls.last.request.headers["authorization"] == "Bearer core-key"


@respx.mock
def test_core_finds_a_work_by_its_title() -> None:
    settings = make_settings(core_api_key="k")
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    respx.get(CORE).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": 1, "title": "Something else entirely", "abstract": "No."},
                    {"id": 2, "title": "A Study of Sleep and Memory", "abstract": "Yes."},
                ]
            },
        )
    )
    found = reader.read(_citation("Walker (2017)", matched_title="A study of sleep and memory"))
    assert found is not None and found.abstract == "Yes."
    assert found.url == "https://core.ac.uk/works/2"


@respx.mock
def test_core_search_returns_papers() -> None:
    settings = make_settings()
    route = respx.get(CORE).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 9,
                        "title": "Sleep and memory",
                        "abstract": "Sleep after learning improves memory consolidation " * 8,
                        "doi": "10.1000/SLEEP",
                        "yearPublished": 2019,
                        "journals": [{"title": "Sleep"}],
                        "documentType": "research",
                    },
                    {"id": 10, "title": "Too short", "abstract": "Brief."},
                    {
                        "id": 11,
                        "title": "Figure 3",
                        "abstract": "Sleep and memory. " * 30,
                        "doi": "10.1371/journal.pone.0210036.g003",
                    },
                ]
            },
        )
    )
    retriever = CoreSearchRetriever(PoliteHttp(settings), "k")
    result = retriever.retrieve("Sleep improves memory consolidation")
    assert [passage.family for passage in result.passages] == ["core"]
    passage = result.passages[0]
    assert passage.url == "https://doi.org/10.1000/sleep" and passage.venue == "Sleep"
    assert route.calls.last.request.url.params["q"].startswith("abstract:(sleep AND")
    assert retriever.retrieve("sleep").passages == []
    route.mock(return_value=httpx.Response(429))
    assert retriever.retrieve("Sleep improves memory consolidation").unavailable


# The attached paper ---------------------------------------------------------


def test_an_attached_pdf_is_read_in_full() -> None:
    paper = attached_paper(_text_pdf(_LINES), "walker2017.pdf")
    assert paper is not None and paper.read == "full text"
    assert paper.title == "Sleep and memory consolidation"
    assert paper.url == "https://doi.org/10.1000/attached"
    assert paper.abstract.startswith("Sleep after learning strengthened")
    assert paper.via == "the PDF you attached (walker2017.pdf)"
    untitled = attached_paper(_text_pdf(_LINES, title="Microsoft Word - draft"), "draft.pdf")
    assert untitled is not None and untitled.title == "draft"
    assert attached_paper(b"not a pdf") is None
    assert attached_paper(_text_pdf(["Too short."])) is None


def test_the_attached_paper_rides_to_the_next_batch() -> None:
    paper = SourceText("T", "https://doi.org/10.1/x", "full text", "A.", "Body " * 100, "v")
    back = unpack_paper(pack_paper(paper))
    assert back == paper
    assert unpack_paper("") is None
    assert unpack_paper("%%%not base64") is None
    bomb = base64.urlsafe_b64encode(zlib.compress(b"0" * 5_000_000)).decode()
    assert unpack_paper(bomb) is None
    odd = base64.urlsafe_b64encode(
        zlib.compress(b'{"body": "B", "url": "https://evil.example"}')
    ).decode()
    unpacked = unpack_paper(odd)
    assert unpacked is not None and unpacked.url is None


def test_the_attached_paper_stands_in_for_the_citation_it_is() -> None:
    paper = SourceText(
        "Sleep and memory consolidation in adults", None, "full text", "", "Body", "v"
    )
    cited = _citation(
        "Walker, M. (2017). Sleep and memory consolidation in adults. Nature.",
        matched_title="Sleep and memory consolidation in adults",
    )
    other = _citation("Northey (2018). Exercise and cognition.")
    assert _attached_citation(paper, [other, cited]) is cited
    links = [Link(other, "(Northey, 2018)"), Link(cited, "(Walker, 2017)")]
    swapped = _with_paper(links, paper, cited)
    assert swapped[0].document is None and swapped[1].document is paper
    assert swapped[1].cited_as == "(Walker, 2017)"

    assert _attached_citation(paper, [other]) is None
    added = _with_paper([Link(other, "(Northey, 2018)")], paper, None)
    assert added[0].cited_as == ATTACHED and added[0].document is paper
    assert added[0].citation.matched_title == paper.title
    by_doi = SourceText("Short", "https://doi.org/10.1000/x", "full text", "", "B", "v")
    assert _attached_citation(by_doi, [_citation("doi:10.1000/x", kind="doi")]) is not None


def test_the_form_takes_an_optional_pdf(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    claims = (
        "Sleep after learning strengthens memory consolidation in adults.\n"
        "Participants who slept recalled more word pairs the next morning."
    )
    with TestClient(create_app(make_settings(max_claims=1, max_upload_bytes=50_000))) as client:
        page = client.post(
            "/check",
            data={"text": claims},
            files={"paper": ("walker.pdf", _text_pdf(_LINES), "application/pdf")},
            headers={"HX-Request": "true"},
        )
        assert page.status_code == 200
        assert "Read with the PDF you attached" in page.text
        assert ATTACHED in page.text
        token = page.text.split('name="paper" value="', 1)[1].split('"', 1)[0]
        assert unpack_paper(token) is not None

        later = client.post(
            "/check/more",
            data={"text": claims.splitlines()[1], "offset": 1, "total": 2, "paper": token},
            headers={"HX-Request": "true"},
        )
        assert later.status_code == 200 and ATTACHED in later.text

        plain = client.post("/check", data={"text": claims}, headers={"HX-Request": "true"})
        assert "Read with the PDF you attached" not in plain.text

        wrong = client.post(
            "/check",
            data={"text": claims},
            files={"paper": ("notes.pdf", b"hello", "application/pdf")},
            headers={"HX-Request": "true"},
        )
        assert "Only PDF files can be attached." in wrong.text
        big = client.post(
            "/check",
            data={"text": claims},
            files={"paper": ("big.pdf", b"%PDF-" + b"0" * 60_000, "application/pdf")},
            headers={"HX-Request": "true"},
        )
        assert "Attach a smaller copy" in big.text
        scan = client.post(
            "/check",
            data={"text": claims},
            files={"paper": ("scan.pdf", _text_pdf(["x"]), "application/pdf")},
            headers={"HX-Request": "true"},
        )
        assert "may be a scan" in scan.text
        home = client.get("/")
        assert 'name="paper"' in home.text and 'hx-encoding="multipart/form-data"' in home.text


@respx.mock
def test_an_attached_paper_is_named_by_the_id_it_prints() -> None:
    settings = make_settings()
    reader = SourceReader(PoliteHttp(settings), settings)
    respx.get("https://api.openalex.org/works/doi:10.48550/arxiv.1706.03762").mock(
        return_value=httpx.Response(200, json={"display_name": "Attention Is All You Need"})
    )
    paper = SourceText("attention", None, "full text", "", "arXiv:1706.03762v7 [cs.CL] text", "v")
    named = reader.named(paper)
    assert named.title == "Attention Is All You Need"
    assert named.url == "https://arxiv.org/abs/1706.03762"
    bare = SourceText("notes", None, "full text", "", "No identifier here.", "v")
    assert reader.named(bare) is bare
    respx.get("https://api.openalex.org/works/doi:10.1000/gone").mock(
        return_value=httpx.Response(404)
    )
    gone = SourceText("x", "https://doi.org/10.1000/gone", "full text", "", "Body", "v")
    assert reader.named(gone) is gone


def test_a_read_marker_is_no_dangling_pointer_and_headlines_read_well() -> None:
    from verifai.experts.citation.extract import unresolvable_mentions
    from verifai.orchestrate import CheckBundle
    from verifai.schemas import CombinedClaim, SourceCheck, Verdict
    from verifai.web.view import build_view, claim_headline

    text = "Residual nets won ILSVRC 2015 (He et al., 2016). Sleep helps memory (Walker, 2017)."
    assert len(unresolvable_mentions(text)) == 2
    claim = CombinedClaim(
        claim="Residual nets won ILSVRC 2015.",
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.0,
        explanation="",
        source_checks=[SourceCheck(citation="He", status="supports", cited_as="(He et al., 2016)")],
    )
    bundle = CheckBundle([], [claim], Verdict.UNVERIFIABLE, 0.0, "")
    pointers = build_view(bundle, judged=True, text=text).unchecked_pointers
    assert pointers == ["(Walker, 2017)"]

    said = claim_headline(Verdict.UNVERIFIABLE, ["The cited source"], [], [])
    assert said.startswith("Only the cited source backs this up.")
    mine = claim_headline(Verdict.UNVERIFIABLE, [], ["Your attached PDF"], [])
    assert mine.startswith("Only your attached PDF contradicts this.")
    named = claim_headline(Verdict.UNVERIFIABLE, ["Wikipedia"], [], [])
    assert named.startswith("Only Wikipedia backs")
