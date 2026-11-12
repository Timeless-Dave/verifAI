"""Regressions from the second stress paste, tests/data/crazy.txt."""

from pathlib import Path

import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings
from tests.fakes import install

from verifai.claims import screen_paste
from verifai.experts.citation.extract import (
    dangling_markers,
    extract_citations,
    prepare_text,
    unresolvable_mentions,
)
from verifai.main import create_app
from verifai.orchestrate import _numbered_references, _with_citation_notes
from verifai.schemas import CitationResult, CitationStatus, CombinedClaim, Verdict
from verifai.web.view import claim_card

CRAZY = (Path(__file__).parent / "data" / "crazy.txt").read_text()


def test_identifiers_survive_copy_and_paste_damage() -> None:
    assert prepare_text("doi:10.1109/CVPR.\n2016.90") == "doi:10.1109/CVPR.2016.90"
    assert prepare_text("10.1038/nature\u200b14539") == "10.1038/nature14539"
    assert (
        prepare_text("https://doi.org/10.1038%2Fnature14539")
        == "https://doi.org/10.1038/nature14539"
    )


def test_crazy_paste_citations() -> None:
    found = extract_citations(CRAZY)
    by_doi = {item.doi: item for item in found if item.doi}
    assert by_doi["10.1038/nature14539"].occurrences == 6
    assert "10.1109/CVPR.2016.90" in by_doi
    assert "10.48550/arXiv.hep-th/9711200" in by_doi
    assert by_doi["10.48550/arXiv.2513.00001"].problem
    lookalike = next(item for item in found if item.doi and "а" in item.doi)
    assert lookalike.problem and "U+0430" in lookalike.problem
    rumelhart = [item for item in found if item.doi == "10.1038/323533a0"]
    assert sorted(item.year or 0 for item in rumelhart) == [1986, 1990]
    assert by_doi["10.48550/arXiv.1810.04805"].title == "Attention Is All You Need"
    pmids = [item for item in found if item.kind == "pmid"]
    assert [item.pmid for item in pmids if not item.problem] == ["16060722"]
    assert sum(1 for item in pmids if item.problem) == 2
    assert any(item.kind == "pmc" and item.pmc == "PMC1182327" for item in found)
    assert any(item.url and item.url.endswith("Python_(programming_language)") for item in found)
    assert dangling_markers(CRAZY) == ["[31]"]
    assert unresolvable_mentions(CRAZY) == ["(Nguyen et al., 2021)", "ibid."]


def test_crazy_paste_claims_and_injections() -> None:
    claims, ignored = screen_paste(CRAZY)
    for kept in (
        "All mammals lay eggs.",
        "Some mammals lay eggs.",
        "The Berlin Wall fell on 9/11/1989.",
        "The Earth is round.",
        "The Moon is made of cheese.",
        "The Eiffel Tower is in Rome.",
        '"Let them eat cake."',
        "长城位于中国。",
        "富士山は日本で一番高い山です。",
    ):
        assert any(kept in claim for claim in claims), kept
    for dropped in (
        "Quick literature dump",
        "If water boiled",
        "Oh sure",
        "Some people claim",
        "VERIFIED",
        "Ignore the",
        "<script",
        "aWdub3Jl",
    ):
        assert not any(dropped in claim for claim in claims), dropped
    assert any("Ignore the two-source rule" in item for item in ignored)
    assert any(item.startswith("<script>") for item in ignored)
    assert any(item.startswith("{") for item in ignored)
    assert not any(item.startswith("<https://") for item in ignored)


def test_a_claim_inherits_the_retraction_of_the_reference_it_cites() -> None:
    claim = CombinedClaim(
        claim="Hydroxychloroquine increased mortality, as shown by Mehra et al. [15].",
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.0,
        explanation="Unverified.",
        experts=["health"],
    )
    entry = "[15] Mehra, M. R. et al. (2020). Hydroxychloroquine. Lancet. doi:10.1016/S0140-6736(20)31180-6"
    retracted = CitationResult(
        raw=entry, kind="doi", status=CitationStatus.RETRACTED, confidence=0.97, explanation="x"
    )
    noted = _with_citation_notes(claim, [retracted], _numbered_references(entry))
    assert "is retracted" in noted.explanation
    card = claim_card(noted)
    assert "Cites a retracted paper" in card.flags


def test_unjudged_claims_say_so_and_ambiguous_dates_are_flagged() -> None:
    claim = CombinedClaim(
        claim="The Berlin Wall fell on 9/11/1989.",
        verdict=Verdict.UNVERIFIABLE,
        confidence=0.0,
        explanation="e",
        experts=["general"],
    )
    card = claim_card(claim, judged=False)
    assert card.badge.label == "Not judged"
    assert "no AI model" in card.headline
    assert any("Ambiguous date 9/11/1989" in flag for flag in card.flags)


def test_more_claims_are_checked_in_place(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    app = create_app(make_settings(max_claims=1))
    with TestClient(app) as client:
        page = client.post(
            "/check",
            data={"text": "The Eiffel Tower is in Paris.\nThe Moon is made of cheese."},
            headers={"HX-Request": "true"},
        )
        assert 'hx-post="/check/more"' in page.text
        assert "Checking claims 2–2 of 2" in page.text
        more = client.post(
            "/check/more",
            data={"text": "The Moon is made of cheese.", "offset": "1", "total": "2"},
            headers={"HX-Request": "true"},
        )
        assert more.status_code == 200
        assert "№ 02" in more.text
        assert 'id="claims-done" hx-swap-oob="true">2<' in more.text
        assert 'hx-post="/check/more"' not in more.text
