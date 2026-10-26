import httpx

from tests.fakes import (
    ATTENTION_TITLE,
    JINEK_DOI,
    WIKI_URL,
    install,
)
from verifai.experts.citation.decide import (
    CONF_DOI_NOT_FOUND,
    CONF_DOI_VERIFIED_MULTI,
    CONF_DOI_VERIFIED_SINGLE_DOWN,
    CONF_NOT_FOUND,
    CONF_NOT_FOUND_PARTIAL_OUTAGE,
    CONF_REFERENCE_PARTIAL_MULTI,
    CONF_REFERENCE_VERIFIED_MULTI,
    CONF_REFERENCE_VERIFIED_SINGLE_DOWN,
    CONF_URL_ARCHIVED,
    CONF_URL_LIVE,
)
from verifai.sample import EXAMPLE_TEXT
from verifai.schemas import CitationStatus, EVIDENCE_PRINCIPLE, Verdict


def _by_kind_marker(result):
    return {(item.kind, (item.doi or item.raw)[:48]): item for item in result.citations}


def test_mixed_example_against_mocked_indexes(expert, http_mock):
    install(http_mock)
    result = expert.check(EXAMPLE_TEXT)

    assert result.expert == "citation"
    assert result.verdict == Verdict.MIXED
    assert result.verdict_scope == "citations"
    assert result.evidence_principle == EVIDENCE_PRINCIPLE
    assert "crossref" in result.sources_consulted
    assert "openalex" in result.sources_consulted
    assert "doi.org" in result.sources_consulted
    assert not any("semanticscholar" in str(call.request.url) for call in http_mock.calls)

    statuses = [item.status for item in result.citations]
    assert statuses.count(CitationStatus.VERIFIED) == 3
    assert statuses.count(CitationStatus.NOT_FOUND) == 3

    attention = next(item for item in result.citations if item.matched_title == ATTENTION_TITLE or (item.title_like if False else "Attention" in item.raw))
    assert attention.status == CitationStatus.VERIFIED
    assert attention.matched_year == 2017
    assert attention.confidence == CONF_REFERENCE_VERIFIED_MULTI
    assert "crossref" in attention.sources_contributed
    assert "openalex" in attention.sources_contributed
    assert "2025" in attention.explanation

    jinek = next(item for item in result.citations if "endonuclease" in item.raw.lower() or "Jinek" in item.raw)
    assert jinek.status == CitationStatus.VERIFIED
    assert jinek.matched_year == 2012
    assert "Science" in (jinek.matched_venue or "")

    plos = next(item for item in result.citations if item.kind == "doi" and "journal.pone" in (item.raw))
    assert plos.status == CitationStatus.VERIFIED
    assert plos.confidence == CONF_DOI_VERIFIED_MULTI
    assert "doi.org" in plos.sources_contributed
    assert "openalex" in plos.sources_contributed

    fake = next(item for item in result.citations if "fabricated" in item.raw)
    assert fake.status == CitationStatus.NOT_FOUND
    assert fake.confidence == CONF_DOI_NOT_FOUND
    assert "not a registered DOI" in fake.explanation
    assert fake.evidence == []

    sourdough = next(item for item in result.citations if "sourdough" in item.raw.lower())
    assert sourdough.status == CitationStatus.NOT_FOUND
    assert sourdough.confidence == CONF_NOT_FOUND
    assert "crossref" in sourdough.sources_contributed
    assert "openalex" in sourdough.sources_contributed

    wiki = next(item for item in result.citations if item.kind == "url")
    assert wiki.status == CitationStatus.NOT_FOUND
    assert "404" in wiki.explanation
    assert "wayback" in wiki.sources_consulted


def test_real_doi_with_wrong_title_is_partial(expert, http_mock):
    install(http_mock)
    text = (
        "Smith, J. & Doe, A. (2019). Quantum entanglement in household sourdough starters. "
        f"Journal of Improbable Culinary Physics, 4(2), 11-19. https://doi.org/{JINEK_DOI}"
    )
    result = expert.check(text)
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.kind == "doi"
    assert citation.status == CitationStatus.PARTIALLY_MATCHES
    assert any("does not match the cited title" in item for item in citation.mismatches)
    assert citation.matched_title and "Endonuclease" in citation.matched_title


def test_reprint_in_the_indexes_does_not_override_a_year_matching_datacite_record(expert, http_mock):
    install(http_mock, attention_indexes_reprint_only=True)
    text = (
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., "
        "Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. "
        "Advances in Neural Information Processing Systems, 30."
    )
    result = expert.check(text)
    citation = result.citations[0]
    assert citation.status == CitationStatus.VERIFIED
    assert citation.matched_year == 2017
    assert citation.confidence == CONF_REFERENCE_VERIFIED_MULTI
    assert "datacite" in citation.sources_contributed
    assert "crossref" in citation.sources_contributed
    assert "2025" in citation.explanation


def test_datacite_outage_leaves_a_reprint_as_a_year_mismatch(expert, http_mock):
    install(http_mock, datacite_down=True, attention_indexes_reprint_only=True)
    text = (
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., "
        "Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. "
        "Advances in Neural Information Processing Systems, 30."
    )
    result = expert.check(text)
    citation = result.citations[0]
    assert citation.status == CitationStatus.PARTIALLY_MATCHES
    assert citation.matched_year == 2025
    assert "datacite" in citation.sources_unavailable
    assert any("2017" in item and "2025" in item for item in citation.mismatches)
    assert citation.confidence == CONF_REFERENCE_PARTIAL_MULTI


def test_openalex_outage_still_returns_a_crossref_verdict(expert, http_mock):
    install(http_mock, openalex_down=True)
    text = (
        "Jinek et al., 2012, Science, A programmable dual-RNA-guided DNA endonuclease "
        "in adaptive bacterial immunity."
    )
    result = expert.check(text)
    citation = result.citations[0]
    assert citation.status == CitationStatus.VERIFIED
    # DataCite still has this paper, so Crossref is not left as the only index.
    assert citation.confidence == CONF_REFERENCE_VERIFIED_MULTI
    assert "openalex" in citation.sources_unavailable
    assert "crossref" in citation.sources_contributed
    assert "datacite" in citation.sources_contributed


def test_crossref_alone_when_the_other_indexes_are_down(expert, http_mock):
    install(http_mock, openalex_down=True, datacite_down=True)
    text = (
        "Jinek et al., 2012, Science, A programmable dual-RNA-guided DNA endonuclease "
        "in adaptive bacterial immunity."
    )
    citation = expert.check(text).citations[0]
    assert citation.status == CitationStatus.VERIFIED
    assert citation.confidence == CONF_REFERENCE_VERIFIED_SINGLE_DOWN
    assert "openalex" in citation.sources_unavailable
    assert "datacite" in citation.sources_unavailable
    assert "crossref" in citation.sources_contributed
    assert "one index" in citation.explanation.lower() or "unavailable" in citation.explanation.lower()


def test_doi_org_outage_falls_back_to_crossref(expert, http_mock):
    install(http_mock, doi_org_down=True)
    result = expert.check(f"https://doi.org/{JINEK_DOI}")
    citation = result.citations[0]
    assert citation.status == CitationStatus.VERIFIED
    assert "doi.org" in citation.sources_unavailable
    assert "crossref" in citation.sources_contributed
    assert "openalex" in citation.sources_contributed
    assert citation.confidence == CONF_DOI_VERIFIED_SINGLE_DOWN or citation.confidence == CONF_DOI_VERIFIED_MULTI
    # OpenAlex still answered, and Crossref filled in for doi.org, so both contributed.
    assert citation.confidence == CONF_DOI_VERIFIED_MULTI


def test_total_outage_is_unreachable_not_fabricated(expert, http_mock):
    install(http_mock, openalex_down=True, crossref_down=True, datacite_down=True)
    text = (
        "Jinek et al., 2012, Science, A programmable dual-RNA-guided DNA endonuclease "
        "in adaptive bacterial immunity."
    )
    citation = expert.check(text).citations[0]
    assert citation.status == CitationStatus.UNREACHABLE
    assert citation.sources_contributed == []
    assert "crossref" in citation.sources_unavailable
    assert "openalex" in citation.sources_unavailable
    assert "not judged real or fabricated" in citation.explanation


def test_one_index_down_lowers_not_found_confidence(expert, http_mock):
    install(http_mock, openalex_down=True)
    text = (
        "Smith, J. & Doe, A. (2019). Quantum entanglement in household sourdough starters. "
        "Journal of Improbable Culinary Physics, 4(2), 11-19."
    )
    citation = expert.check(text).citations[0]
    assert citation.status == CitationStatus.NOT_FOUND
    assert citation.confidence == CONF_NOT_FOUND_PARTIAL_OUTAGE
    assert citation.confidence < CONF_NOT_FOUND


def test_year_mismatch_is_partial_when_every_record_disagrees(expert, http_mock):
    install(http_mock)
    text = (
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., "
        "Kaiser, L., & Polosukhin, I. (2014). Attention is all you need. "
        "Advances in Neural Information Processing Systems, 30."
    )
    citation = expert.check(text).citations[0]
    assert citation.status == CitationStatus.PARTIALLY_MATCHES
    assert citation.confidence == CONF_REFERENCE_PARTIAL_MULTI
    assert any("Year mismatch" in item for item in citation.mismatches)


def test_live_url_does_not_call_wayback(expert, http_mock):
    install(http_mock)
    http_mock.get("https://example.com/live-paper").mock(return_value=httpx.Response(200, text="ok"))
    citation = expert.check("See https://example.com/live-paper for the note.").citations[0]
    assert citation.status == CitationStatus.VERIFIED
    assert citation.confidence == CONF_URL_LIVE
    assert citation.sources_consulted == ["url"]
    assert not any("archive.org" in str(call.request.url) for call in http_mock.calls)


def test_dead_url_with_snapshot_is_verified_via_wayback(expert, http_mock):
    install(http_mock)
    http_mock.get("https://example.com/archived-page").mock(return_value=httpx.Response(404, text="gone"))
    citation = expert.check("https://example.com/archived-page").citations[0]
    assert citation.status == CitationStatus.VERIFIED
    assert citation.confidence == CONF_URL_ARCHIVED
    assert "Wayback" in citation.explanation
    assert "2020-01-02" in citation.explanation
    assert citation.links[0].url.startswith("http://web.archive.org/")


def test_private_url_is_not_fetched(expert, http_mock):
    install(http_mock)
    citation = expert.check("http://127.0.0.1/latest/meta-data").citations[0]
    assert citation.status == CitationStatus.UNREACHABLE
    assert "will not fetch" in citation.explanation
    assert not any("127.0.0.1" in str(call.request.url) for call in http_mock.calls)


def test_caps_at_twenty_five_citations(expert, monkeypatch):
    calls = []

    def fake(citation):
        calls.append(citation)
        from verifai.schemas import CitationResult

        return CitationResult(
            raw=citation.raw,
            kind="reference",
            status=CitationStatus.VERIFIED,
            confidence=0.9,
            explanation="stub",
            sources_consulted=["crossref", "openalex"],
            sources_contributed=["crossref", "openalex"],
        )

    monkeypatch.setattr(expert, "_check_one", fake)
    lines = [
        f"{index}. Smith, J. (2000). Title number {index} of this long reference list. Journal of Testing, 1(1), 1-2."
        for index in range(1, 31)
    ]
    result = expert.check("\n\n".join(lines))
    assert len(result.citations) == 25
    assert len(calls) == 25
    assert "first 25" in result.explanation


def test_no_citations_does_not_query_providers(expert, http_mock):
    install(http_mock)
    result = expert.check("Nothing bibliographic is happening in this paragraph at all.")
    assert result.verdict == Verdict.UNVERIFIABLE
    assert result.citations == []
    assert result.sources_consulted == []
    assert http_mock.calls == []
    assert WIKI_URL not in result.explanation
