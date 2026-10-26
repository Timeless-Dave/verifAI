import pytest

from verifai.evidence.base import EVIDENCE_PRINCIPLE
from verifai.experts.base import Expert
from verifai.experts.citation.expert import CitationExpert
from verifai.schemas import EVIDENCE_PRINCIPLE as SCHEMA_PRINCIPLE


def test_principle_is_shared_and_forbids_a_sole_decider():
    assert "sole decider" in EVIDENCE_PRINCIPLE
    assert "Google Fact Check" in EVIDENCE_PRINCIPLE
    assert SCHEMA_PRINCIPLE == EVIDENCE_PRINCIPLE
    assert "Google Fact Check" in Expert.__doc__
    assert "Crossref and OpenAlex" in Expert.__doc__


def test_citation_expert_refuses_a_single_index(polite_http):
    from verifai.experts.citation.providers.crossref import CrossrefProvider
    from verifai.experts.citation.providers.doi_org import DoiOrgProvider
    from verifai.experts.citation.providers.url_check import UrlLivenessProvider
    from verifai.experts.citation.providers.wayback import WaybackProvider

    with pytest.raises(ValueError, match="OpenAlex"):
        CitationExpert(
            [
                DoiOrgProvider(polite_http),
                CrossrefProvider(polite_http),
                UrlLivenessProvider(polite_http),
                WaybackProvider(polite_http),
            ]
        )
