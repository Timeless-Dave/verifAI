"""Health and science claims. PubMed and Europe PMC, reviews first.

The wording is deliberately not medical advice.
"""

from __future__ import annotations

from verifai.config import Settings
from verifai.evidence.base import EvidenceRegistry
from verifai.experts.claim_check import NOT_MEDICAL_ADVICE, check_claim
from verifai.http import PoliteHttp
from verifai.models.judge import Judge
from verifai.retrieval.europepmc import EuropePMCRetriever
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.news_search import build_us_government
from verifai.retrieval.pubmed import PubMedRetriever
from verifai.retrieval.types import Retriever
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import ExpertResult


class HealthExpert:
    name = "health"
    description = (
        "Checks a health or science claim against PubMed and Europe PMC, "
        "preferring systematic reviews, plus Wikipedia, the FDA, CDC, NIH and other "
        "federal health agencies, and Google Fact Check when it is configured. This is not medical advice."
    )

    def __init__(
        self,
        retrievers: list[Retriever],
        judge: Judge,
        *,
        max_concurrency: int = 4,
    ) -> None:
        self.judge = judge
        self.max_concurrency = max(1, max_concurrency)
        self.registry = EvidenceRegistry()
        self._retrievers: list[Retriever] = []
        for retriever in retrievers:
            self.registry.register(retriever)
            self._retrievers.append(retriever)

    def check(self, text: str) -> ExpertResult:
        return check_claim(
            expert=self.name,
            scope="health",
            claim=text,
            retrievers=self._retrievers,
            judge=self.judge,
            workers=self.max_concurrency,
            disclaimer=NOT_MEDICAL_ADVICE,
        )


def build_health_expert(http: PoliteHttp, settings: Settings, judge: Judge) -> HealthExpert:
    retrievers: list[Retriever] = [
        PubMedRetriever(http, email=settings.mailto),
        EuropePMCRetriever(http),
        # An encyclopedia covers the dangerous myths ("bleach cures COVID")
        # that no study tests, and it is independent of the literature.
        WikipediaRetriever(http),
        # What the FDA approved, recalled, or warned about, and what the CDC
        # and NIH say, in the agencies' own words. Each agency is one source.
        build_us_government(http),
    ]
    # Health myths ("vaccines cause autism") are what fact-checkers review
    # most. Their reviews still count as one source, beside the literature.
    if settings.google_factcheck_api_key:
        retrievers.append(GoogleFactCheckRetriever(http, settings.google_factcheck_api_key))
    return HealthExpert(retrievers, judge, max_concurrency=settings.max_concurrency)
