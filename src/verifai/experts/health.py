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
from verifai.retrieval.pubmed import PubMedRetriever
from verifai.retrieval.types import Retriever
from verifai.schemas import ExpertResult


class HealthExpert:
    name = "health"
    description = (
        "Checks a health or science claim against PubMed and Europe PMC, "
        "preferring systematic reviews. This is not medical advice."
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
    ]
    return HealthExpert(retrievers, judge, max_concurrency=settings.max_concurrency)
