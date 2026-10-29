"""Everyday claims: Wikipedia, Wikidata, optional fact checks, and pasted pages."""

from __future__ import annotations

from verifai.config import Settings
from verifai.evidence.base import EvidenceRegistry
from verifai.experts.claim_check import check_claim, google_note
from verifai.http import PoliteHttp
from verifai.models.judge import Judge
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.pages import PageRetriever
from verifai.retrieval.types import Retriever
from verifai.retrieval.wikidata import WikidataRetriever
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import ExpertResult


class GeneralExpert:
    name = "general"
    description = (
        "Checks an everyday claim against Wikipedia, Wikidata, pages linked in "
        "the paste, and Google Fact Check reviews when an API key is set."
    )

    def __init__(
        self,
        retrievers: list[Retriever],
        judge: Judge,
        *,
        google_configured: bool,
        max_concurrency: int = 4,
    ) -> None:
        self.judge = judge
        self.google_configured = google_configured
        self.max_concurrency = max(1, max_concurrency)
        self.registry = EvidenceRegistry()
        self._retrievers: list[Retriever] = []
        for retriever in retrievers:
            self.registry.register(retriever)
            self._retrievers.append(retriever)

    def check(self, text: str) -> ExpertResult:
        return check_claim(
            expert=self.name,
            scope="claim",
            claim=text,
            retrievers=self._retrievers,
            judge=self.judge,
            workers=self.max_concurrency,
            note=google_note(configured=self.google_configured),
        )


def build_general_expert(http: PoliteHttp, settings: Settings, judge: Judge) -> GeneralExpert:
    retrievers: list[Retriever] = [
        WikipediaRetriever(http),
        WikidataRetriever(http),
        PageRetriever(http),
    ]
    configured = bool(settings.google_factcheck_api_key)
    if settings.google_factcheck_api_key:
        retrievers.append(GoogleFactCheckRetriever(http, settings.google_factcheck_api_key))
    return GeneralExpert(
        retrievers,
        judge,
        google_configured=configured,
        max_concurrency=settings.max_concurrency,
    )
