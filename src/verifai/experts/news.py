"""News and politics claims. Show who said what, and do not let Google decide."""

from __future__ import annotations

from verifai.config import Settings
from verifai.evidence.base import EvidenceRegistry
from verifai.experts.claim_check import check_claim, google_note
from verifai.http import PoliteHttp
from verifai.models.judge import Judge
from verifai.retrieval.current_events import CurrentEventsRetriever
from verifai.retrieval.gdelt import GdeltRetriever
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.pages import PageRetriever
from verifai.retrieval.types import Retriever
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import ExpertResult


class NewsExpert:
    name = "news"
    description = (
        "Checks a news or politics claim against Google Fact Check reviews, "
        "GDELT outlets, Wikipedia current events, and article pages. "
        "Each record names who said it and links the source."
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
            scope="news",
            claim=text,
            retrievers=self._retrievers,
            judge=self.judge,
            workers=self.max_concurrency,
            note=google_note(configured=self.google_configured),
        )


def build_news_expert(http: PoliteHttp, settings: Settings, judge: Judge) -> NewsExpert:
    retrievers: list[Retriever] = [
        GdeltRetriever(http),
        CurrentEventsRetriever(http),
        WikipediaRetriever(http),
        PageRetriever(http),
    ]
    configured = bool(settings.google_factcheck_api_key)
    if settings.google_factcheck_api_key:
        retrievers.insert(0, GoogleFactCheckRetriever(http, settings.google_factcheck_api_key))
    return NewsExpert(
        retrievers,
        judge,
        google_configured=configured,
        max_concurrency=settings.max_concurrency,
    )
