"""News and politics claims. Show who said what, and do not let Google decide."""

from __future__ import annotations

from verifai.config import Settings
from verifai.evidence.base import EvidenceRegistry
from verifai.experts.claim_check import check_claim, google_note
from verifai.http import PoliteHttp
from verifai.models.judge import Judge
from verifai.retrieval.current_events import CurrentEventsRetriever
from verifai.retrieval.federal_register import FederalRegisterRetriever
from verifai.retrieval.google_factcheck import GoogleFactCheckRetriever
from verifai.retrieval.news_search import build_major_news, build_us_government
from verifai.retrieval.pages import PageRetriever
from verifai.retrieval.types import Retriever
from verifai.retrieval.wikidata import WikidataRetriever
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import ExpertResult


class NewsExpert:
    name = "news"
    description = (
        "Checks a news or politics claim against established newsrooms (AP, Reuters, "
        "the BBC, NPR, and others), official US government sites such as the White "
        "House and the FDA, the Federal Register, Google Fact Check reviews, Wikipedia, "
        "and linked pages. Each outlet and each agency counts as its own source."
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
        build_major_news(http),
        build_us_government(http),
        FederalRegisterRetriever(http),
        CurrentEventsRetriever(http),
        WikipediaRetriever(http),
        WikidataRetriever(http),
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
