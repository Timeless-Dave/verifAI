"""Research claims: the scholarly literature first, the encyclopedia beside it.

A sentence from a paper or an essay ("Growth mindset interventions improve
achievement for lower-achieving students") is searched in OpenAlex and arXiv,
and in Semantic Scholar when a key is set. Each paper is its own source.
Wikipedia stays as one more, independent source. When the paste cites a work
for the claim, that work is read too (see ``verifai.source_check``).
"""

from __future__ import annotations

from verifai.config import Settings
from verifai.evidence.base import EvidenceRegistry
from verifai.experts.claim_check import check_claim
from verifai.http import PoliteHttp
from verifai.models.judge import Judge
from verifai.retrieval.scholarly import (
    ArxivSearchRetriever,
    OpenAlexSearchRetriever,
    SemanticScholarSearchRetriever,
)
from verifai.retrieval.types import Retriever
from verifai.retrieval.wikipedia import WikipediaRetriever
from verifai.schemas import ExpertResult


class ResearchExpert:
    name = "research"
    description = (
        "Checks a research claim against the scholarly literature: OpenAlex (about 250 "
        "million works), arXiv, and Semantic Scholar when a key is set, plus Wikipedia. "
        "Each paper counts as its own source."
    )

    def __init__(self, retrievers: list[Retriever], judge: Judge, *, max_concurrency: int = 4):
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
            scope="research",
            claim=text,
            retrievers=self._retrievers,
            judge=self.judge,
            workers=self.max_concurrency,
        )


def build_research_expert(http: PoliteHttp, settings: Settings, judge: Judge) -> ResearchExpert:
    retrievers: list[Retriever] = [
        OpenAlexSearchRetriever(http, settings),
        ArxivSearchRetriever(http),
        WikipediaRetriever(http),
    ]
    if settings.semantic_scholar_api_key:
        retrievers.insert(
            1, SemanticScholarSearchRetriever(http, settings.semantic_scholar_api_key)
        )
    return ResearchExpert(retrievers, judge, max_concurrency=settings.max_concurrency)
