"""Optional Semantic Scholar Graph API.

Built only when ``VERIFAI_SEMANTIC_SCHOLAR_API_KEY`` is set. Without a key
the public endpoint rate-limits hard, so it is not called at all. When it
is configured it is a third opinion, never the only one.
"""

from __future__ import annotations

from verifai.experts.citation.providers.base import ProviderOutcome
from verifai.experts.citation.providers.parsing import record_from_semantic_scholar
from verifai.http import RETRY_STATUSES, HttpResult, PoliteHttp

_FIELDS = "title,year,authors,venue,externalIds,url,paperId"
_PAPER = "https://api.semanticscholar.org/graph/v1/paper"


class SemanticScholarProvider:
    name = "semantic_scholar"

    def __init__(self, http: PoliteHttp, *, api_key: str) -> None:
        self.http = http
        self.api_key = api_key

    def lookup_doi(self, doi: str) -> ProviderOutcome:
        result = self.http.get(
            f"{_PAPER}/DOI:{doi}",
            params={"fields": _FIELDS},
            headers={"x-api-key": self.api_key},
        )
        if result.status_code == 404:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        return self._one(result)

    def search(
        self,
        *,
        title: str | None,
        authors: list[str],
        year: int | None,
        venue: str | None,
        raw: str,
    ) -> ProviderOutcome:
        del authors, year, venue
        query = title or raw[:300]
        result = self.http.get(
            f"{_PAPER}/search",
            params={"query": query, "limit": 5, "fields": _FIELDS},
            headers={"x-api-key": self.api_key},
        )
        if _down(result) or result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        body = result.json_body()
        papers = body.get("data") if isinstance(body, dict) else None
        if not isinstance(papers, list):
            return ProviderOutcome(
                self.name, unavailable=True, error="Semantic Scholar response was not a paper list"
            )
        records = []
        for paper in papers:
            record = record_from_semantic_scholar(paper)
            if record is not None:
                records.append(record)
        return ProviderOutcome(self.name, unavailable=False, records=records[:5])

    def _one(self, result: HttpResult) -> ProviderOutcome:
        if _down(result) or result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        body = result.json_body()
        if not isinstance(body, dict):
            return ProviderOutcome(
                self.name, unavailable=True, error="Semantic Scholar response was not JSON"
            )
        record = record_from_semantic_scholar(body)
        if record is None:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        return ProviderOutcome(self.name, unavailable=False, records=[record])


def _down(result: HttpResult) -> bool:
    return (
        result.error is not None
        or result.status_code is None
        or result.status_code in RETRY_STATUSES
    )


def _error(result: HttpResult) -> str:
    if result.error:
        return result.error[:180]
    return f"HTTP {result.status_code}"
