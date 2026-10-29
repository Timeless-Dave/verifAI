"""GDELT DOC API. Each news outlet is its own source; GDELT is the index."""

from __future__ import annotations

from urllib.parse import urlsplit

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json, search_terms
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltRetriever:
    name = "gdelt"

    def __init__(self, http: PoliteHttp, *, limit: int = 5) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = search_terms(query, limit=6)
        if not terms:
            return RetrievalResult(self.name)
        body, error = fetch_json(
            self.http,
            _API,
            {
                "query": " OR ".join(terms),
                "mode": "ArtList",
                "maxrecords": self.limit,
                "format": "json",
                "sort": "datedesc",
            },
            source=self.name,
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        articles = body.get("articles")
        if articles is None:
            return RetrievalResult(self.name)
        if not isinstance(articles, list):
            return RetrievalResult(
                self.name, unavailable=True, error="GDELT articles were not a list"
            )
        passages = [passage for item in articles if (passage := _article(item))]
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _article(item: object) -> Passage | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    url = item.get("url")
    if not title or not isinstance(url, str):
        return None
    domain = str(item.get("domain") or urlsplit(url).hostname or "").lower().removeprefix("www.")
    seen = str(item.get("seendate") or "")
    country = str(item.get("sourcecountry") or "")
    extra = ", ".join(part for part in (seen, country) if part)
    text = title if not extra else f"{title} ({extra})"
    return Passage(
        source="gdelt",
        family=f"outlet:{domain or 'unknown'}",
        title=title,
        url=url,
        text=text,
        speaker=domain or None,
    )
