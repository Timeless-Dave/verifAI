"""Wikidata entity search. Descriptions are a separate database from Wikipedia prose."""

from __future__ import annotations

from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://www.wikidata.org/w/api.php"


class WikidataRetriever:
    name = "wikidata"

    def __init__(self, http: PoliteHttp, *, limit: int = 3) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        text = " ".join((query or "").split())[:200]
        if not text:
            return RetrievalResult(self.name)
        body, error = fetch_json(
            self.http,
            _API,
            {
                "action": "wbsearchentities",
                "search": text,
                "language": "en",
                "format": "json",
                "limit": self.limit,
                "type": "item",
            },
            source=self.name,
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        passages = [
            passage for hit in _hits(body)[: self.limit] if (passage := _passage(hit)) is not None
        ]
        return RetrievalResult(self.name, passages=passages)


def _hits(body: dict[str, Any]) -> list[dict[str, Any]]:
    search = body.get("search")
    if not isinstance(search, list):
        return []
    return [hit for hit in search if isinstance(hit, dict)]


def _passage(hit: dict[str, Any]) -> Passage | None:
    label = str(hit.get("label") or "").strip()
    description = str(hit.get("description") or "").strip()
    if not label and not description:
        return None
    entity_id = str(hit.get("id") or "")
    url = hit.get("concepturi")
    if not isinstance(url, str) or not url:
        url = f"https://www.wikidata.org/wiki/{entity_id}" if entity_id else None
    text = f"{label}. {description}".strip(". ")
    return Passage(
        source="wikidata",
        family="wikidata",
        title=label or entity_id or "Wikidata",
        url=url,
        text=text,
    )
