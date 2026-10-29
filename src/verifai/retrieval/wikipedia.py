"""Wikipedia search plus the lead extract. One encyclopedia, one family."""

from __future__ import annotations

from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json, search_terms
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://en.wikipedia.org/w/api.php"


class WikipediaRetriever:
    name = "wikipedia"

    def __init__(self, http: PoliteHttp, *, limit: int = 3) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = " ".join(search_terms(query, limit=10)) or query.strip()[:200]
        if not terms:
            return RetrievalResult(self.name)
        body, error = fetch_json(
            self.http,
            _API,
            {
                "action": "query",
                "list": "search",
                "srsearch": terms,
                "srlimit": self.limit,
                "format": "json",
                "utf8": 1,
            },
            source=self.name,
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        hits = _search_hits(body)[: self.limit]
        if not hits:
            return RetrievalResult(self.name)
        passages = self._extracts(hits)
        return RetrievalResult(self.name, passages=passages)

    def _extracts(self, hits: list[dict[str, Any]]) -> list[Passage]:
        pageids = "|".join(str(hit["pageid"]) for hit in hits if hit.get("pageid"))
        pages: dict[str, Any] = {}
        if pageids:
            body, _error = fetch_json(
                self.http,
                _API,
                {
                    "action": "query",
                    "prop": "extracts|info",
                    "exintro": 1,
                    "explaintext": 1,
                    "inprop": "url",
                    "pageids": pageids,
                    "format": "json",
                },
                source=self.name,
            )
            if body is not None:
                raw_pages = body.get("query", {}).get("pages", {})
                if isinstance(raw_pages, dict):
                    pages = raw_pages
        passages: list[Passage] = []
        for hit in hits:
            page = pages.get(str(hit.get("pageid")), {})
            extract = page.get("extract") if isinstance(page, dict) else None
            text = (
                extract
                if isinstance(extract, str) and extract.strip()
                else str(hit.get("snippet", ""))
            )
            title = str(hit.get("title") or "Wikipedia")
            url = page.get("fullurl") if isinstance(page, dict) else None
            if not isinstance(url, str):
                url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            passages.append(
                Passage(
                    source=self.name,
                    family="wikipedia",
                    title=title,
                    url=url,
                    text=_strip_markup(text)[:2000],
                )
            )
        return passages


def _search_hits(body: dict[str, Any]) -> list[dict[str, Any]]:
    search = body.get("query", {}).get("search") if isinstance(body.get("query"), dict) else None
    if not isinstance(search, list):
        return []
    return [hit for hit in search if isinstance(hit, dict)]


def _strip_markup(text: str) -> str:
    return (
        text.replace('<span class="searchmatch">', "").replace("</span>", "").replace("&quot;", '"')
    )
