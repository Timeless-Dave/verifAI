"""Wikipedia's featured news feed, filtered to stories that share the claim's words."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json, search_terms
from verifai.retrieval.types import Passage, RetrievalResult

_FEED = "https://en.wikipedia.org/api/rest_v1/feed/featured"


class CurrentEventsRetriever:
    name = "wikipedia_current"

    def __init__(self, http: PoliteHttp, *, today: Callable[[], date] | None = None) -> None:
        self.http = http
        self._today = today or _utc_today

    def retrieve(self, query: str) -> RetrievalResult:
        terms = set(search_terms(query, limit=8))
        if not terms:
            return RetrievalResult(self.name)
        day = self._today()
        body, error = fetch_json(
            self.http,
            f"{_FEED}/{day.year}/{day.month:02d}/{day.day:02d}",
            None,
            source=self.name,
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        news = body.get("news")
        if not isinstance(news, list):
            return RetrievalResult(self.name)
        passages: list[Passage] = []
        for item in news:
            passage = _story(item, terms)
            if passage is not None:
                passages.append(passage)
            if len(passages) >= 3:
                break
        return RetrievalResult(self.name, passages=passages)


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _story(item: object, terms: set[str]) -> Passage | None:
    if not isinstance(item, dict):
        return None
    story = str(item.get("story") or "").strip()
    folded = story.casefold()
    if not story or not any(term in folded for term in terms):
        return None
    url, title = _link(item, story)
    return Passage(
        source="wikipedia_current",
        family="wikipedia",
        title=title,
        url=url,
        text=story,
        speaker="Wikipedia current events",
    )


def _link(item: dict[str, Any], story: str) -> tuple[str | None, str]:
    links = item.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            title = str(link.get("title") or "").strip()
            content = link.get("content_urls")
            desktop = content.get("desktop") if isinstance(content, dict) else None
            page = desktop.get("page") if isinstance(desktop, dict) else None
            if isinstance(page, str) and page:
                return page, title or story[:80]
    return "https://en.wikipedia.org/wiki/Portal:Current_events", story[:80]
