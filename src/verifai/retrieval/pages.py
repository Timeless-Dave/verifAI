"""Fetch public pages linked from a paste. Each site is its own source."""

from __future__ import annotations

from verifai.http import PoliteHttp
from verifai.net import public_url_problem
from verifai.retrieval.textutil import extract_urls, host_family, html_to_text
from verifai.retrieval.types import Passage, RetrievalResult


class PageRetriever:
    name = "page"

    def __init__(self, http: PoliteHttp, *, limit: int = 3) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        urls = extract_urls(query, limit=self.limit)
        passages: list[Passage] = []
        failures = 0
        for url in urls:
            passage, failed = self.fetch(url)
            if failed:
                failures += 1
            if passage is not None:
                passages.append(passage)
        if urls and failures == len(urls) and not passages:
            return RetrievalResult(self.name, unavailable=True, error="linked pages did not load")
        return RetrievalResult(self.name, passages=passages)

    def fetch(
        self, url: str, *, family: str | None = None, source: str | None = None
    ) -> tuple[Passage | None, bool]:
        problem = public_url_problem(url)
        if problem:
            return None, False
        result = self.http.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
        if result.error or result.status_code != 200:
            return None, True
        content_type = result.headers.get("content-type", "")
        if "pdf" in content_type or "octet-stream" in content_type:
            return None, False
        text = html_to_text(result.content)
        if len(text) < 40:
            return None, False
        return (
            Passage(
                source=source or self.name,
                family=family or host_family(url),
                title=_title(text, url),
                url=url,
                text=text,
                speaker=host_family(url).removeprefix("page:"),
            ),
            False,
        )


def _title(text: str, url: str) -> str:
    first = text.split("\n", 1)[0].strip()
    if 8 <= len(first) <= 140:
        return first
    return url
