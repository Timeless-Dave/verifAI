"""Direct liveness check for an http(s) URL, and what the page says it is."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from verifai.experts.citation.providers.base import UrlCheck
from verifai.http import PoliteHttp
from verifai.net import public_url_problem

_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r"""([a-zA-Z_:.-]+)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_DOI_IN = re.compile(r"\b(10\.\d{4,9}/\S+)", re.IGNORECASE)
_DOI_NAMES = ("citation_doi", "dc.identifier", "prism.doi", "bepress_citation_doi")
_TITLE_NAMES = ("citation_title", "dc.title", "og:title", "twitter:title")


@dataclass(frozen=True)
class PageMeta:
    doi: str | None
    title: str | None


class UrlLivenessProvider:
    name = "url"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def check(self, url: str) -> UrlCheck:
        problem = public_url_problem(url)
        if problem:
            return UrlCheck(status_code=None, final_url=None, error=None, blocked_reason=problem)
        result = self.http.probe(url)
        return UrlCheck(
            status_code=result.status_code,
            final_url=result.url,
            error=result.error,
            blocked_reason=None,
        )

    def page_meta(self, url: str) -> PageMeta | None:
        """The DOI and title a live HTML page declares, or None if it cannot be read."""
        if public_url_problem(url):
            return None
        result = self.http.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
        if result.error or result.status_code != 200:
            return None
        if "html" not in result.headers.get("content-type", "html"):
            return None
        return parse_page_meta(result.content[:400_000].decode("utf-8", "replace"))


def parse_page_meta(document: str) -> PageMeta:
    named: dict[str, str] = {}
    for tag in _META_TAG.findall(document):
        attrs = {key.lower(): value.strip("\"'") for key, value in _ATTR.findall(tag)}
        key = (attrs.get("name") or attrs.get("property") or "").strip().lower()
        content = attrs.get("content")
        if key and content and key not in named:
            named[key] = html.unescape(content).strip()
    doi = None
    for key in _DOI_NAMES:
        found = _DOI_IN.search(named.get(key, ""))
        if found:
            doi = found.group(1).rstrip(".,;")
            break
    title = next((named[key] for key in _TITLE_NAMES if named.get(key)), None)
    if title is None:
        match = _TITLE.search(document)
        if match:
            title = " ".join(html.unescape(match.group(1)).split()) or None
    return PageMeta(doi=doi, title=title[:300] if title else None)
