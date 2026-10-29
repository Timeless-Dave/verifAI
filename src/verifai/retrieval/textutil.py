"""Small text helpers shared by the free-source retrievers."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from verifai.corroboration import Stance
from verifai.http import PoliteHttp

_URL = re.compile(r"https?://[^\s<>\"']+")
_YEAR = re.compile(r"(?:18|19|20)\d{2}")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]{3,}")
_STOP = frozenset(
    [
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "into",
        "over",
        "after",
        "before",
        "about",
        "than",
        "then",
        "have",
        "been",
        "were",
        "they",
        "them",
        "their",
        "your",
        "what",
        "when",
        "where",
        "which",
        "while",
        "would",
        "could",
        "should",
        "there",
        "their",
        "just",
        "also",
        "more",
        "most",
        "some",
        "such",
        "only",
        "other",
        "into",
        "does",
        "did",
        "done",
        "doing",
        "make",
        "made",
        "many",
        "much",
        "very",
        "will",
        "shall",
        "into",
        "onto",
    ]
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(content: bytes, *, limit: int = 4000) -> str:
    parser = _TextExtractor()
    parser.feed(content.decode("utf-8", errors="replace"))
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:limit]


def extract_urls(text: str, *, limit: int = 3) -> list[str]:
    found: list[str] = []
    for match in _URL.finditer(text or ""):
        url = match.group(0).rstrip(").,;]")
        host = (urlsplit(url).hostname or "").lower()
        if host in {"doi.org", "dx.doi.org"} or url in found:
            continue
        found.append(url)
        if len(found) >= limit:
            break
    return found


def search_terms(text: str, *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for word in _WORD.findall(text or ""):
        folded = word.casefold()
        if folded in _STOP or folded in terms:
            continue
        terms.append(folded)
        if len(terms) >= limit:
            break
    return terms


def year_of(value: object) -> int | None:
    if isinstance(value, int) and 1800 <= value <= 2100:
        return value
    if isinstance(value, str):
        match = _YEAR.search(value)
        if match:
            return int(match.group(0))
    return None


def document_ids(*, doi: str | None = None, pmid: str | None = None) -> tuple[str, ...]:
    ids: list[str] = []
    if doi:
        cleaned = doi.strip().lower()
        cleaned = cleaned.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        cleaned = cleaned.removeprefix("doi:")
        if cleaned:
            ids.append(f"doi:{cleaned}")
    if pmid and str(pmid).isdigit():
        ids.append(f"pmid:{pmid}")
    return tuple(ids)


def stance_from_rating(rating: str) -> tuple[Stance, float]:
    text = " ".join(rating.casefold().split())
    if not text:
        return Stance.NEUTRAL, 0.0
    for bit in ("half true", "half-true", "mixture", "mixed", "unproven", "satire", "unverified"):
        if bit in text:
            return Stance.NEUTRAL, 0.4
    for bit, confidence in (
        ("pants on fire", 0.9),
        ("four pinocchios", 0.9),
        ("mostly false", 0.75),
    ):
        if bit in text:
            return Stance.REFUTES, confidence
    if "mostly true" in text:
        return Stance.SUPPORTS, 0.75
    for bit in ("false", "fake", "incorrect", "misleading"):
        if bit in text:
            return Stance.REFUTES, 0.65 if bit == "misleading" else 0.9
    for bit in ("true", "correct", "accurate"):
        if bit in text:
            return Stance.SUPPORTS, 0.9
    return Stance.NEUTRAL, 0.3


def fetch_json(
    http: PoliteHttp,
    url: str,
    params: dict[str, Any] | None,
    *,
    source: str,
) -> tuple[dict[str, Any] | None, str | None]:
    result = http.get(url, params=params)
    if result.error or result.status_code != 200:
        detail = result.error or f"HTTP {result.status_code}"
        return None, detail
    body = result.json_body()
    if not isinstance(body, dict):
        return None, f"{source} response was not JSON"
    return body, None


def host_family(url: str) -> str:
    host = (urlsplit(url).hostname or "page").lower().removeprefix("www.")
    return f"page:{host}"
