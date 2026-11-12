"""Wikipedia search plus the lead extract. One encyclopedia, one family."""

from __future__ import annotations

from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import detect_language, fetch_json, search_terms_for
from verifai.retrieval.types import Passage, RetrievalResult
from verifai.retrieval.wikidata import entity_names

_DISAMBIGUATION = ("may refer to", "may also refer to", "can refer to")


def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


class WikipediaRetriever:
    name = "wikipedia"

    def __init__(self, http: PoliteHttp, *, limit: int = 3) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        # A French claim is searched on French Wikipedia. It is still one
        # encyclopedia, so every language edition is the same family.
        lang = detect_language(query)
        terms = " ".join(search_terms_for(query, lang, limit=10)) or query.strip()[:200]
        if not terms:
            return RetrievalResult(self.name)
        body, error = fetch_json(
            self.http,
            _api(lang),
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
        # The article the claim is about comes first. A keyword search for
        # "Napoleon born Corsica" ranks his mother's article above his own.
        named = self._named(query, lang)
        seen = {passage.title for passage in named}
        hits = [hit for hit in _search_hits(body) if hit.get("title") not in seen]
        hits = hits[: max(self.limit - len(named), 1)]
        passages = named + (self._extracts(hits, lang) if hits else [])
        return RetrievalResult(self.name, passages=passages)

    def _named(self, query: str, lang: str) -> list[Passage]:
        """Articles titled with the names the claim uses, redirects followed."""
        names = [name[:1].upper() + name[1:] for name in entity_names(query)[:2]]
        if not names:
            return []
        body, _error = fetch_json(
            self.http,
            _api(lang),
            {
                "action": "query",
                "titles": "|".join(names),
                "redirects": 1,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
            },
            source=self.name,
        )
        query_part = body.get("query") if body else None
        if not isinstance(query_part, dict) or not isinstance(query_part.get("pages"), dict):
            return []
        pages: dict[str, Any] = query_part["pages"]
        # Keep the claim's own order: its subject first.
        moved: dict[str, str] = {}
        for step in ("normalized", "redirects"):
            for hop in query_part.get(step) or []:
                if isinstance(hop, dict):
                    moved[str(hop.get("from"))] = str(hop.get("to"))
        order: list[str] = []
        for name in names:
            title = name
            for _ in range(3):
                title = moved.get(title, title)
            order.append(title)
        found: list[Passage] = []
        ranked = sorted(
            (page for page in pages.values() if isinstance(page, dict)),
            key=lambda page: _rank(order, page.get("title")),
        )
        for page in ranked:
            extract = page.get("extract")
            heading = page.get("title")
            if "missing" in page or not isinstance(extract, str) or not isinstance(heading, str):
                continue
            if not extract.strip():
                continue
            if any(marker in extract[:300] for marker in _DISAMBIGUATION):
                continue
            url = page.get("fullurl")
            found.append(
                Passage(
                    source=self.name,
                    family="wikipedia",
                    title=heading,
                    url=url
                    if isinstance(url, str)
                    else f"https://{lang}.wikipedia.org/wiki/" + heading.replace(" ", "_"),
                    text=_strip_markup(extract)[:2000],
                    about=_original_name(names, order, heading),
                )
            )
        return found[:2]

    def _extracts(self, hits: list[dict[str, Any]], lang: str = "en") -> list[Passage]:
        pageids = "|".join(str(hit["pageid"]) for hit in hits if hit.get("pageid"))
        pages: dict[str, Any] = {}
        if pageids:
            body, _error = fetch_json(
                self.http,
                _api(lang),
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
                url = f"https://{lang}.wikipedia.org/wiki/" + title.replace(" ", "_")
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


def _original_name(names: list[str], order: list[str], title: str) -> str | None:
    """The claim's own words for an article reached through a redirect."""
    return names[order.index(title)] if title in order else None


def _rank(order: list[str], title: object) -> int:
    return order.index(title) if isinstance(title, str) and title in order else len(order)


def _search_hits(body: dict[str, Any]) -> list[dict[str, Any]]:
    search = body.get("query", {}).get("search") if isinstance(body.get("query"), dict) else None
    if not isinstance(search, list):
        return []
    return [hit for hit in search if isinstance(hit, dict)]


def _strip_markup(text: str) -> str:
    return (
        text.replace('<span class="searchmatch">', "").replace("</span>", "").replace("&quot;", '"')
    )
