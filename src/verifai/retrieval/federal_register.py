"""The Federal Register: the US government's official journal.

Every federal rule, proposed rule, agency notice, and presidential document
(executive orders and proclamations) is published here with a summary. It
is one source, however many of its documents match.
"""

from __future__ import annotations

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json, search_terms
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://www.federalregister.gov/api/v1/documents.json"
_FIELDS = ("title", "abstract", "html_url", "agencies", "publication_date", "type")


class FederalRegisterRetriever:
    name = "federal_register"

    def __init__(self, http: PoliteHttp, *, limit: int = 2) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        words = search_terms(query, limit=8)
        if len(words) < 2:
            return RetrievalResult(self.name)
        params: dict[str, object] = {
            "conditions[term]": " ".join(words),
            "per_page": self.limit,
            "order": "relevance",
            "fields[]": list(_FIELDS),
        }
        body, error = fetch_json(self.http, _API, params, source=self.name)
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        results = body.get("results")
        if not isinstance(results, list):
            return RetrievalResult(self.name)
        passages = [passage for item in results if (passage := _document(item))]
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _document(item: object) -> Passage | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    url = item.get("html_url")
    if not title or not isinstance(url, str):
        return None
    abstract = str(item.get("abstract") or "").strip()
    kind = str(item.get("type") or "Document").strip()
    published = str(item.get("publication_date") or "").strip()
    agencies = [
        str(agency.get("name"))
        for agency in item.get("agencies") or []
        if isinstance(agency, dict) and agency.get("name")
    ]
    by = ", ".join(dict.fromkeys(agencies)) or "a federal agency"
    header = f"{kind} by {by}" + (f", published {published}" if published else "")
    text = f"{title}\n({header}.)" + (f"\n{abstract}" if abstract else "")
    year = int(published[:4]) if published[:4].isdigit() else None
    return Passage(
        source="federal_register",
        family="federal_register",
        title=title,
        url=url,
        text=text,
        year=year,
        venue="Federal Register",
        speaker=by,
        publication_type=kind.casefold(),
    )
