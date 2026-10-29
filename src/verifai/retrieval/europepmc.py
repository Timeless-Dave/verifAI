"""Europe PMC search. Review articles are requested before other records."""

from __future__ import annotations

from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import document_ids, fetch_json, search_terms, year_of
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_REVIEW = '(PUB_TYPE:"systematic review" OR PUB_TYPE:"meta-analysis" OR PUB_TYPE:"review")'


class EuropePMCRetriever:
    name = "europepmc"

    def __init__(self, http: PoliteHttp, *, limit: int = 4) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = " ".join(search_terms(query, limit=8))
        if not terms:
            return RetrievalResult(self.name)
        records, error = self._search(f"({terms}) AND {_REVIEW}")
        if error:
            return RetrievalResult(self.name, unavailable=True, error=error)
        kind = "systematic review" if records else None
        if not records:
            records, error = self._search(terms)
            if error:
                return RetrievalResult(self.name, unavailable=True, error=error)
        passages = [passage for item in records[: self.limit] if (passage := _passage(item, kind))]
        return RetrievalResult(self.name, passages=passages)

    def _search(self, query: str) -> tuple[list[dict[str, Any]], str | None]:
        body, error = fetch_json(
            self.http,
            _API,
            {
                "query": query,
                "format": "json",
                "pageSize": self.limit,
                "resultType": "core",
            },
            source=self.name,
        )
        if body is None:
            return [], error
        result_list = body.get("resultList")
        results = result_list.get("result") if isinstance(result_list, dict) else None
        if results is None:
            return [], None
        if not isinstance(results, list):
            return [], "Europe PMC result list was not a list"
        return [item for item in results if isinstance(item, dict)], None


def _passage(item: dict[str, Any], kind: str | None) -> Passage | None:
    title = str(item.get("title") or "").strip()
    abstract = str(item.get("abstractText") or "").strip()
    if not title and not abstract:
        return None
    pmid = str(item.get("pmid") or item.get("id") or "")
    doi = str(item.get("doi") or "")
    pub_type = str(item.get("pubType") or "")
    publication = kind
    if (
        "systematic" in pub_type.casefold()
        or "meta-analysis" in pub_type.casefold()
        or "review" in pub_type.casefold()
    ):
        publication = (
            "systematic review"
            if "systematic" in pub_type.casefold() or "meta-analysis" in pub_type.casefold()
            else "review"
        )
    url = f"https://europepmc.org/article/MED/{pmid}" if pmid.isdigit() else None
    if doi:
        url = f"https://doi.org/{doi}"
    return Passage(
        source="europepmc",
        family="europepmc",
        title=title or f"PMID {pmid}",
        url=url,
        text=abstract or title,
        document_ids=document_ids(doi=doi or None, pmid=pmid if pmid.isdigit() else None),
        authors=tuple(
            part.strip() for part in str(item.get("authorString") or "").split(",") if part.strip()
        ),
        year=year_of(item.get("pubYear")),
        venue=str(item.get("journalTitle") or "") or None,
        publication_type=publication,
    )
