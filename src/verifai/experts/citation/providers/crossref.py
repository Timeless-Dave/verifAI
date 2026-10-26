"""Crossref REST API. Polite pool membership comes from the process User-Agent."""

from __future__ import annotations

from urllib.parse import quote

from verifai.experts.citation.providers.base import ProviderOutcome, Record
from verifai.experts.citation.providers.parsing import crossref_message, record_from_crossref_like
from verifai.http import RETRY_STATUSES, HttpResult, PoliteHttp

_WORKS = "https://api.crossref.org/works"


class CrossrefProvider:
    name = "crossref"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def lookup_doi(self, doi: str) -> ProviderOutcome:
        result = self.http.get(f"{_WORKS}/{quote(doi, safe='/')}")
        return self._one(result)

    def search(
        self,
        *,
        title: str | None,
        authors: list[str],
        year: int | None,
        venue: str | None,
        raw: str,
    ) -> ProviderOutcome:
        del (
            year,
            venue,
        )  # kept in the signature so providers stay interchangeable; year is judged later
        params: dict[str, object] = {"rows": 5}
        if title:
            params["query.title"] = title
            if authors:
                params["query.author"] = authors[0]
        else:
            params["query.bibliographic"] = raw[:500]
        result = self.http.get(_WORKS, params=params)
        if _unavailable(result):
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        if result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        message = crossref_message(result.json_body())
        if message is None:
            return ProviderOutcome(
                self.name, unavailable=True, error="Crossref response was not JSON"
            )
        raw_items = message.get("items")
        items = raw_items if isinstance(raw_items, list) else []
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            record = record_from_crossref_like(self.name, item)
            if record is not None:
                records.append(record)
        return ProviderOutcome(self.name, unavailable=False, records=records[:5])

    def _one(self, result: HttpResult) -> ProviderOutcome:
        if result.status_code == 404:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        if _unavailable(result) or result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        message = crossref_message(result.json_body())
        if message is None:
            return ProviderOutcome(
                self.name, unavailable=True, error="Crossref response was not JSON"
            )
        record = record_from_crossref_like(self.name, message)
        records: list[Record] = [record] if record and record.title else []
        if record and record.title is None and record.doi:
            records = [record]
        return ProviderOutcome(self.name, unavailable=False, records=records)


def _unavailable(result: HttpResult) -> bool:
    return (
        result.error is not None
        or result.status_code is None
        or result.status_code in RETRY_STATUSES
    )


def _error(result: HttpResult) -> str:
    if result.error:
        return result.error[:180]
    return f"HTTP {result.status_code}"
