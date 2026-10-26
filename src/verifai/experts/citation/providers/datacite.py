"""DataCite REST API.

DataCite registers many preprints (including arXiv) that Crossref title
search does not return. It is an extra independent source, not a replacement
for Crossref or OpenAlex and not a sole decider. A cited year and the first
author are part of the query so a later reprint of the same title does not
crowd out the edition the citation names. One request per reference: a miss
is a negative finding, and Crossref and OpenAlex still cover a different year.
"""

from __future__ import annotations

import re

from verifai.experts.citation.providers.base import ProviderOutcome, Record
from verifai.experts.citation.providers.parsing import record_from_datacite
from verifai.http import RETRY_STATUSES, HttpResult, PoliteHttp

_DOIS = "https://api.datacite.org/dois"


class DataCiteProvider:
    name = "datacite"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def search(
        self,
        *,
        title: str | None,
        authors: list[str],
        year: int | None,
        venue: str | None,
        raw: str,
    ) -> ProviderOutcome:
        del venue, raw
        if not title:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        author = authors[0] if authors else None
        records, error = self._query(title, author, year)
        if error is not None:
            return ProviderOutcome(self.name, unavailable=True, error=error)
        return ProviderOutcome(self.name, unavailable=False, records=records[:5])

    def _query(
        self,
        title: str,
        author: str | None,
        year: int | None,
    ) -> tuple[list[Record], str | None]:
        result = self.http.get(
            _DOIS,
            params={"query": _lucene(title, author, year), "page[size]": 5},
        )
        if _down(result) or result.status_code != 200:
            return [], _error(result)
        body = result.json_body()
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            return [], "DataCite response was not a DOI list"
        records: list[Record] = []
        for item in body["data"]:
            if not isinstance(item, dict):
                continue
            record = record_from_datacite(item)
            if record is not None and record.title:
                records.append(record)
        return records, None


def _lucene(title: str, author: str | None, year: int | None) -> str:
    parts = [f'titles.title:"{_phrase(title)}"']
    if year is not None:
        parts.append(f"publicationYear:{year}")
    if author:
        parts.append(f'creators.name:"{_phrase(author)}"')
    return " AND ".join(parts)


def _phrase(value: str) -> str:
    cleaned = value.replace("\\", " ").replace('"', " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:300]


def _down(result: HttpResult) -> bool:
    return (
        result.error is not None
        or result.status_code is None
        or result.status_code in RETRY_STATUSES
    )


def _error(result: HttpResult) -> str:
    if result.error:
        return result.error[:180]
    return f"HTTP {result.status_code}"
