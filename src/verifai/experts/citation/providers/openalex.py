"""OpenAlex works API.

Title search uses ``filter=title.search:`` so the request stays a title
lookup. The bare ``search`` parameter currently scans full text and spends
the shared unauthenticated budget much faster. Calls still send ``mailto``.
Without an API key, OpenAlex applies a small shared daily budget per IP;
a 402 or 429 is treated as this provider being unavailable, not as a
negative finding.
"""

from __future__ import annotations

import re

from verifai.experts.citation.providers.base import ProviderOutcome
from verifai.experts.citation.providers.parsing import record_from_openalex
from verifai.http import RETRY_STATUSES, HttpResult, PoliteHttp


class OpenAlexProvider:
    name = "openalex"

    def __init__(self, http: PoliteHttp, *, mailto: str, api_key: str | None = None) -> None:
        self.http = http
        self.mailto = mailto
        self.api_key = api_key

    def lookup_doi(self, doi: str) -> ProviderOutcome:
        # The DOI is part of the path (`doi:10.x/y`). httpx keeps slashes.
        result = self.http.get(
            f"https://api.openalex.org/works/doi:{doi}",
            params=self._params(),
        )
        if result.status_code == 404:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        return self._parse_one(result)

    def search(
        self,
        *,
        title: str | None,
        authors: list[str],
        year: int | None,
        venue: str | None,
        raw: str,
    ) -> ProviderOutcome:
        del authors, year, venue, raw
        if not title:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        params = self._params()
        params["filter"] = "title.search:" + _filter_value(title)
        params["per-page"] = 5
        result = self.http.get("https://api.openalex.org/works", params=params)
        if _down(result):
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        if result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        body = result.json_body()
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            return ProviderOutcome(
                self.name, unavailable=True, error="OpenAlex response was not a work list"
            )
        records = []
        for work in body["results"]:
            record = record_from_openalex(work)
            if record is not None:
                records.append(record)
        return ProviderOutcome(self.name, unavailable=False, records=records[:5])

    def _parse_one(self, result: HttpResult) -> ProviderOutcome:
        if _down(result) or result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=_error(result))
        body = result.json_body()
        if not isinstance(body, dict):
            return ProviderOutcome(
                self.name, unavailable=True, error="OpenAlex response was not JSON"
            )
        record = record_from_openalex(body)
        if record is None:
            return ProviderOutcome(self.name, unavailable=True, error="OpenAlex work had no title")
        return ProviderOutcome(self.name, unavailable=False, records=[record])

    def _params(self) -> dict[str, object]:
        params: dict[str, object] = {"mailto": self.mailto}
        if self.api_key:
            params["api_key"] = self.api_key
        return params


def _filter_value(title: str) -> str:
    cleaned = title.replace(":", " ").replace(",", " ").replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:250]


def _down(result: HttpResult) -> bool:
    return (
        result.error is not None
        or result.status_code is None
        or result.status_code in RETRY_STATUSES | {402}
    )


def _error(result: HttpResult) -> str:
    if result.error:
        return result.error[:180]
    return f"HTTP {result.status_code}"
