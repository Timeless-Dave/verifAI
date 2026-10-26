"""doi.org content negotiation.

This is the registry resolver. For Crossref-registered DOIs the document
comes from Crossref; DataCite and other agencies answer here too. It is not
an independent copy of the Crossref index. OpenAlex is the cross-check.
"""

from __future__ import annotations

from urllib.parse import quote

from verifai.experts.citation.providers.base import ProviderOutcome
from verifai.experts.citation.providers.parsing import crossref_message, record_from_crossref_like
from verifai.http import RETRY_STATUSES, PoliteHttp

_ACCEPT = "application/vnd.citationstyles.csl+json, application/json;q=0.9"


class DoiOrgProvider:
    name = "doi.org"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def lookup_doi(self, doi: str) -> ProviderOutcome:
        result = self.http.get(
            f"https://doi.org/{quote(doi, safe='/')}",
            headers={"Accept": _ACCEPT},
        )
        if result.status_code == 404:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        if (
            result.error is not None
            or result.status_code is None
            or result.status_code in RETRY_STATUSES
        ):
            return ProviderOutcome(
                self.name, unavailable=True, error=result.error or f"HTTP {result.status_code}"
            )
        if result.status_code != 200:
            return ProviderOutcome(self.name, unavailable=True, error=f"HTTP {result.status_code}")
        message = crossref_message(result.json_body())
        if message is None:
            return ProviderOutcome(
                self.name, unavailable=True, error="doi.org response was not JSON"
            )
        record = record_from_crossref_like(self.name, message)
        if record is None:
            return ProviderOutcome(
                self.name, unavailable=True, error="doi.org record had no title or DOI"
            )
        return ProviderOutcome(self.name, unavailable=False, records=[record])
