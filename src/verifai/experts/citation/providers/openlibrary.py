"""Open Library: does a book with this ISBN exist?

Open Library's catalogue is large but incomplete. A miss there is not
evidence that a book is fake. A number that fails the ISBN checksum is.
"""

from __future__ import annotations

from verifai.experts.citation.providers.base import ProviderOutcome, Record
from verifai.http import RETRY_STATUSES, PoliteHttp


class OpenLibraryProvider:
    name = "openlibrary"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def lookup_isbn(self, isbn: str) -> ProviderOutcome:
        result = self.http.get(f"https://openlibrary.org/isbn/{isbn}.json")
        if result.status_code == 404:
            return ProviderOutcome(self.name, unavailable=False, records=[])
        if result.error is not None or result.status_code in RETRY_STATUSES or not result.ok:
            return ProviderOutcome(
                self.name, unavailable=True, error=result.error or f"HTTP {result.status_code}"
            )
        body = result.json_body()
        if not isinstance(body, dict) or not isinstance(body.get("title"), str):
            return ProviderOutcome(self.name, unavailable=True, error="Open Library sent no title")
        year = _year(body.get("publish_date"))
        publishers = body.get("publishers")
        venue = publishers[0] if isinstance(publishers, list) and publishers else None
        key = body.get("key") if isinstance(body.get("key"), str) else None
        record = Record(
            source=self.name,
            title=body["title"],
            authors=[],
            surnames=[],
            year=year,
            years=[year] if year else [],
            venue=venue if isinstance(venue, str) else None,
            doi=None,
            url=f"https://openlibrary.org{key}" if key else f"https://openlibrary.org/isbn/{isbn}",
            record_id=isbn,
        )
        return ProviderOutcome(self.name, unavailable=False, records=[record])


def _year(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    digits = [part for part in value.replace(",", " ").split() if part.isdigit() and len(part) == 4]
    return int(digits[-1]) if digits else None
