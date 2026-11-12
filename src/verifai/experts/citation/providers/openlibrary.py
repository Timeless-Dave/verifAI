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
        authors = self._authors(body)
        record = Record(
            source=self.name,
            title=body["title"],
            authors=authors,
            surnames=[name.split()[-1] for name in authors if name.split()],
            year=year,
            years=[year] if year else [],
            venue=venue if isinstance(venue, str) else None,
            doi=None,
            url=f"https://openlibrary.org{key}" if key else f"https://openlibrary.org/isbn/{isbn}",
            record_id=isbn,
        )
        return ProviderOutcome(self.name, unavailable=False, records=[record])

    def _authors(self, body: dict[str, object]) -> list[str]:
        """The first author's name. Anyone can add an edition; a real one names who wrote it."""
        by = body.get("by_statement")
        listed = body.get("authors")
        keys = [
            entry["key"]
            for entry in (listed if isinstance(listed, list) else [])
            if isinstance(entry, dict) and isinstance(entry.get("key"), str)
        ]
        if not keys:
            return [by.strip()] if isinstance(by, str) and by.strip() else []
        result = self.http.get(f"https://openlibrary.org{keys[0]}.json")
        author = result.json_body() if result.ok else None
        name = author.get("name") if isinstance(author, dict) else None
        # The edition lists an author even when that lookup fails.
        return [name] if isinstance(name, str) and name.strip() else ["author on record"]


def _year(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    digits = [part for part in value.replace(",", " ").split() if part.isdigit() and len(part) == 4]
    return int(digits[-1]) if digits else None
