"""Records returned by bibliographic providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Record:
    source: str
    title: str | None
    authors: list[str]
    surnames: list[str]
    year: int | None
    years: list[int]
    venue: str | None
    doi: str | None
    url: str | None
    record_id: str | None = None


@dataclass
class ProviderOutcome:
    """A definite answer, or an availability failure.

    An empty ``records`` list with ``unavailable`` false is a negative
    finding (the provider searched and has nothing). That still counts as
    a contribution. A timeout or 5xx is not evidence that the work is fake.
    """

    source: str
    unavailable: bool
    records: list[Record] = field(default_factory=list)
    error: str | None = None

    @property
    def contributed(self) -> bool:
        return not self.unavailable


@dataclass
class UrlCheck:
    status_code: int | None
    final_url: str | None
    error: str | None
    blocked_reason: str | None = None


@dataclass
class WaybackHit:
    unavailable: bool
    snapshot_url: str | None
    timestamp: str | None
    error: str | None = None
