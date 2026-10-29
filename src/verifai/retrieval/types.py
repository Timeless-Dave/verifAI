"""Passages retrieved for a claim. Providers do not choose the verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from verifai.corroboration import Stance


@dataclass(frozen=True)
class Passage:
    source: str
    family: str
    title: str
    url: str | None
    text: str
    document_ids: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    speaker: str | None = None
    publication_type: str | None = None
    native_stance: Stance | None = None
    native_confidence: float | None = None
    rating: str | None = None


@dataclass
class RetrievalResult:
    source: str
    passages: list[Passage] = field(default_factory=list)
    unavailable: bool = False
    error: str | None = None


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str) -> RetrievalResult:
        """Search this source. An error is unavailable, not a refutation."""
