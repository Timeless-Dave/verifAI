"""Passages retrieved for a claim. Providers do not choose the verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
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
    # The statement a fact-checker rated, when it is not the claim itself.
    reviewed_claim: str | None = None
    # The name in the claim this passage was looked up by: the item or
    # article for "Frogs" is about frogs even when it is titled "Anura".
    about: str | None = None
    # When a news report or official statement was published.
    published: date | None = None


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
