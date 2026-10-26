"""Independent evidence providers.

A checker calls every provider that is relevant to the question, records
which ones answered, and aggregates those answers. Providers do not see
each other's responses and do not emit the final verdict.

The Google Fact Check Tools API is a possible future provider. It must be
registered beside other providers and must never be the sole decider.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verifai.schemas import EVIDENCE_PRINCIPLE


@runtime_checkable
class EvidenceProvider(Protocol):
    """One independent source of evidence.

    Implementations are interchangeable. Callers must keep working when
    ``unavailable`` is the only thing a provider can say, and must not
    treat any single provider — including a future Google Fact Check
    provider — as sufficient on its own when another configured provider
    could have been asked.
    """

    name: str


class EvidenceRegistry:
    """Named providers plugged into a checker. Registration order is kept."""

    def __init__(self) -> None:
        self._providers: dict[str, EvidenceProvider] = {}

    def register(self, provider: EvidenceProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"provider {provider.name!r} is already registered")
        self._providers[provider.name] = provider

    def get(self, name: str) -> EvidenceProvider:
        return self._providers[name]

    def names(self) -> list[str]:
        return list(self._providers)

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    def __iter__(self):
        return iter(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)


__all__ = ["EVIDENCE_PRINCIPLE", "EvidenceProvider", "EvidenceRegistry"]
