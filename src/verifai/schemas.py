"""Shared result types for every expert.

Verdicts describe what retrieved evidence showed. They are not a model's
recollection of a paper, article, or claim.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

EVIDENCE_PRINCIPLE = (
    "Verdicts aggregate evidence from multiple independent sources, record "
    "which sources were consulted and which ones contributed, and still come "
    "back when any single provider is unavailable. No provider is the sole "
    "decider. The Google Fact Check Tools API, if it is added later, is one "
    "more provider under this rule and must never be the only source a "
    "verdict depends on."
)

SOURCE_LABELS = {
    "doi.org": "doi.org",
    "crossref": "Crossref",
    "datacite": "DataCite",
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "url": "Direct fetch",
    "wayback": "Wayback Machine",
}

SOURCE_ORDER = list(SOURCE_LABELS)


class Verdict(str, Enum):
    SUPPORTED = "supported"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_MATCHES = "partially_matches"
    NOT_FOUND = "not_found"
    UNREACHABLE = "unreachable"


VERDICT_LABELS = {
    Verdict.SUPPORTED: "Supported",
    Verdict.MIXED: "Mixed",
    Verdict.UNSUPPORTED: "Unsupported",
    Verdict.UNVERIFIABLE: "Unverifiable",
}

STATUS_LABELS = {
    CitationStatus.VERIFIED: "Verified",
    CitationStatus.PARTIALLY_MATCHES: "Partially matches",
    CitationStatus.NOT_FOUND: "Not found",
    CitationStatus.UNREACHABLE: "Unreachable",
}


def source_label(name: str) -> str:
    return SOURCE_LABELS.get(name, name)


def ordered_sources(names: list[str]) -> list[str]:
    rank = {name: index for index, name in enumerate(SOURCE_ORDER)}
    unique = list(dict.fromkeys(names))
    return sorted(unique, key=lambda name: (rank.get(name, 100), name))


class EvidenceItem(BaseModel):
    """One record retrieved from one provider."""

    model_config = ConfigDict(extra="ignore")

    source: str
    title: str | None = None
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    snippet: str | None = None
    score: float | None = None


class Link(BaseModel):
    label: str
    url: str


class CitationResult(BaseModel):
    """Status of a single extracted citation."""

    raw: str
    kind: Literal["doi", "url", "reference"]
    status: CitationStatus
    confidence: float = Field(ge=0, le=1)
    explanation: str
    mismatches: list[str] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    sources_contributed: list[str] = Field(default_factory=list)
    sources_unavailable: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    matched_title: str | None = None
    matched_authors: str | None = None
    matched_year: int | None = None
    matched_venue: str | None = None
    matched_url: str | None = None

    @computed_field
    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]


class ExpertResult(BaseModel):
    """What one expert concluded from the evidence it retrieved.

    ``sources_consulted`` is every provider the expert called.
    ``sources_contributed`` answered with a definite result, including an
    authoritative not-found. ``sources_unavailable`` failed (timeout, 5xx,
    network) and were not treated as evidence either way.
    """

    expert: str
    verdict: Verdict
    verdict_scope: str
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    sources_contributed: list[str] = Field(default_factory=list)
    sources_unavailable: list[str] = Field(default_factory=list)
    citations: list[CitationResult] = Field(default_factory=list)
    evidence_principle: str = EVIDENCE_PRINCIPLE

    @computed_field
    @property
    def verdict_label(self) -> str:
        return VERDICT_LABELS[self.verdict]


class CheckRequest(BaseModel):
    text: str


class CheckResponse(BaseModel):
    results: list[ExpertResult]
