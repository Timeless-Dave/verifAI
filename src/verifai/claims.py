"""Split a paste into checkable claims without a model."""

from __future__ import annotations

import re

from verifai.retrieval.textutil import extract_urls

_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REFERENCE = re.compile(
    r"\b10\.\d{4,9}/|\bet\s+al\.?|\(\s*(?:19|20)\d{2}\s*\)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+")


def extract_claims(text: str, *, limit: int = 5) -> list[str]:
    """Return declarative sentences that are specific enough to check."""
    claims: list[str] = []
    for chunk in re.split(r"\n+", text or ""):
        for sentence in _SPLIT.split(chunk.strip()):
            cleaned = " ".join(sentence.split())
            if _keep(cleaned):
                claims.append(cleaned)
            if len(claims) >= limit:
                return claims
    return claims


def _keep(sentence: str) -> bool:
    if len(sentence) < 30 or len(sentence) > 500 or sentence.endswith("?"):
        return False
    if _REFERENCE.search(sentence) and re.search(r"\b(?:19|20)\d{2}\b", sentence):
        return False
    return not (extract_urls(sentence) and len(_URL.sub("", sentence).strip(" .")) < 25)
