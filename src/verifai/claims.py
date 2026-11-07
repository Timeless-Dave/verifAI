"""Split a paste into checkable claims without a model."""

from __future__ import annotations

import re
from collections.abc import Callable

from verifai.experts.citation.extract import ExtractedCitation, find_references
from verifai.retrieval.textutil import extract_urls

_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REFERENCE = re.compile(
    r"\b10\.\d{4,9}/|\bet\s+al\.?|\(\s*(?:19|20)\d{2}\s*\)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+")
_DOI = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)
_BULLET = re.compile(r"^(?:[-*•]+|\d+[.)]|\[\d+\])\s+")
_WORD = re.compile(r"[A-Za-z][\w'’\-]*|\d+")
_MINOR = {"a", "an", "and", "of", "the", "in", "on", "for", "to", "at", "by", "with", "&"}


def extract_claims(text: str, *, limit: int = 5) -> list[str]:
    """Return declarative sentences that are specific enough to check.

    Reference-list entries are removed first. They belong to the citation
    expert, and splitting one at its periods leaves author and venue
    fragments that read like sentences.
    """
    claims: list[str] = []
    for chunk in re.split(r"\n+", without_references(text or "")):
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
    return not only_links(sentence)


def only_links(text: str) -> bool:
    """True when ``text`` is a DOI or link with too little prose to check."""
    if not (extract_urls(text) or _DOI.search(text) or _URL.search(text)):
        return False
    return len(_DOI.sub("", _URL.sub("", text)).strip(" .,;:")) < 25


def without_references(text: str) -> str:
    """Blank out lines that are reference entries, keeping prose lines.

    A line is a reference when a structured entry (``Author (Year). Title.``
    or ``Author et al., Year, Venue, Title``) starts it, alone or wrapped onto
    the next two lines. Title-case lines right after an entry, such as the
    venue and volume, belong to it too.
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    drop = [False] * len(lines)
    index = 0
    while index < len(lines):
        covered = _reference_at(lines, index, find_references)
        if not covered:
            index += 1
            continue
        end = index + covered
        while end < len(lines) and _continues_reference(lines[end]):
            end += 1
        for line in range(index, end):
            drop[line] = True
        index = end
    return "\n".join("" if dropped else line for line, dropped in zip(lines, drop, strict=True))


def _reference_at(
    lines: list[str], index: int, find: Callable[[str], list[ExtractedCitation]]
) -> int:
    """How many lines from ``index`` a reference that starts there spans, or 0."""
    head = _BULLET.sub("", lines[index].strip())
    if not head:
        return 0
    for count in (1, 2, 3):
        parts = [head, *(line.strip() for line in lines[index + 1 : index + count])]
        if count > 1 and (len(parts) < count or not all(parts)):
            return 0
        joined = " ".join(parts)
        if any(item.start == 0 for item in find(joined)):
            return count
    return 0


def _continues_reference(line: str) -> bool:
    """Venue, volume, pages, or a link left over after a wrapped reference."""
    stripped = line.strip()
    if not stripped or _BULLET.match(stripped):
        return False
    rest = _DOI.sub("", _URL.sub("", stripped)).strip(" .,;:")
    if not rest:
        return True
    words = [word for word in _WORD.findall(rest) if word.casefold() not in _MINOR]
    if not words:
        return True
    capped = sum(1 for word in words if word[0].isupper() or word[0].isdigit())
    return capped / len(words) >= 0.8
