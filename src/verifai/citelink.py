"""Which source does a sentence cite?

Academic prose names its sources in the sentence: "(Northey et al., 2018)",
"[3]", "Vaswani et al. (2017) showed", or a DOI or link. Each marker is
matched to an entry of the paste's reference list, so the claim can be read
against the work it leans on.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from verifai.schemas import CitationResult, CitationStatus
from verifai.source_reader import SourceText

_NUMBERED = re.compile(r"\[\s*(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\s*\]")
_PAREN = re.compile(r"\(([^()]*?(?:19|20)\d{2}[a-z]?[^()]*)\)")
_AUTHOR_YEAR = re.compile(
    r"(?P<name>[A-Z][\w'’-]+(?:\s+(?:de|van|von|der|da|del|di|le|la)\s+[A-Z][\w'’-]+)?)"
    r"(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’-]+)?,?\s*(?P<year>(?:19|20)\d{2})[a-z]?"
)
_NARRATIVE = re.compile(
    r"(?P<name>[A-Z][\w'’-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’-]+)?\s*"
    r"\(\s*(?P<year>(?:19|20)\d{2})[a-z]?(?:,\s*p+\.\s*[\d–-]+)?\s*\)"
)
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_ARXIV = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_URL = re.compile(r"https?://[^\s<>\"]+")

# Words that look like a surname in a parenthesis but are not one.
_NOT_NAMES = frozenset({"see", "cf", "eg", "ie", "and", "in", "the", "table", "figure", "fig"})

# A citation worth reading: it was found, even if something printed beside it is off.
READABLE = frozenset(
    {
        CitationStatus.VERIFIED,
        CitationStatus.PARTIALLY_MATCHES,
        CitationStatus.WRONG_WORK,
        CitationStatus.RETRACTED,
    }
)


@dataclass(frozen=True)
class Marker:
    """One in-text citation: a number, an author and year, or an identifier."""

    text: str
    number: int | None = None
    surname: str | None = None
    year: int | None = None
    identifier: str | None = None


@dataclass(frozen=True)
class Link:
    citation: CitationResult
    # How the sentence cited it: "(Northey et al., 2018)", or None when the
    # paste holds this one source and the sentence names none.
    cited_as: str | None
    # The work's text when the user attached it, so it need not be fetched.
    document: SourceText | None = None


def markers_in(sentence: str) -> list[Marker]:
    found: list[Marker] = []
    for match in _NUMBERED.finditer(sentence):
        found.extend(
            Marker(text=match.group(0), number=number) for number in _numbers(match.group(1))
        )
    found.extend(
        Marker(text=match.group(0), surname=match.group("name"), year=int(match.group("year")))
        for match in _NARRATIVE.finditer(sentence)
        if match.group("name").casefold() not in _NOT_NAMES
    )
    for match in _PAREN.finditer(sentence):
        for part in match.group(1).split(";"):
            inner = _AUTHOR_YEAR.search(part)
            if inner and inner.group("name").casefold() not in _NOT_NAMES:
                found.append(
                    Marker(
                        text=f"({part.strip()})",
                        surname=inner.group("name"),
                        year=int(inner.group("year")),
                    )
                )
    for pattern in (_DOI, _URL, _ARXIV):
        for match in pattern.finditer(sentence):
            value = match.group(1) if pattern is _ARXIV else match.group(0)
            found.append(Marker(text=match.group(0), identifier=value.rstrip(".,;)")))
    unique: dict[tuple[object, ...], Marker] = {}
    for marker in found:
        unique.setdefault(
            (marker.number, _fold(marker.surname), marker.year, marker.identifier), marker
        )
    return list(unique.values())


def link_sources(
    written: str,
    citations: list[CitationResult],
    references: dict[str, str],
    *,
    single_claim: bool,
) -> list[Link]:
    """The citations a sentence leans on, in the order it names them.

    With no marker in the sentence, a paste with exactly one source (a claim
    and the paper it came from) links that source. A single claim pasted with
    a few sources is read against each of them.
    """
    usable = [item for item in citations if item.kind != "isbn"]
    links: list[Link] = []
    seen: set[int] = set()

    def add(citation: CitationResult, cited_as: str | None) -> None:
        if id(citation) not in seen:
            seen.add(id(citation))
            links.append(Link(citation, cited_as))

    for marker in markers_in(written):
        for citation in usable:
            if _matches(marker, citation, references):
                add(citation, marker.text)
                break
    if links or not usable:
        return links
    if len(usable) == 1:
        add(usable[0], None)
    elif single_claim and len(usable) <= 3:
        for citation in usable:
            add(citation, None)
    return links


def _matches(marker: Marker, citation: CitationResult, references: dict[str, str]) -> bool:
    raw = _fold(citation.raw)
    if marker.number is not None:
        entry = references.get(str(marker.number))
        if not entry:
            return False
        folded = _fold(entry)
        return raw in folded or folded in raw
    if marker.identifier is not None:
        identifier = _fold(marker.identifier)
        return identifier in raw or identifier in _fold(citation.matched_url or "")
    if marker.surname and marker.year:
        surname = _fold(marker.surname)
        names = f"{raw} {_fold(citation.matched_authors or '')}"
        year_ok = str(marker.year) in citation.raw or citation.matched_year == marker.year
        return year_ok and re.search(rf"\b{re.escape(surname)}\b", names) is not None
    return False


def _numbers(spec: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", spec):
        bounds = re.split(r"\s*[–-]\s*", part)
        if len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
            start, end = int(bounds[0]), int(bounds[1])
            if 0 < end - start <= 20:
                numbers.extend(range(start, end + 1))
                continue
        if part.strip().isdigit():
            numbers.append(int(part))
    return numbers


def _fold(text: str | None) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())
