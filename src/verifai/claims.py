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

# A period after these does not end a sentence.
_ABBREVIATIONS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "al",
    "vol",
    "no",
    "nos",
    "pp",
    "p",
    "fig",
    "figs",
    "eq",
    "jr",
    "sr",
    "inc",
    "ltd",
    "co",
    "approx",
    "ca",
    "cf",
    "ed",
    "eds",
    "ch",
    "sec",
    "dept",
    "univ",
    "u.s",
    "u.k",
    "e.u",
    "a.m",
    "p.m",
}
_INITIAL_END = re.compile(r"(?:^|[\s(])[A-Z]\.$")
# Anything that marks a line as carrying citations.
_CITATION_MARK = re.compile(
    r"\b10\.\d{4,9}/|https?://|\barXiv\b|\bISBN\b|\bPMID\b|\bet\s+al\b|"
    r"\(\s*(?:19|20)\d{2}\s*\)",
    re.IGNORECASE,
)
# Bibliographic fragments: identifiers, author lists, volume and page runs.
_BIBLIOGRAPHIC = re.compile(
    r"\b(?:doi|arxiv|isbn|pmid)\b|\bProc\.|\bProceedings of\b|=\s*\{|"
    r"\b\d+\s*\(\d+\)\s*[:,]|\b(?:19|20)\d{2};\s*\d+",
    re.IGNORECASE,
)
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-z'’\-]+,\s+(?:[A-Z]\.\s*){1,3}")
_VANCOUVER_AUTHOR = re.compile(r"\b[A-Z][a-z'’\-]+\s+[A-Z]{1,3},")
_FINITE_VERB = re.compile(
    r"\b(?:is|are|was|were|be|been|has|have|had|does|did|do|can|could|will|would|may|"
    r"might|must|shall|should|said|says|causes?|shows?|found|won|lies|holds|contains?)\b|"
    r"\b[a-z]{3,}ed\b",
    re.IGNORECASE,
)
# Text that talks to the checker, not about the world.
_INSTRUCTION = re.compile(
    r"\b(?:ignore|disregard)\s+(?:all\s+|any\s+)?(?:prior|previous|above|earlier)\b|"
    r"\bsystem\s+(?:note|prompt|message)\b|\bmark\s+(?:every|all|each)\b|"
    r"\byou\s+are\s+now\b|\bas\s+an\s+ai\b",
    re.IGNORECASE,
)
_META = re.compile(
    r"^(?:here'?s|here\s+is|here\s+are|below\s+(?:is|are)|the\s+following|in\s+summary|"
    r"to\s+summari[sz]e|sources?:|references?:|books?:|see\s+also\b)",
    re.IGNORECASE,
)
_OPINION = re.compile(
    r"\b(?:i|we)\s+(?:think|believe|feel)\b|\b(?:feel|feels|felt)\b|"
    r"\bin\s+my\s+(?:view|opinion)\b|\barguably\b|\b(?:elegant|beautiful|overrated|underrated)\b",
    re.IGNORECASE,
)
_YEAR_ONLY = re.compile(r"^\s*(?:19|20)\d{2}[a-z]?\s*$")


def extract_claims(text: str, *, limit: int = 5) -> list[str]:
    """Return up to ``limit`` declarative sentences that are specific enough to check."""
    return find_claims(text)[:limit]


def find_claims(text: str) -> list[str]:
    """Every checkable sentence, in order and without repeats.

    Reference entries and BibTeX blocks are removed first. They belong to
    the citation expert, and splitting one at its periods leaves author and
    venue fragments that read like sentences. Citation-only parentheses are
    dropped from prose, so "X changed Y (Smith, J. et al. …)" keeps "X
    changed Y". Opinions, notes to the reader, and instructions aimed at the
    checker are not claims.
    """
    claims: list[str] = []
    seen: set[str] = set()
    body = _without_bibtex(without_references(text or ""))
    for chunk in re.split(r"\n+", body):
        citation_line = bool(_CITATION_MARK.search(chunk) or find_references(chunk))
        for sentence in split_sentences(_without_citation_parens(chunk)):
            cleaned = " ".join(sentence.split()).replace(" ,", ",").replace(" .", ".")
            folded = cleaned.casefold()
            if folded in seen or not _keep(cleaned, citation_line=citation_line):
                continue
            seen.add(folded)
            claims.append(cleaned)
    return claims


def split_sentences(text: str) -> list[str]:
    """Split at sentence ends, but not after an initial or a common abbreviation."""
    pieces = [piece for piece in _SPLIT.split(text.strip()) if piece]
    merged: list[str] = []
    for piece in pieces:
        if merged and _continues(merged[-1], piece):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _continues(previous: str, piece: str) -> bool:
    if piece[:1].islower() or piece[:1].isdigit():
        return True
    if _INITIAL_END.search(previous):
        return True
    last = previous.rsplit(None, 1)[-1].lstrip("(").casefold().rstrip(".")
    return last in _ABBREVIATIONS


def _without_citation_parens(line: str) -> str:
    """Drop "(…)" groups that only hold a citation or a year, innermost first."""
    kept: list[str] = []

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if (
            _YEAR_ONLY.match(inner)
            or _CITATION_MARK.search(inner)
            or _AUTHOR_INITIALS.search(inner)
        ):
            return ""
        kept.append(match.group(0))
        return f"\x00{len(kept) - 1}\x01"

    previous = None
    while previous != line:
        previous = line
        line = re.sub(r"\(([^()]*)\)", replace, line)
    return re.sub(r"\x00(\d+)\x01", lambda found: kept[int(found.group(1))], line)


def _without_bibtex(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    depth = 0
    for line in lines:
        if depth == 0 and re.match(r"\s*@\w+\s*\{", line):
            depth = line.count("{") - line.count("}")
            out.append("")
            continue
        if depth > 0:
            depth += line.count("{") - line.count("}")
            out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def _keep(sentence: str, *, citation_line: bool = False) -> bool:
    if len(sentence) < 30 or len(sentence) > 500 or sentence.endswith("?"):
        return False
    if _REFERENCE.search(sentence) and re.search(r"\b(?:19|20)\d{2}\b", sentence):
        return False
    if only_links(sentence) or _INSTRUCTION.search(sentence) or _META.search(sentence):
        return False
    if _OPINION.search(sentence) or _BIBLIOGRAPHIC.search(sentence):
        return False
    if (
        len(_AUTHOR_INITIALS.findall(sentence)) >= 2
        or len(_VANCOUVER_AUTHOR.findall(sentence)) >= 2
    ):
        return False
    if _title_case(sentence):
        return False
    # On a line full of citations, a phrase with no verb is a title or a venue.
    return not (citation_line and not _FINITE_VERB.search(sentence))


def _title_case(sentence: str) -> bool:
    words = [word for word in _WORD.findall(sentence) if word.casefold() not in _MINOR]
    if len(words) < 4:
        return False
    capped = sum(1 for word in words if word[0].isupper() or word[0].isdigit())
    return capped / len(words) >= 0.8


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
