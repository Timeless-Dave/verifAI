"""Rule-based citation extraction.

Handles DOIs, http(s) URLs, APA-like references, and
``Surname et al., Year, Venue, Title`` lines. Bare ``(Smith et al., 2021)``
mentions are ignored: without a title they are not specific enough to search
without calling a real paper fabricated.

No model is required, and none is loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from verifai.experts.citation.textnorm import word_count

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>\"'\]\[]+", re.IGNORECASE)
_YEAR_PAREN_RE = re.compile(r"\(((?:19|20)\d{2})\)\.\s+")
_NARRATIVE_RE = re.compile(
    r"(?P<authors>[A-Z][\w'’\-]+(?:\s+et\s+al\.?)?)\s*,\s*"
    r"(?P<year>(?:19|20)\d{2})\s*,\s*"
    r"(?P<venue>[A-Z][^,\n]{1,120}?)\s*,\s*"
    r"(?P<title>[A-Za-z“\"'][^.\n]{11,400})"
)
_SURNAME_RE = re.compile(
    r"((?:(?:van|von|de|del|der|den|di|da|le|la)\s+){0,3}[A-Z][\w'’\-]+)\s*,\s*\S"
)
_ET_AL_RE = re.compile(r"\b([A-Z][\w'’\-]+)\s+et\s+al\.?", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(?:[-*•]+|\d+[.)]|\[\d+\])\s+")
_DOI_HOSTS = {"doi.org", "dx.doi.org", "www.doi.org"}

_STOP_AUTHORS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "the",
    "a",
    "an",
    "and",
    "but",
    "for",
    "from",
    "with",
    "this",
    "that",
    "these",
    "those",
    "however",
    "although",
    "because",
    "while",
    "after",
    "before",
    "during",
    "since",
    "until",
    "in",
    "on",
    "at",
    "by",
    "we",
    "they",
    "he",
    "she",
    "it",
    "see",
    "also",
    "note",
    "figure",
    "table",
    "chapter",
    "section",
    "page",
    "vol",
    "volume",
    "issue",
    "pp",
    "ed",
    "retrieved",
    "available",
    "http",
    "https",
    "doi",
}


@dataclass
class ExtractedCitation:
    kind: str
    raw: str
    start: int
    end: int
    doi: str | None = None
    url: str | None = None
    title: str | None = None
    surnames: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    et_al: bool = False


def extract_citations(text: str) -> list[ExtractedCitation]:
    if not text or not text.strip():
        return []
    normalized = _normalize(text)
    found: list[ExtractedCitation] = []

    for match in _DOI_RE.finditer(normalized):
        doi = clean_doi(match.group(1))
        if not _doi_ok(doi):
            continue
        line_start, line_end = _line_bounds(normalized, match.start(), match.end())
        raw_line = normalized[line_start:line_end].strip()
        meta = _parse_bibliographic(raw_line)
        found.append(
            ExtractedCitation(
                kind="doi",
                raw=_clip(raw_line or doi),
                start=line_start,
                end=line_end,
                doi=doi,
                title=meta.get("title"),
                surnames=list(meta.get("surnames") or []),
                year=meta.get("year"),
                venue=meta.get("venue"),
                et_al=bool(meta.get("et_al")),
            )
        )

    occupied = [(item.start, item.end) for item in found]
    for reference in _find_references(normalized):
        if _overlaps(reference.start, reference.end, occupied):
            continue
        found.append(reference)
        occupied.append((reference.start, reference.end))

    for match in _URL_RE.finditer(normalized):
        url = clean_url(match.group(0))
        if not url or _is_doi_url(url):
            continue
        if _overlaps(match.start(), match.end(), occupied):
            continue
        line_start, line_end = _line_bounds(normalized, match.start(), match.end())
        if _overlaps(line_start, line_end, occupied) or _overlaps(
            match.start(), match.end(), occupied
        ):
            continue
        # A URL sitting on a line we already accepted as a reference or DOI
        # belongs to that citation. Compare against line spans we stored.
        if any(line_start >= start and line_end <= end for start, end in occupied):
            continue
        found.append(
            ExtractedCitation(kind="url", raw=url, start=match.start(), end=match.end(), url=url)
        )
        occupied.append((match.start(), match.end()))

    found.sort(key=lambda item: (item.start, item.end))
    return _dedupe(found)


def clean_doi(raw: str) -> str:
    doi = raw.strip()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.IGNORECASE)
    while doi and doi[-1] in ".,;:":
        doi = doi[:-1]
    if doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    if doi.endswith("]") and doi.count("[") < doi.count("]"):
        doi = doi[:-1]
    return doi


def clean_url(raw: str) -> str:
    url = raw.strip()
    while url and url[-1] in ".,;:":
        url = url[:-1]
    if url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    if url.endswith("]") and url.count("[") < url.count("]"):
        url = url[:-1]
    return url


def _find_references(text: str) -> list[ExtractedCitation]:
    found: list[ExtractedCitation] = []
    for match in _YEAR_PAREN_RE.finditer(text):
        window_start = max(0, match.start() - 500)
        prefix = text[window_start : match.start()]
        head = _author_segment(prefix)
        if not head:
            continue
        surnames, et_al = _surnames(head)
        if not surnames and not et_al:
            continue
        title, venue = _split_title_venue(text[match.end() :])
        if not title:
            continue
        head_index = prefix.rfind(head)
        start = window_start + head_index if head_index >= 0 else match.start()
        end = min(len(text), match.end() + len(title) + 2)
        year = int(match.group(1))
        raw = _clip(
            " ".join(part for part in (head, f"({year}).", title + ".", venue or "") if part)
        )
        found.append(
            ExtractedCitation(
                kind="reference",
                raw=raw,
                start=start,
                end=end,
                title=title,
                surnames=surnames,
                year=year,
                venue=venue,
                et_al=et_al,
            )
        )

    for match in _NARRATIVE_RE.finditer(text):
        if not _narrative_ok(match):
            continue
        if _overlaps(match.start(), match.end(), [(item.start, item.end) for item in found]):
            continue
        surnames, et_al = _surnames(match.group("authors"))
        if not surnames:
            continue
        title = match.group("title").strip(" .“”\"'")
        venue = match.group("venue").strip(" .")
        found.append(
            ExtractedCitation(
                kind="reference",
                raw=_clip(match.group(0).strip(" .")),
                start=match.start(),
                end=match.end(),
                title=title,
                surnames=surnames,
                year=int(match.group("year")),
                venue=venue or None,
                et_al=et_al or True,
            )
        )
    found.sort(key=lambda item: (item.start, -(item.end - item.start)))
    return _drop_overlaps(found)


def _parse_bibliographic(line: str) -> dict:
    """Best-effort metadata for a line that already has a DOI or URL."""
    cleaned = _URL_RE.sub(" ", line)
    cleaned = _DOI_RE.sub(" ", cleaned)
    cleaned = _BULLET_RE.sub("", cleaned.strip())
    match = _YEAR_PAREN_RE.search(cleaned)
    if match:
        head = _author_segment(cleaned[: match.start()])
        surnames, et_al = _surnames(head)
        title, venue = _split_title_venue(cleaned[match.end() :])
        if title and (surnames or et_al):
            return {
                "title": title,
                "venue": venue,
                "year": int(match.group(1)),
                "surnames": surnames,
                "et_al": et_al,
            }
    narrative = _NARRATIVE_RE.search(cleaned)
    if narrative and _narrative_ok(narrative):
        surnames, et_al = _surnames(narrative.group("authors"))
        title = narrative.group("title").strip(" .“”\"'")
        if surnames and title:
            return {
                "title": title,
                "venue": narrative.group("venue").strip(" .") or None,
                "year": int(narrative.group("year")),
                "surnames": surnames,
                "et_al": True,
            }
    return {}


def _author_segment(prefix: str) -> str:
    """Last sentence of the last paragraph, without splitting on initials such as ``A.``."""
    text = prefix.strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        text = paragraphs[-1]
    # Split on sentence endings, but not on initials ("A. N." or "J.").
    # The two-character lookbehind sees the letter before the period.
    parts = [
        part.strip() for part in re.split(r"(?<![A-Z][.?!])(?<=[.?!])\s+", text) if part.strip()
    ]
    if not parts:
        return ""
    segment = parts[-1]
    if segment.casefold().rstrip(".") == "et al" and len(parts) >= 2:
        segment = parts[-2] + " et al."
    segment = _BULLET_RE.sub("", segment).strip(" \t:—–-")
    surname = _SURNAME_RE.search(segment) or _ET_AL_RE.search(segment)
    if surname and surname.start() > 0:
        segment = segment[surname.start() :].strip()
    return segment


def _surnames(text: str) -> tuple[list[str], bool]:
    et_al = bool(re.search(r"\bet\s+al\.?", text, flags=re.IGNORECASE))
    names: list[str] = []
    for match in _SURNAME_RE.finditer(text):
        surname = re.sub(r"\s+", " ", match.group(1)).strip()
        if surname.casefold() in _STOP_AUTHORS:
            continue
        if surname not in names:
            names.append(surname)
    if not names:
        for match in _ET_AL_RE.finditer(text):
            surname = match.group(1)
            if surname.casefold() in _STOP_AUTHORS:
                continue
            if surname not in names:
                names.append(surname)
            et_al = True
    return names, et_al


def _split_title_venue(rest: str) -> tuple[str | None, str | None]:
    rest = rest.lstrip()
    quoted = re.match(r"[“\"]([^”\"\n]{10,300})[”\"]\.?\s*(.*)$", rest)
    if quoted:
        title = quoted.group(1).strip()
        tail = quoted.group(2)
    else:
        matched = re.match(r"([^\n]{12,}?)([.?!])(?:\s+|$)", rest)
        if not matched:
            return None, None
        title = matched.group(1).strip(" \"“”'")
        tail = rest[matched.end() :]
    if word_count(title) < 3 or len(title) < 12:
        return None, None
    tail = tail.split("\n", 1)[0]
    tail = _URL_RE.sub(" ", tail)
    tail = _DOI_RE.sub(" ", tail)
    venue = re.split(r",\s*\d|\.\s+", tail, maxsplit=1)[0]
    venue = venue.strip(" .;,")
    if not venue or len(venue) < 2 or venue.casefold() in _STOP_AUTHORS:
        venue = None
    return title, venue


def _narrative_ok(match: re.Match[str]) -> bool:
    author = match.group("authors").split()[0]
    if author.casefold() in _STOP_AUTHORS:
        return False
    title = match.group("title").strip(" .“”\"'")
    if word_count(title) < 4:
        return False
    venue = match.group("venue").strip()
    return bool(venue) and venue.split()[0].casefold() not in _STOP_AUTHORS


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    cleaned: list[str] = []
    for part in parts:
        joined = re.sub(r"[ \t]*\n[ \t]*", " ", part).strip()
        if joined:
            cleaned.append(joined)
    return "\n\n".join(cleaned)


def _dedupe(items: list[ExtractedCitation]) -> list[ExtractedCitation]:
    seen: set[tuple] = set()
    kept: list[ExtractedCitation] = []
    for item in items:
        if item.kind == "doi" and item.doi:
            key = ("doi", item.doi.lower())
        elif item.kind == "url" and item.url:
            key = ("url", item.url.lower().rstrip("/"))
        else:
            folded = re.sub(r"\W+", " ", (item.title or "").casefold()).strip()
            key = ("ref", folded, item.year or 0)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _drop_overlaps(items: list[ExtractedCitation]) -> list[ExtractedCitation]:
    kept: list[ExtractedCitation] = []
    for item in items:
        if _overlaps(item.start, item.end, [(other.start, other.end) for other in kept]):
            continue
        kept.append(item)
    return kept


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in spans)


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def _doi_ok(doi: str) -> bool:
    if "/" not in doi:
        return False
    prefix, suffix = doi.split("/", 1)
    return bool(re.fullmatch(r"10\.\d{4,9}", prefix, flags=re.IGNORECASE)) and len(suffix) >= 3


def _is_doi_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host.lower() in _DOI_HOSTS


def _clip(value: str, limit: int = 500) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
