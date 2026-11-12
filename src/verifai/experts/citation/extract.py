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
from typing import Any
from urllib.parse import unquote, urlparse

from verifai.experts.citation.textnorm import word_count

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:%A-Za-z0-9]+)", re.IGNORECASE)
# New-style arXiv ids. arXiv registers each one with DataCite as 10.48550/arXiv.<id>.
_ARXIV_RE = re.compile(
    r"(?:\barXiv:\s*|\barxiv\.org/(?:abs|pdf)/)"
    r"(\d{4}\.\d{4,5}|[a-z][a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
# A DOI that runs into a non-ASCII character, such as a Cyrillic "а" that
# looks like a Latin "a". Cut at that character it would look up the wrong DOI.
_DOI_LOOKALIKE = re.compile(r"\b10\.\d{4,9}/[^\s<>\"';]*?[^\x00-\x7f][^\s<>\"';]*")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_MD_LINK = re.compile(r"\[([^\]\n]{3,200})\]\((https?://[^\s()]+(?:\([^\s()]*\)[^\s()]*)*)\)")
_PMC_RE = re.compile(r"\bPMC\s?(\d{3,9})\b")
_AUTHOR_YEAR = re.compile(
    r"\(([A-Z][\w'’\-]+(?:\s+et\s+al\.?)?(?:\s+(?:and|&)\s+[A-Z][\w'’\-]+)?),?\s+((?:19|20)\d{2}[a-z]?)\)"
)
_IBID = re.compile(r"\b(?:ibid|op\.\s*cit|loc\.\s*cit)\.?", re.IGNORECASE)
_VAGUE_LABEL = re.compile(
    r"\b(?:paper|here|link|this|pdf|article|source|website|page)\b", re.IGNORECASE
)
_PMID_RE = re.compile(r"\bPMID:?\s*(\d+)\b")
_ISBN_RE = re.compile(r"\bISBN(?:-1[03])?:?\s*((?:97[89][-\s]?)?(?:\d[-\s]?){9}[\dXx])\b")
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
# Vancouver style: "Wakefield AJ, Murch SH" or "Wakefield AJ et al." The
# capitals after the surname are initials, never a name.
_VANCOUVER_NAME = re.compile(r"\b([A-Z][a-z][\w'’\-]*)\s+([A-Z]{1,3})\b(?=\s*(?:,|\.|et\s+al))")
_INITIALS = re.compile(r"^[A-Z]{1,3}$")
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
    isbn: str | None = None
    pmid: str | None = None
    pmc: str | None = None
    occurrences: int = 1
    # Why the identifier cannot be real, decided before any lookup.
    problem: str | None = None
    # The words on the identifier's line, for an ISBN: what the text says the book is.
    context: str | None = None


def extract_citations(text: str) -> list[ExtractedCitation]:
    if not text or not text.strip():
        return []
    text = prepare_text(text)
    normalized = _normalize(text)
    found: list[ExtractedCitation] = []

    lookalikes = list(_DOI_LOOKALIKE.finditer(normalized))
    for match in lookalikes:
        token = clean_doi(match.group(0))
        odd = next(char for char in token if ord(char) > 127)
        found.append(
            ExtractedCitation(
                kind="doi",
                raw=_clip(_original_line_at(text, normalized, match) or match.group(0)),
                start=match.start(),
                end=match.end(),
                doi=token,
                problem=(
                    f"The DOI {token} contains “{odd}” (U+{ord(odd):04X}), a non-ASCII "
                    "character that looks like a Latin letter. DOIs are written in ASCII, so "
                    "this one was altered, by accident or to pass as a real DOI."
                ),
            )
        )
    lookalike_spans = [(match.start(), match.end()) for match in lookalikes]
    links = {match.group(2): match.group(1) for match in _MD_LINK.finditer(normalized)}

    dois = [
        (match, clean_doi(match.group(1)))
        for match in _DOI_RE.finditer(normalized)
        if not _overlaps(match.start(), match.end(), lookalike_spans)
    ]
    dois += [
        (match, f"10.48550/arXiv.{match.group(1)}") for match in _ARXIV_RE.finditer(normalized)
    ]
    dois.sort(key=lambda pair: pair[0].start())
    for match, doi in dois:
        if not _doi_ok(doi):
            continue
        line_start, line_end = _line_bounds(normalized, match.start(), match.end())
        raw_line = normalized[line_start:line_end].strip()
        # Consecutive lines are joined so a wrapped reference reads as one.
        # A list of references with one per line must not be read as one, or
        # every DOI takes the first line's title and year.
        crowded = sum(1 for other, _ in dois if line_start <= other.start() < line_end) > 1
        if crowded:
            raw_line = _original_line_at(text, normalized, match) or raw_line
        meta = _parse_bibliographic(raw_line)
        raw = match.group(0) if doi.startswith("10.48550/arXiv.") else (raw_line or doi)
        title = meta.get("title")
        label = _link_label(links, match.group(0))
        if label:
            # The link text names a work. Compare it with the work the link
            # really points to: "Attention Is All You Need" pointing at BERT.
            title, raw = label, f"[{label}]({_link_target(links, match.group(0))})"
        found.append(
            ExtractedCitation(
                kind="doi",
                raw=_clip(raw),
                start=line_start,
                end=line_end,
                doi=doi,
                title=title,
                surnames=list(meta.get("surnames") or []),
                year=meta.get("year"),
                venue=meta.get("venue"),
                et_al=bool(meta.get("et_al")),
                problem=_arxiv_problem(doi),
            )
        )

    # A line that holds a DOI or a reference owns the URLs on it. URLs do not
    # own their line, so two links in one sentence are both checked.
    occupied = [(item.start, item.end) for item in found]
    for reference in _find_references(normalized):
        if _overlaps(reference.start, reference.end, occupied):
            continue
        found.append(reference)
        occupied.append((reference.start, reference.end))

    for match in [*_PMID_RE.finditer(normalized), *_PMC_RE.finditer(normalized)]:
        # Judge ownership by the line as pasted. A reference list with no
        # blank lines is joined into one paragraph, and a DOI elsewhere in it
        # must not swallow this identifier.
        line = _original_line_at(text, normalized, match) or ""
        if _DOI_RE.search(line) or _ARXIV_RE.search(line):
            continue  # the DOI on this line already identifies the work
        meta = _parse_bibliographic(line)
        is_pmc = match.re is _PMC_RE
        number = match.group(1)
        problem = None
        if not is_pmc and not 0 < int(number) < 100_000_000:
            problem = f"PMID {number} is not a valid PubMed id. PubMed ids are 1 to 8 digits."
        found.append(
            ExtractedCitation(
                kind="pmc" if is_pmc else "pmid",
                raw=_clip(line.strip() or match.group(0)),
                start=match.start(),
                end=match.end(),
                pmid=None if is_pmc else number,
                pmc=f"PMC{number}" if is_pmc else None,
                title=meta.get("title"),
                year=meta.get("year"),
                venue=meta.get("venue"),
                surnames=list(meta.get("surnames") or []),
                problem=problem,
            )
        )

    for match in _ISBN_RE.finditer(normalized):
        isbn = re.sub(r"[-\s]", "", match.group(1)).upper()
        line_start = normalized.rfind("\n", 0, match.start()) + 1
        line_end = normalized.find("\n", match.end())
        line = normalized[line_start : line_end if line_end != -1 else len(normalized)]
        context = " ".join(line.replace(match.group(0), " ").split()).strip(" .,;:()[]-")
        found.append(
            ExtractedCitation(
                kind="isbn",
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
                isbn=isbn,
                context=context or None,
            )
        )

    url_spans: list[tuple[int, int]] = []
    for match in _URL_RE.finditer(normalized):
        url = clean_url(match.group(0))
        if not url or _is_doi_url(url) or _ARXIV_RE.search(url):
            continue
        if _overlaps(match.start(), match.end(), url_spans):
            continue
        # A URL on the same pasted line as a DOI or a structured reference
        # belongs to that citation. Other URLs are checked on their own.
        line = _original_line_at(text, normalized, match) or ""
        if _DOI_RE.search(line) or _find_references(line):
            continue
        found.append(
            ExtractedCitation(kind="url", raw=url, start=match.start(), end=match.end(), url=url)
        )
        url_spans.append((match.start(), match.end()))

    found.sort(key=lambda item: (item.start, item.end))
    return _dedupe(found)


def prepare_text(text: str) -> str:
    """Undo what copying from a PDF or a web page does to identifiers.

    Zero-width characters are invisible and never part of an identifier. A
    DOI link written with %2F for its slash is the same DOI. A DOI broken at
    a period at the end of a line ("10.1109/CVPR.\\n2016.90") is one DOI.
    """
    text = _ZERO_WIDTH.sub("", text or "")
    text = re.sub(r"(\b10\.\d{4,9})%2F", r"\1/", text, flags=re.IGNORECASE)
    return re.sub(r"(\b10\.\d{4,9}/[^\s<>\"']*[.\-])[ \t]*\r?\n[ \t]*(?=\d)", r"\1", text)


def unresolvable_mentions(text: str) -> list[str]:
    """Pointers with nothing to look up: "(Nguyen et al., 2021)" and "ibid."."""
    text = prepare_text(text)
    found = [match.group(0) for match in _AUTHOR_YEAR.finditer(text)]
    found += [match.group(0) for match in _IBID.finditer(text)]
    return list(dict.fromkeys(found))


def _arxiv_problem(doi: str) -> str | None:
    match = re.fullmatch(r"10\.48550/arXiv\.(\d{2})(\d{2})\.\d{4,5}", doi)
    if not match:
        return None
    month = int(match.group(2))
    if 1 <= month <= 12:
        return None
    arxiv_id = doi.split("arXiv.", 1)[1]
    return (
        f"arXiv:{arxiv_id} cannot exist. New-style arXiv ids start with the year and month, "
        f"and {month:02d} is not a month."
    )


def _link_label(links: dict[str, str], token: str) -> str | None:
    target = _link_target(links, token)
    if target is None:
        return None
    label = links[target].strip()
    if word_count(label) < 3 or _VAGUE_LABEL.search(label):
        return None
    return label


def _link_target(links: dict[str, str], token: str) -> str | None:
    return next((url for url in links if token in url), None)


def clean_doi(raw: str) -> str:
    doi = raw.strip()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.IGNORECASE)
    if "%" in doi:
        doi = unquote(doi)  # 10.1016/S0140-6736%2897%2911096-0 is (97)11096-0
    # "…-2.)" ends with a period inside a closing parenthesis. Strip until stable.
    previous = None
    while doi != previous:
        previous = doi
        doi = doi.rstrip(".,;:")
        if doi.endswith(")") and doi.count("(") < doi.count(")"):
            doi = doi[:-1]
        if doi.endswith("]") and doi.count("[") < doi.count("]"):
            doi = doi[:-1]
    return doi


def isbn_valid(isbn: str) -> bool:
    """ISBN-10 or ISBN-13 checksum. A number that fails it cannot be a real ISBN."""
    if re.fullmatch(r"\d{9}[\dX]", isbn):
        total = sum(
            (10 - index) * (10 if char == "X" else int(char)) for index, char in enumerate(isbn)
        )
        return total % 11 == 0
    if re.fullmatch(r"97[89]\d{10}", isbn):
        total = sum(int(char) * (1 if index % 2 == 0 else 3) for index, char in enumerate(isbn))
        return total % 10 == 0
    return False


def clean_url(raw: str) -> str:
    url = raw.strip()
    while url and url[-1] in ".,;:":
        url = url[:-1]
    if url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    if url.endswith("]") and url.count("[") < url.count("]"):
        url = url[:-1]
    return url


def find_references(text: str) -> list[ExtractedCitation]:
    """Structured reference entries in ``text``, without DOIs or bare URLs."""
    return _find_references(text)


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


def _parse_bibliographic(line: str) -> dict[str, Any]:
    """Best-effort metadata for a line that already has a DOI or URL."""
    cleaned = _URL_RE.sub(" ", line)
    cleaned = _DOI_RE.sub(" ", cleaned)
    cleaned = _BULLET_RE.sub("", cleaned.strip())
    match = _YEAR_PAREN_RE.search(cleaned)
    if match:
        head = _author_segment(cleaned[: match.start()])
        surnames, et_al = _surnames(head)
        title, venue = _split_title_venue(cleaned[match.end() :], min_words=2)
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
    surname = (
        _VANCOUVER_NAME.search(segment) or _SURNAME_RE.search(segment) or _ET_AL_RE.search(segment)
    )
    if surname and surname.start() > 0:
        segment = segment[surname.start() :].strip()
    return segment


def _surnames(text: str) -> tuple[list[str], bool]:
    et_al = bool(re.search(r"\bet\s+al\.?", text, flags=re.IGNORECASE))
    names: list[str] = []
    vancouver = [
        match.group(1)
        for match in _VANCOUVER_NAME.finditer(text)
        if match.group(1).casefold() not in _STOP_AUTHORS
    ]
    if vancouver:
        return list(dict.fromkeys(vancouver)), et_al
    for match in _SURNAME_RE.finditer(text):
        surname = re.sub(r"\s+", " ", match.group(1)).strip()
        if surname.casefold() in _STOP_AUTHORS or _INITIALS.match(surname):
            continue
        if surname not in names:
            names.append(surname)
    if not names:
        for match in _ET_AL_RE.finditer(text):
            surname = match.group(1)
            if surname.casefold() in _STOP_AUTHORS or _INITIALS.match(surname):
                continue
            if surname not in names:
                names.append(surname)
            et_al = True
    return names, et_al


def _split_title_venue(rest: str, *, min_words: int = 3) -> tuple[str | None, str | None]:
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
    # A free-text search needs three words to be specific. With a DOI the
    # record is already known, so "Deep learning" is enough to compare.
    if word_count(title) < min_words or len(title) < (12 if min_words >= 3 else 6):
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
    seen: dict[tuple[str, ...], ExtractedCitation] = {}
    kept: list[ExtractedCitation] = []
    for item in items:
        key: tuple[str, ...]
        if item.kind == "doi" and item.doi:
            # The same DOI printed with different details is a different
            # citation: [2] "Hinton 1990, Science" is not a repeat of [1].
            details = _details_key(item)
            key = ("doi", item.doi.lower(), *details)
            if not details and any(k[:2] == key for k in seen):
                key = next(k for k in seen if k[:2] == key)
        elif item.kind == "url" and item.url:
            key = ("url", item.url.lower().rstrip("/"))
        elif item.kind == "isbn" and item.isbn:
            key = ("isbn", item.isbn)
        elif item.kind == "pmid" and item.pmid:
            key = ("pmid", item.pmid)
        elif item.kind == "pmc" and item.pmc:
            key = ("pmc", item.pmc)
        else:
            folded = re.sub(r"\W+", " ", (item.title or "").casefold()).strip()
            key = ("ref", folded, str(item.year or 0))
        if key in seen:
            seen[key].occurrences += 1  # checked once, reported as cited twice
            continue
        seen[key] = item
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


_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def dangling_markers(text: str) -> list[str]:
    """Numbered markers such as "[12]" with no "[12] …" reference line to match."""
    entries = set(re.findall(r"(?m)^\s*\[(\d{1,3})\]\s+\S", text or ""))
    found: list[str] = []
    for match in _MARKER_RE.finditer(text or ""):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if not text[line_start : match.start()].strip():
            continue  # this is a reference-list entry, not a pointer
        if match.group(1) in entries:
            continue
        label = f"[{match.group(1)}]"
        if label not in found:
            found.append(label)
    return found


def _original_line(text: str, token: str) -> str | None:
    """The line of the paste, before joining, that holds ``token``."""
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if token in line:
            return line.strip()
    return None


def _details_key(item: ExtractedCitation) -> tuple[str, ...]:
    if not item.title and not item.year:
        return ()
    title = re.sub(r"\W+", " ", (item.title or "").casefold()).strip()
    return (title, str(item.year or 0), (item.venue or "").casefold())


def _original_line_at(text: str, normalized: str, match: re.Match[str]) -> str | None:
    """The pasted line holding this match, counting repeats of the same token.

    Joining lines changes only whitespace, so the n-th copy of a token in the
    joined text is the n-th copy in the paste. Two references that cite the
    same DOI each get their own line.
    """
    token = match.group(0)
    nth = normalized[: match.start()].count(token)
    position = -1
    for _ in range(nth + 1):
        position = text.find(token, position + 1)
        if position < 0:
            return _original_line(text, token)
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start : end if end >= 0 else len(text)].strip()
