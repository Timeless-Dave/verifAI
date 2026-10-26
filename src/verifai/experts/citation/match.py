"""Compare a parsed citation with a retrieved bibliographic record.

Scores are deterministic string similarity (RapidFuzz). Nothing here calls
a model or the network.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from verifai.experts.citation.providers.base import Record
from verifai.experts.citation.textnorm import fold, format_people, surnames_match
from verifai.schemas import CitationStatus

_VENUE_STOP = {"of", "the", "and", "for", "in", "on", "de", "la", "a", "an"}


@dataclass
class Query:
    title: str | None
    surnames: list[str]
    year: int | None
    venue: str | None
    et_al: bool


@dataclass
class TitleScores:
    set_ratio: float
    sort_ratio: float
    ratio: float
    blend: float


@dataclass
class Assessment:
    status: CitationStatus
    title_blend: float
    mismatches: list[str]
    author_status: str
    year_status: str
    venue_status: str


def title_scores(left: str | None, right: str | None) -> TitleScores:
    a = fold(left or "")
    b = fold(right or "")
    if not a or not b:
        return TitleScores(0.0, 0.0, 0.0, 0.0)
    set_ratio = fuzz.token_set_ratio(a, b) / 100
    sort_ratio = fuzz.token_sort_ratio(a, b) / 100
    ratio = fuzz.ratio(a, b) / 100
    blend = (0.25 * set_ratio) + (0.35 * sort_ratio) + (0.40 * ratio)
    return TitleScores(set_ratio, sort_ratio, ratio, blend)


def assess(query: Query, record: Record) -> Assessment | None:
    """Return a comparison, or None when the record is not about this citation."""
    if not query.title or not record.title:
        return None
    scores = title_scores(query.title, record.title)
    strong = scores.blend >= 0.90 or (scores.ratio >= 0.92 and scores.sort_ratio >= 0.88)
    medium = scores.blend >= 0.80 or (scores.set_ratio >= 0.96 and scores.ratio >= 0.75)
    if not medium:
        return None

    author_status = compare_authors(query, record)
    year_status = compare_year(query, record)
    venue_status = compare_venue(query.venue, record.venue)

    # A different paper can share a few title words. Require a near-exact
    # title before a double mismatch (wrong authors and wrong year) counts
    # as the same work with bad metadata.
    if author_status == "mismatch" and year_status == "mismatch" and scores.blend < 0.96:
        return None

    mismatches: list[str] = []
    if author_status in {"mismatch", "partial"}:
        mismatches.append(_author_message(query, record, author_status))
    if year_status == "mismatch":
        mismatches.append(
            f"Year mismatch: citation says {query.year}, record says {_year_label(record)}."
        )
    elif year_status == "off_by_one":
        mismatches.append(
            f"Year is off by one: citation says {query.year}, record says {_year_label(record)}. "
            "That sometimes means a preprint year versus the journal year."
        )
    if venue_status == "mismatch":
        mismatches.append(
            f"Venue mismatch: citation says {query.venue}, record says {record.venue}."
        )

    if strong and not mismatches:
        status = CitationStatus.VERIFIED
    elif mismatches or (
        author_status in {"match", "partial", "unknown"}
        and year_status in {"match", "off_by_one", "unknown"}
    ):
        status = CitationStatus.PARTIALLY_MATCHES
        if not mismatches:
            percent = round(scores.blend * 100)
            mismatches.append(
                f"Title is only a partial match ({percent}% similar) to “{record.title}”."
            )
    else:
        return None

    return Assessment(
        status=status,
        title_blend=scores.blend,
        mismatches=mismatches,
        author_status=author_status,
        year_status=year_status,
        venue_status=venue_status,
    )


def compare_authors(query: Query, record: Record) -> str:
    if not query.surnames or not record.surnames:
        return "unknown"
    cited = query.surnames
    recorded = record.surnames

    def listed(surname: str, pool: list[str]) -> bool:
        return any(surnames_match(surname, other) for other in pool)

    first_ok = surnames_match(cited[0], recorded[0])
    matched = sum(1 for surname in cited if listed(surname, recorded))
    if query.et_al or len(cited) == 1:
        if first_ok:
            return "match"
        if listed(cited[0], recorded):
            return "partial"
        return "mismatch"
    if matched == len(cited) and first_ok:
        return "match"
    if matched == 0:
        return "mismatch"
    return "partial"


def compare_year(query: Query, record: Record) -> str:
    if query.year is None or not record.years:
        return "unknown"
    if query.year in record.years:
        return "match"
    if any(abs(query.year - year) == 1 for year in record.years):
        return "off_by_one"
    return "mismatch"


def compare_venue(query_venue: str | None, record_venue: str | None) -> str:
    if not query_venue or not record_venue:
        return "unknown"
    scores = title_scores(query_venue, record_venue)
    left = fold(query_venue)
    right = fold(record_venue)
    if scores.blend >= 0.80 or scores.ratio >= 0.86:
        return "match"
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if _is_abbreviation(shorter, longer):
        return "match"
    if scores.blend < 0.50 and len(shorter) >= 4:
        return "mismatch"
    return "unknown"


def _is_abbreviation(short: str, long: str) -> bool:
    compact = short.replace(" ", "")
    if not 3 <= len(compact) <= 12:
        return False
    words = [word for word in long.split() if word not in _VENUE_STOP]
    initials = "".join(word[0] for word in words if word)
    return compact == initials


def _author_message(query: Query, record: Record, status: str) -> str:
    cited = " and ".join(query.surnames)
    found = format_people(record.authors, limit=4) or " and ".join(record.surnames)
    if status == "partial":
        return (
            f"Author mismatch: citation lists {cited}; only some of those names "
            f"are on the record ({found})."
        )
    return f"Author mismatch: citation lists {cited}; the record lists {found}."


def _year_label(record: Record) -> str:
    if record.year is None:
        return "an unknown year"
    if len(record.years) > 1 and record.year is not None:
        extras = [str(year) for year in record.years if year != record.year]
        if extras:
            return f"{record.year} (also {', '.join(extras)})"
    return str(record.year)
