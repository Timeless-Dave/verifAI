"""Pin "today", "yesterday", and "this week" to calendar dates.

"Jensen Huang resigned today" is a different claim on every day it is
pasted. Judged as "Jensen Huang resigned on 13 November 2026", a dated news
report can confirm it, or show that it happened on another day.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

# "USA Today" and "Today show" name things; they are not dates.
_NOT_A_DATE = r"(?<!USA )(?<!usa )"
_RELATIVE: list[tuple[re.Pattern[str], Callable[[date], str]]] = [
    (
        re.compile(
            rf"\b(?:earlier |later )?{_NOT_A_DATE}(?:today|tonight|this morning|this "
            r"afternoon|this evening)\b(?! show)",
            re.IGNORECASE,
        ),
        lambda day: f"on {_day(day)}",
    ),
    (
        re.compile(r"\b(?:yesterday|last night)\b", re.IGNORECASE),
        lambda day: f"on {_day(day - timedelta(days=1))}",
    ),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), lambda day: f"on {_day(day + timedelta(days=1))}"),
    (
        re.compile(r"\bthis week\b", re.IGNORECASE),
        lambda day: f"in the week of {_day(day - timedelta(days=day.weekday()))}",
    ),
    (
        re.compile(r"\blast week\b", re.IGNORECASE),
        lambda day: f"in the week of {_day(day - timedelta(days=day.weekday() + 7))}",
    ),
    (re.compile(r"\bthis month\b", re.IGNORECASE), lambda day: f"in {day:%B %Y}"),
    (
        re.compile(r"\blast month\b", re.IGNORECASE),
        lambda day: f"in {(day.replace(day=1) - timedelta(days=1)):%B %Y}",
    ),
    (re.compile(r"\bthis year\b", re.IGNORECASE), lambda day: f"in {day.year}"),
    (re.compile(r"\blast year\b", re.IGNORECASE), lambda day: f"in {day.year - 1}"),
]


@dataclass(frozen=True)
class DatedClaim:
    # The claim with each relative date replaced by a calendar date.
    dated: str
    # The claim with the relative words removed, for searching.
    search: str
    # What each phrase was read as: ("today", "on 13 November 2026").
    readings: tuple[tuple[str, str], ...]


def pin_dates(claim: str, today: date | None = None) -> DatedClaim:
    day = today or datetime.now(UTC).date()
    dated = claim
    search = claim
    readings: list[tuple[str, str]] = []
    for pattern, render in _RELATIVE:
        when = render(day)
        readings.extend((match.group(0), when) for match in pattern.finditer(dated))
        dated = pattern.sub(when, dated)
        search = pattern.sub(" ", search)
    return DatedClaim(
        dated=" ".join(dated.split()),
        search=re.sub(r"\s+([.,;:!?])", r"\1", " ".join(search.split())),
        readings=tuple(dict.fromkeys(readings)),
    )


def _day(day: date) -> str:
    return f"{day.day} {day:%B %Y}"
