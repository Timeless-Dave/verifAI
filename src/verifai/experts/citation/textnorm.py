"""Deterministic string normalization for titles, venues, and surnames."""

from __future__ import annotations

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
_PARTICLES = {"van", "von", "de", "del", "der", "den", "di", "da", "le", "la", "st"}


def fold(value: str) -> str:
    """Casefold, strip accents, and turn punctuation into spaces."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def word_count(value: str) -> int:
    return len(re.findall(r"\w+", value, flags=re.UNICODE))


def surname_from_display(name: str) -> str:
    parts = [part for part in re.split(r"\s+", name.strip()) if part]
    while parts and parts[-1].casefold().rstrip(".") in _SUFFIXES:
        parts.pop()
    if not parts:
        return name.strip()
    if (
        len(parts) >= 3
        and parts[-2].casefold() in _PARTICLES
        and parts[-3].casefold() in _PARTICLES
    ):
        return " ".join(parts[-3:])
    if len(parts) >= 2 and parts[-2].casefold() in _PARTICLES:
        return " ".join(parts[-2:])
    return parts[-1]


def surnames_match(left: str, right: str) -> bool:
    a = fold(left)
    b = fold(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 4:
        return False
    return longer.endswith(" " + shorter) or longer.replace(" ", "").endswith(shorter)


def format_people(names: list[str], limit: int = 4) -> str:
    shown = [name for name in names if name][:limit]
    if not shown:
        return ""
    text = ", ".join(shown)
    if len(names) > limit:
        text += ", et al."
    return text


def canonical_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    text = doi.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text.lower() or None
