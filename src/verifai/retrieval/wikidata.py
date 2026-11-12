"""Wikidata facts about the people, places, and things a claim names.

Wikidata is curated separately from Wikipedia's article text, so it is a
second, independent source for everyday claims. A label and a one-line
description are rarely enough to judge, so each matched item is rendered
as a short passage of its key statements: dates, offices with their start
and end, creators, heights, election winners, and so on.
"""

from __future__ import annotations

import re
from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import detect_language, fetch_json
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://www.wikidata.org/w/api.php"
_MAX_ITEMS = 3
_MAX_VALUES = 4
_MAX_CHARS = 1600

# Statements worth showing the judge, in display order.
PROPERTIES: dict[str, str] = {
    "P31": "instance of",
    "P569": "date of birth",
    "P570": "date of death",
    "P19": "place of birth",
    "P27": "country of citizenship",
    "P106": "occupation",
    "P39": "position held",
    "P102": "member of political party",
    "P571": "inception",
    "P577": "publication date",
    "P575": "time of discovery or invention",
    "P61": "discoverer or inventor",
    "P170": "creator",
    "P178": "developer",
    "P112": "founded by",
    "P50": "author",
    "P800": "notable work",
    "P166": "award received",
    "P585": "point in time",
    "P991": "successful candidate",
    "P1346": "winner",
    "P726": "candidate",
    "P17": "country",
    "P131": "located in",
    "P36": "capital",
    "P159": "headquarters location",
    "P1082": "population",
    "P2044": "elevation above sea level",
    "P2048": "height",
    "P2043": "length",
    "P2046": "area",
    "P2067": "mass",
    "P2101": "melting point",
    "P2102": "boiling point",
    "P1181": "numeric value",
    "P2442": "also expressed as",
    "P2097": "term length of office",
    "P26": "spouse",
    "P69": "educated at",
    "P108": "employer",
}
_QUALIFIERS = {"P580": "from", "P582": "until", "P585": "in", "P1545": "number"}
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_CONNECTOR = r"(?:of|van|von|de|der|den|du|la|le|da|di|the|and)"
_NAME = re.compile(
    rf"[A-Z][\w'’.\-]*[\w'’](?:\s+(?:{_CONNECTOR}\s+)*[A-Z][\w'’.\-]*[\w'’])*",
)
_EVENT = re.compile(
    r"\b((?:19|20)\d{2})\s+((?:[A-Za-z.]+\s+){0,4}?"
    r"(?:presidential election|general election|election|summer olympics|winter olympics|"
    r"olympics|world cup|census|referendum))\b",
    re.IGNORECASE,
)
_SUBJECT = re.compile(
    r"(?:the\s+|a\s+|an\s+)?([a-z][a-z\s'’-]{2,50}?)\s+(?:is|are|was|were|has|have|equals)\b",
    re.IGNORECASE,
)
_ABBREVIATED = {"U.S.": "United States", "US": "United States", "U.K.": "United Kingdom"}
_STARTERS = frozenset(
    [
        "a",
        "an",
        "the",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "in",
        "on",
        "at",
        "as",
        "by",
        "for",
        "from",
        "with",
        "contrary",
        "according",
        "many",
        "most",
        "some",
        "all",
        "if",
        "when",
        "while",
        "after",
        "before",
        "although",
        "taking",
        "smoking",
        "eating",
        "drinking",
        "studies",
        "research",
        "experts",
        "scientists",
        "doctors",
    ]
)


class WikidataRetriever:
    name = "wikidata"

    def __init__(self, http: PoliteHttp, *, limit: int = _MAX_ITEMS) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        text = " ".join((query or "").split())[:400]
        if not text:
            return RetrievalResult(self.name)
        lang = detect_language(text)
        ids: list[str] = []
        failed = 0
        names = entity_names(text)[: self.limit]
        for name in names:
            found, ok = self._search(name, lang)
            failed += not ok
            if found and found not in ids:
                ids.append(found)
        if names and failed == len(names):
            return RetrievalResult(self.name, unavailable=True, error="Wikidata search failed")
        if not ids:
            return RetrievalResult(self.name)
        entities = self._entities(ids, lang, props="labels|descriptions|claims")
        if entities is None:
            return RetrievalResult(self.name, unavailable=True, error="Wikidata entities failed")
        referenced = sorted({ref for qid in ids for ref in _references(entities.get(qid, {}))})
        labels = self._labels(referenced, lang)
        passages = [
            passage
            for qid in ids
            if (passage := render_item(qid, entities.get(qid, {}), labels, lang)) is not None
        ]
        return RetrievalResult(self.name, passages=passages)

    def _search(self, name: str, lang: str) -> tuple[str | None, bool]:
        body, _error = fetch_json(
            self.http,
            _API,
            {
                "action": "wbsearchentities",
                "search": name[:120],
                "language": lang,
                "uselang": lang,
                "strictlanguage": 0,
                "format": "json",
                "limit": 1,
                "type": "item",
            },
            source=self.name,
        )
        if body is None:
            return None, False
        search = body.get("search")
        if not isinstance(search, list) or not search or not isinstance(search[0], dict):
            return None, True
        found = search[0].get("id")
        return (found if isinstance(found, str) else None), True

    def _entities(self, ids: list[str], lang: str, *, props: str) -> dict[str, Any] | None:
        if not ids:
            return {}
        body, _error = fetch_json(
            self.http,
            _API,
            {
                "action": "wbgetentities",
                "ids": "|".join(ids[:50]),
                "props": props,
                "languages": "|".join(dict.fromkeys([lang, "en"])),
                "languagefallback": 1,
                "format": "json",
            },
            source=self.name,
        )
        if body is None:
            return None
        entities = body.get("entities")
        return entities if isinstance(entities, dict) else {}

    def _labels(self, ids: list[str], lang: str) -> dict[str, str]:
        entities = self._entities(ids[:50], lang, props="labels") or {}
        return {qid: label for qid, item in entities.items() if (label := _label(item, lang))}


def entity_names(claim: str) -> list[str]:
    """Names to look up: dated events first, then capitalised names in reading order."""
    text = claim
    for short, full in _ABBREVIATED.items():
        text = re.sub(rf"(?<![\w.]){re.escape(short)}(?![\w])", full, text)
    names = [f"{match.group(1)} {match.group(2)}".strip() for match in _EVENT.finditer(text)]
    spans: list[str] = []
    for match in _NAME.finditer(text):
        span = match.group(0).strip(" .")
        words = span.split()
        while words and words[0].casefold() in _STARTERS:
            words = words[1:]
        if not words:
            continue
        span = re.sub(r"['’]s$", "", " ".join(words))
        at_start = match.start() == 0 or not text[: match.start()].strip()
        if len(words) == 1 and (at_start and words[0].casefold().endswith("ing")):
            continue
        if len(span) < 3 or span.isdigit():
            continue
        spans.append(span)
    for span in spans:
        if not any(span in other for other in names):
            names.append(span)
    if not names:
        # "The speed of light is …" names no proper noun. Its subject is the item.
        subject = _SUBJECT.match(text.strip())
        if subject:
            names.append(subject.group(1).strip())
    return names


def render_item(
    qid: str, item: dict[str, Any], labels: dict[str, str], lang: str
) -> Passage | None:
    label = _label(item, lang)
    if not label:
        return None
    description = _description(item, lang)
    facts: list[str] = []
    claims = _dict(item.get("claims"))
    for prop, name in PROPERTIES.items():
        values = [
            text
            for statement in _best(claims.get(prop))
            if (text := _statement_text(statement, labels))
        ][:_MAX_VALUES]
        if values:
            facts.append(f"{name}: {', '.join(values)}")
    head = f"{label} ({description})" if description else label
    body = "; ".join(facts)
    text = f"{head}. {body}." if body else f"{head}."
    return Passage(
        source="wikidata",
        family="wikidata",
        title=label,
        url=f"https://www.wikidata.org/wiki/{qid}",
        text=text[:_MAX_CHARS],
        document_ids=(f"wikidata:{qid}",),
    )


def _best(statements: object) -> list[dict[str, Any]]:
    """Preferred statements if any, else normal ones. Deprecated ones never."""
    if not isinstance(statements, list):
        return []
    usable = [
        item for item in statements if isinstance(item, dict) and item.get("rank") != "deprecated"
    ]
    preferred = [item for item in usable if item.get("rank") == "preferred"]
    # A position held or an award is a history, not one current value.
    if preferred and not any("qualifiers" in item for item in usable):
        return preferred
    return usable


def _statement_text(statement: dict[str, Any], labels: dict[str, str]) -> str:
    main = _value_text(statement.get("mainsnak"), labels)
    if not main:
        return ""
    qualifiers = _dict(statement.get("qualifiers"))
    extra = [
        f"{word} {text}"
        for prop, word in _QUALIFIERS.items()
        for snak in (qualifiers.get(prop) or [])[:1]
        if (text := _value_text(snak, labels))
    ]
    return f"{main} ({', '.join(extra)})" if extra else main


def _value_text(snak: object, labels: dict[str, str]) -> str:
    if not isinstance(snak, dict) or snak.get("snaktype") not in {None, "value"}:
        return ""
    data = snak.get("datavalue")
    if not isinstance(data, dict):
        return ""
    value = data.get("value")
    kind = data.get("type")
    if kind == "wikibase-entityid" and isinstance(value, dict):
        return labels.get(str(value.get("id")), "")
    if kind == "time" and isinstance(value, dict):
        return _time_text(value)
    if kind == "quantity" and isinstance(value, dict):
        amount = str(value.get("amount", "")).lstrip("+")
        unit = str(value.get("unit", "")).rsplit("/", 1)[-1]
        unit_label = labels.get(unit, "") if unit not in {"", "1"} else ""
        return f"{amount} {unit_label}".strip()
    if kind == "monolingualtext" and isinstance(value, dict):
        return str(value.get("text", ""))
    if kind == "string" and isinstance(value, str):
        return value
    return ""


def _time_text(value: dict[str, Any]) -> str:
    match = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", str(value.get("time", "")))
    if not match:
        return ""
    sign, year, month, day = match.groups()
    era = " BCE" if sign == "-" else ""
    precision = value.get("precision")
    if precision == 11 and month != "00" and day != "00":
        return f"{int(day)} {_MONTHS[int(month) - 1]} {int(year)}{era}"
    if precision == 10 and month != "00":
        return f"{_MONTHS[int(month) - 1]} {int(year)}{era}"
    return f"{int(year)}{era}"


def _references(item: dict[str, Any]) -> set[str]:
    """Item ids whose labels the rendered passage needs, including units."""
    found: set[str] = set()
    claims = _dict(item.get("claims"))
    for prop in PROPERTIES:
        for statement in _best(claims.get(prop))[:_MAX_VALUES]:
            snaks = [statement.get("mainsnak")]
            qualifiers = statement.get("qualifiers")
            if isinstance(qualifiers, dict):
                snaks += [snak for key in _QUALIFIERS for snak in (qualifiers.get(key) or [])[:1]]
            for snak in snaks:
                data = snak.get("datavalue") if isinstance(snak, dict) else None
                value = data.get("value") if isinstance(data, dict) else None
                if not isinstance(value, dict):
                    continue
                if isinstance(value.get("id"), str):
                    found.add(value["id"])
                unit = str(value.get("unit", "")).rsplit("/", 1)[-1]
                if unit.startswith("Q"):
                    found.add(unit)
    return found


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _label(item: dict[str, Any], lang: str) -> str:
    return _localized(item.get("labels"), lang)


def _description(item: dict[str, Any], lang: str) -> str:
    return _localized(item.get("descriptions"), lang)


def _localized(values: object, lang: str) -> str:
    if not isinstance(values, dict):
        return ""
    for key in (lang, "en", *values.keys()):
        entry = values.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("value"), str):
            return str(entry["value"])
    return ""
