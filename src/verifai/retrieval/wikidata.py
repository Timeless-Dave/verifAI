"""Wikidata facts about the people, places, and things a claim names.

Wikidata is curated separately from Wikipedia's article text, so it is a
second, independent source for everyday claims. A label and a one-line
description are rarely enough to judge, so each matched item is rendered
as a short passage of its key statements: dates, offices with their start
and end, creators, heights, election winners, and so on.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import detect_language, fetch_json
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://www.wikidata.org/w/api.php"
_SPARQL = "https://query.wikidata.org/sparql"
# Taxon ranks worth naming, broadest last: species up to kingdom.
_RANKS = {
    "Q7432": "species",
    "Q34740": "genus",
    "Q35409": "family",
    "Q36602": "order",
    "Q37517": "class",
    "Q38348": "phylum",
    "Q36732": "kingdom",
}
_RANK_ORDER = list(_RANKS.values())
# Words in a claim that say what kind of thing its subject is: "Apple was
# founded" is about a company, so a company's description should match.
_COMPANY = frozenset({"company", "corporation", "business", "enterprise", "organization", "brand"})
_HINTS = {
    "founded": _COMPANY,
    "headquartered": _COMPANY,
    "released": frozenset({"album", "film", "song", "smartphone", "product", "software"}),
}
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
    "P287": "designed by",
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
    "P580": "start time",
    "P582": "end time",
    "P276": "location",
    "P30": "continent",
    "P403": "mouth of the watercourse",
    "P885": "origin of the watercourse",
    "P4511": "vertical depth",
    "P246": "element symbol",
    "P1086": "atomic number",
    "P84": "architect",
    "P86": "composer",
    "P57": "director",
    "P175": "performer",
    "P495": "country of origin",
    "P1120": "number of deaths",
    "P279": "subclass of",
    "P361": "part of",
    "P186": "made from material",
    "P527": "has part",
    "P35": "head of state",
    "P6": "head of government",
    "P37": "official language",
    "P38": "currency",
    "P169": "chief executive officer",
    "P1056": "product or material produced",
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
    rf"(?:[A-Z]|\b[a-z]{{1,2}}(?=[A-Z]))[\w'’.\-]*[\w'’]"
    rf"(?:\s+(?:{_CONNECTOR}\s+)*[A-Z][\w'’.\-]*[\w'’])*",
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
        self._claim = ""

    def retrieve(self, query: str) -> RetrievalResult:
        text = " ".join((query or "").split())[:400]
        if not text:
            return RetrievalResult(self.name)
        lang = detect_language(text)
        self._claim = text
        names = entity_names(text)[: self.limit]
        searched: list[tuple[str, list[dict[str, Any]]]] = []
        failed = 0
        for name in names:
            hits, ok = self._search(name, lang)
            single = _singular(name)
            if ok and single:
                # "Spiders are insects" is about the spider, not a studio
                # called Spiders. Search both and let the ranking choose.
                more, _ok = self._search(single, lang)
                known = {hit["id"] for hit in hits}
                hits = hits + [hit for hit in more if hit["id"] not in known]
            # "the chemical symbol for gold" is no item. Gold is.
            tail = re.search(r"\b(?:of|for)\s+(?:the\s+)?([a-z][a-z'’-]{2,})$", name)
            if ok and not hits and tail:
                hits, ok = self._search(tail.group(1), lang)
            failed += not ok
            searched.append((name, hits))
        if names and failed == len(names):
            return RetrievalResult(self.name, unavailable=True, error="Wikidata search failed")
        reach = self._sitelinks([str(hit["id"]) for _name, hits in searched for hit in hits])
        ids: list[str] = []
        named: dict[str, str] = {}
        for name, hits in searched:
            # "Bats are blind" capitalises Bats only because it starts the
            # sentence. Then case says nothing: the bat, not the French village.
            opening = " " not in name and self._claim.lstrip().startswith(name)
            found = _pick(hits, name, self._claim, reach, also=_singular(name), fold=opening)
            if found and found not in ids:
                ids.append(found)
                named[found] = name
        if not ids:
            return RetrievalResult(self.name)
        # One item at a time: a big item (a country) can pass the size cap
        # alone, and it must not take the others down with it.
        entities: dict[str, Any] = {}
        for qid in ids:
            entities.update(self._entities([qid], lang, props="labels|descriptions|claims") or {})
        if not entities:
            return RetrievalResult(self.name, unavailable=True, error="Wikidata entities failed")
        referenced = sorted({ref for qid in ids for ref in _references(entities.get(qid, {}))})
        labels = self._labels(referenced, lang)
        passages = [
            passage
            for qid in ids
            if qid in entities
            and (
                passage := render_item(
                    qid,
                    entities[qid],
                    labels,
                    lang,
                    lineage=self._lineage(qid, entities[qid]),
                    about=named.get(qid),
                )
            )
            is not None
        ]
        return RetrievalResult(self.name, passages=passages)

    def _lineage(self, qid: str, item: dict[str, Any]) -> str:
        """Where a living thing sits: "species dolphin; order Cetacea; class mammal".

        A taxon names only its direct parent, so "Dolphins are mammals" needs
        the whole line up to the kingdom. One query fetches it.
        """
        if not _dict(item.get("claims")).get("P171") or not re.fullmatch(r"Q\d+", qid):
            return ""
        # Walk the parent links from the item up, so each rank takes its
        # nearest ancestor: a penguin's class is bird, not the reptile clade
        # Wikidata also files birds under.
        query = (
            f"SELECT ?taxon ?parent ?rank ?label WHERE {{ wd:{qid} wdt:P171* ?taxon . "
            "OPTIONAL { ?taxon wdt:P171 ?parent } OPTIONAL { ?taxon wdt:P105 ?rank } "
            "OPTIONAL { ?taxon rdfs:label ?label . FILTER(LANG(?label) = 'en') } }"
        )
        body, _error = fetch_json(
            self.http, _SPARQL, {"query": query, "format": "json"}, source=self.name
        )
        rows = (body or {}).get("results", {}).get("bindings", []) if body else []
        parent: dict[str, str] = {}
        rank_of: dict[str, str] = {}
        label_of: dict[str, str] = {}
        for row in rows if isinstance(rows, list) else []:
            taxon = _entity(row, "taxon")
            if not taxon:
                continue
            up = _entity(row, "parent")
            if up:
                parent.setdefault(taxon, up)
            rank = _entity(row, "rank")
            if rank in _RANKS:
                rank_of.setdefault(taxon, _RANKS[rank])
            cell = row.get("label")
            label = cell.get("value") if isinstance(cell, dict) else None
            if isinstance(label, str):
                label_of.setdefault(taxon, label)
        found: dict[str, str] = {}
        node: str | None = qid
        for _step in range(60):
            if node is None:
                break
            rank_name = rank_of.get(node)
            if rank_name and rank_name not in found and node in label_of:
                found[rank_name] = label_of[node]
            node = parent.get(node)
        return "; ".join(f"{rank} {found[rank]}" for rank in _RANK_ORDER if rank in found)

    def _search(self, name: str, lang: str) -> tuple[list[dict[str, Any]], bool]:
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
                "limit": 7,
                "type": "item",
            },
            source=self.name,
        )
        if body is None:
            return [], False
        hits = [
            hit
            for hit in (body.get("search") or [])
            if isinstance(hit, dict) and isinstance(hit.get("id"), str)
        ]
        return hits, True

    def _sitelinks(self, ids: list[str]) -> dict[str, int]:
        """How many Wikipedias cover each item: the river, not the band named after it."""
        reach: dict[str, int] = {}
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), 25):
            body, _error = fetch_json(
                self.http,
                _API,
                {
                    "action": "wbgetentities",
                    "ids": "|".join(unique[start : start + 25]),
                    "props": "sitelinks",
                    "format": "json",
                },
                source=self.name,
            )
            entities = body.get("entities") if body else None
            for qid, item in (entities if isinstance(entities, dict) else {}).items():
                links = item.get("sitelinks") if isinstance(item, dict) else None
                reach[qid] = len(links) if isinstance(links, dict) else 0
        return reach

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


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]{4,}", text.casefold()))


def _entity(row: dict[str, Any], key: str) -> str:
    cell = row.get(key)
    value = cell.get("value") if isinstance(cell, dict) else None
    return str(value).rsplit("/", 1)[-1] if isinstance(value, str) else ""


def _singular(name: str) -> str:
    """ "Spiders" gives "spider". Only one-word plurals, and not "Paris" or "Texas"."""
    if " " in name or len(name) < 4 or not name.endswith("s") or name.endswith(("ss", "us", "is")):
        return ""
    stem = name[:-3] + "y" if name.endswith("ies") else name[:-1]
    return stem.casefold()


def _pick(
    hits: list[dict[str, Any]],
    name: str,
    claim: str,
    reach: dict[str, int],
    *,
    also: str = "",
    fold: bool = False,
) -> str | None:
    """The item a name means in this claim.

    A description that shares words with the rest of the claim decides first:
    the professor, not the basketball player. The name's own words do not
    count, or "Apple Music, by Apple" would beat Apple. Then an item called
    what the claim wrote, capitals included: "Apple" the company,
    not "apple" the fruit. Then the item most Wikipedias cover: the river
    Nile, not the band.
    """
    if not hits:
        return None
    context = _words(claim) - _words(name)
    for cue, hints in _HINTS.items():
        if cue in context:
            context |= hints
    ranked = sorted(
        enumerate(hits),
        key=lambda pair: (
            not (
                _exact(pair[1], name, fold=fold)
                or (bool(also) and _exact(pair[1], also, fold=True))
            ),
            -len(context & _words(str(pair[1].get("description") or ""))),
            -reach.get(str(pair[1]["id"]), 0),
            pair[0],
        ),
    )
    return str(ranked[0][1]["id"])


def _exact(hit: dict[str, Any], name: str, *, fold: bool = False) -> bool:
    """The item is called what the claim wrote, capitals kept, accents not.

    "Apple Inc." counts for "Apple"; "Brazilian Portuguese" does not count
    for "Brazil": the name must end at a word boundary.
    """
    match = hit.get("match")
    written = match.get("text") if isinstance(match, dict) else None
    want = _plain(name).casefold() if fold else _plain(name)
    for text in (hit.get("label"), written):
        if not text:
            continue
        have = _plain(str(text)).casefold() if fold else _plain(str(text))
        if re.match(rf"{re.escape(want)}(?![^\W_])", have):
            return True
    return False


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(char for char in folded if not unicodedata.combining(char))


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
    qid: str,
    item: dict[str, Any],
    labels: dict[str, str],
    lang: str,
    *,
    lineage: str = "",
    about: str | None = None,
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
    if lineage:
        facts.insert(0, f"classification: {lineage}")
    body = "; ".join(facts)
    text = f"{head}. {body}." if body else f"{head}."
    return Passage(
        source="wikidata",
        family="wikidata",
        title=label,
        url=f"https://www.wikidata.org/wiki/{qid}",
        text=text[:_MAX_CHARS],
        document_ids=(f"wikidata:{qid}",),
        about=about,
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
    # Only a few values are shown, so what holds now comes first, then the
    # newest: the United States' head of government today, not George
    # Washington, and its latest population, not the 1790 census.
    if any(_when(item) for item in usable):
        return sorted(usable, key=lambda item: (not _ongoing(item), _when(item)))
    return usable


def _when(statement: dict[str, Any]) -> str:
    """The start or point-in-time of a statement, as a sortable key (newest smallest)."""
    qualifiers = _dict(statement.get("qualifiers"))
    for prop in ("P580", "P585", "P582"):
        for snak in qualifiers.get(prop) or []:
            data = snak.get("datavalue") if isinstance(snak, dict) else None
            value = data.get("value") if isinstance(data, dict) else None
            stamp = str(value.get("time", "")) if isinstance(value, dict) else ""
            match = re.match(r"\+(\d{4,})-(\d\d)-(\d\d)", stamp)
            if match:
                year, month, day = match.groups()
                # Inverted so an ascending sort puts the newest first.
                return f"{99999 - int(year):05d}{99 - int(month):02d}{99 - int(day):02d}"
    return "~"


def _ongoing(statement: dict[str, Any]) -> bool:
    """Started and not ended: an office still held, a marriage still current."""
    qualifiers = _dict(statement.get("qualifiers"))
    return bool(qualifiers.get("P580")) and not qualifiers.get("P582")


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
    if _ongoing(statement):
        extra.append("to the present")
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
