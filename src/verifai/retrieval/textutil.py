"""Small text helpers shared by the free-source retrievers."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from verifai.corroboration import Stance
from verifai.http import PoliteHttp

_URL = re.compile(r"https?://[^\s<>\"']+")
_YEAR = re.compile(r"(?:18|19|20)\d{2}")
_WORD = re.compile(r"[^\W\d_][\w'-]{3,}")
_STOP = frozenset(
    [
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "into",
        "over",
        "after",
        "before",
        "about",
        "than",
        "then",
        "have",
        "been",
        "were",
        "they",
        "them",
        "their",
        "your",
        "what",
        "when",
        "where",
        "which",
        "while",
        "would",
        "could",
        "should",
        "there",
        "their",
        "just",
        "also",
        "more",
        "most",
        "some",
        "such",
        "only",
        "other",
        "into",
        "does",
        "did",
        "done",
        "doing",
        "make",
        "made",
        "many",
        "much",
        "very",
        "will",
        "shall",
        "into",
        "onto",
    ]
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(content: bytes, *, limit: int = 4000) -> str:
    parser = _TextExtractor()
    parser.feed(content.decode("utf-8", errors="replace"))
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:limit]


def extract_urls(text: str, *, limit: int = 3) -> list[str]:
    found: list[str] = []
    for match in _URL.finditer(text or ""):
        url = match.group(0).rstrip(").,;]")
        host = (urlsplit(url).hostname or "").lower()
        if host in {"doi.org", "dx.doi.org"} or url in found:
            continue
        found.append(url)
        if len(found) >= limit:
            break
    return found


def search_terms(text: str, *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for word in _WORD.findall(text or ""):
        folded = word.casefold()
        if folded in _STOP or folded in terms:
            continue
        terms.append(folded)
        if len(terms) >= limit:
            break
    return terms


def year_of(value: object) -> int | None:
    if isinstance(value, int) and 1800 <= value <= 2100:
        return value
    if isinstance(value, str):
        match = _YEAR.search(value)
        if match:
            return int(match.group(0))
    return None


def document_ids(*, doi: str | None = None, pmid: str | None = None) -> tuple[str, ...]:
    ids: list[str] = []
    if doi:
        cleaned = doi.strip().lower()
        cleaned = cleaned.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        cleaned = cleaned.removeprefix("doi:")
        if cleaned:
            ids.append(f"doi:{cleaned}")
    if pmid and str(pmid).isdigit():
        ids.append(f"pmid:{pmid}")
    return tuple(ids)


_NEGATED_TRUE = re.compile(
    r"\b(?:not|isn'?t|is not|un|no es|n'est pas|nicht|non è|não é)\s*(?:true|correct|accurate|"
    r"verdadero|cierto|vrai|wahr|vero|verdadeiro)\b"
)
_PARTLY = re.compile(r"\b(?:partly|partially|half|mostly|largely|in part)\b")
# English first, then French, Spanish, German, Italian, Portuguese.
_FALSE_WORDS = (
    "false",
    "fake",
    "incorrect",
    "wrong",
    "misleading",
    "fabricated",
    "hoax",
    "baseless",
    "faux",
    "fausse",
    "trompeur",
    "falso",
    "falsa",
    "engañoso",
    "falsch",
    "irreführend",
    "fuorviante",
    "enganoso",
)
_TRUE_WORDS = (
    "true",
    "correct",
    "accurate",
    "verdadero",
    "cierto",
    "vrai",
    "wahr",
    "richtig",
    "vero",
    "verdadeiro",
    "correcto",
    "exact",
)
_UNDECIDED = (
    "half true",
    "half-true",
    "mixture",
    "mixed",
    "unproven",
    "satire",
    "unverified",
    "missing context",
    "out of context",
    "lacks context",
    "exaggerat",
    "needs context",
    "unsupported",
    "no evidence",
    "sin pruebas",
    "trompeur sans",
    "à nuancer",
)


def stance_from_rating(rating: str) -> tuple[Stance, float]:
    """Map a fact-checker's verdict to a stance on the claim it reviewed.

    "Not true" is a refutation, not support. A partial rating such as
    "partly false" refutes at lower confidence. Words that mean false or
    true in the five other languages detect_language knows are read too.
    """
    text = " ".join(rating.casefold().split()).strip(" !.")
    if not text:
        return Stance.NEUTRAL, 0.0
    if _NEGATED_TRUE.search(text):
        return Stance.REFUTES, 0.85
    if any(bit in text for bit in _UNDECIDED):
        return Stance.NEUTRAL, 0.4
    for bit, confidence in (
        ("pants on fire", 0.9),
        ("four pinocchios", 0.9),
        ("three pinocchios", 0.75),
    ):
        if bit in text:
            return Stance.REFUTES, confidence
    words = set(re.findall(r"[\w-]+", text))
    partial = bool(_PARTLY.search(text))
    if any(bit in words or bit in text.split("-") for bit in _FALSE_WORDS) or "misleading" in text:
        confidence = 0.65 if partial or "misleading" in text or "engañoso" in text else 0.9
        return Stance.REFUTES, confidence
    if any(bit in words for bit in _TRUE_WORDS):
        if re.search(r"\b(?:partly|partially|half|in part)\b", text):
            return Stance.NEUTRAL, 0.4  # a mixed rating, like "half true"
        return Stance.SUPPORTS, 0.75 if partial else 0.9
    return Stance.NEUTRAL, 0.3


def fetch_json(
    http: PoliteHttp,
    url: str,
    params: dict[str, Any] | None,
    *,
    source: str,
) -> tuple[dict[str, Any] | None, str | None]:
    result = http.get(url, params=params)
    if result.error or result.status_code != 200:
        detail = result.error or f"HTTP {result.status_code}"
        return None, detail
    body = result.json_body()
    if not isinstance(body, dict):
        return None, f"{source} response was not JSON"
    return body, None


def host_family(url: str) -> str:
    host = (urlsplit(url).hostname or "page").lower().removeprefix("www.")
    return f"page:{host}"


# Common function words. Enough to tell which Wikipedia to search, not a
# general language detector.
_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "is",
            "are",
            "was",
            "were",
            "of",
            "and",
            "to",
            "in",
            "a",
            "an",
            "that",
            "it",
            "on",
            "for",
            "with",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "des",
            "est",
            "une",
            "un",
            "se",
            "du",
            "et",
            "au",
            "aux",
            "sont",
            "dans",
            "pour",
            "qui",
            "pas",
            "ne",
            "à",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "los",
            "las",
            "es",
            "del",
            "al",
            "una",
            "un",
            "y",
            "en",
            "se",
            "que",
            "por",
            "son",
            "con",
            "no",
            "está",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "ist",
            "und",
            "ein",
            "eine",
            "nicht",
            "mit",
            "von",
            "auf",
            "den",
            "dem",
            "im",
            "zu",
            "sich",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "lo",
            "gli",
            "è",
            "della",
            "che",
            "una",
            "per",
            "sono",
            "nel",
            "del",
            "di",
            "un",
            "non",
            "si",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "os",
            "as",
            "é",
            "da",
            "do",
            "uma",
            "um",
            "não",
            "em",
            "que",
            "são",
            "na",
            "no",
            "se",
            "com",
        ]
    ),
}
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _share(letters: list[str], pattern: str) -> float:
    return sum(1 for char in letters if re.match(pattern, char)) / len(letters)


def detect_language(text: str) -> str:
    """A Wikipedia language code for ``text``. English unless another clearly wins."""
    letters = [char for char in text or "" if char.isalpha()]
    if letters:
        if _share(letters, r"[\u3040-\u30ff]") > 0.05:
            return "ja"  # kana means Japanese even beside kanji
        if _share(letters, r"[\u4e00-\u9fff]") > 0.3:
            return "zh"
        if _share(letters, r"[\u0600-\u06ff]") > 0.3:
            return "ar"
    words = [word.casefold() for word in _TOKEN.findall(text or "")]
    scores = {lang: sum(word in markers for word in words) for lang, markers in _MARKERS.items()}
    best = max(scores, key=lambda lang: (scores[lang], lang == "en"))
    if best != "en" and scores[best] >= 2 and scores[best] > scores["en"]:
        return best
    return "en"


def needs_english(text: str) -> bool:
    """True when ``text`` is not English, so its sources should be searched in English.

    A detected language says so. A Latin-script sentence with no English
    function word, such as Yoruba or Swahili, is not English either.
    """
    if detect_language(text) != "en":
        return True
    words = [word.casefold() for word in _TOKEN.findall(text or "")]
    if len(words) < 3:
        return False
    if any(not char.isascii() for char in text if char.isalpha()) and not any(
        word in _MARKERS["en"] for word in words
    ):
        return True
    return not any(word in _MARKERS["en"] for word in words)


def search_terms_for(text: str, lang: str, *, limit: int = 8) -> list[str]:
    """Search terms with that language's function words removed."""
    if lang == "en":
        return search_terms(text, limit=limit)
    if lang in {"zh", "ja"}:
        # No spaces between words; the search engine segments the sentence.
        return [re.sub(r"[。．！？、,.!?\s]+", " ", text).strip()][:1]
    stop = _MARKERS.get(lang, frozenset())
    terms: list[str] = []
    for word in _TOKEN.findall(text or ""):
        folded = word.casefold()
        if len(folded) < 3 or folded in stop or folded in terms:
            continue
        terms.append(folded)
        if len(terms) >= limit:
            break
    return terms
