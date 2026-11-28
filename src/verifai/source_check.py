"""Read a claim against the very source the paste cites for it.

"Exercise improves cognition in adults over 50 (Northey et al., 2018)"
is checked against Northey et al. itself: its full text where it is open,
else its abstract. The answer is one of: the source says this, the source
contradicts it, the source does not say it, or the source could not be read.
A real paper cited for something it never says is the commonest fault in
AI-written text, and a check that only confirms the paper exists misses it.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

from verifai.citelink import READABLE, Link
from verifai.corroboration import Stance
from verifai.experts.claim_check import _NOT_ABOUT, key_terms
from verifai.models.judge import Judge
from verifai.models.parsing import Judgment
from verifai.schemas import CitationResult, CitationStatus, EvidenceItem, SourceCheck
from verifai.schemas import Link as PageLink
from verifai.source_reader import SourceReader, SourceText, identifiers

_ATTRIBUTION = re.compile(
    r"^\s*(?:according to|as (?:shown|reported|noted|argued|described|found) (?:by|in))\s+"
    r"[^,]{2,90},\s*",
    re.IGNORECASE,
)
_REPORTING = re.compile(
    r"^\s*(?:the\s+(?:study|paper|authors?|review|researchers?)|[A-Z][\w'’-]+"
    r"(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][\w'’-]+)?)\s*(?:\(\s*\d{4}\s*\))?,?\s+"
    r"(?:found|find|finds|showed|shows|show|reported|reports|demonstrated|demonstrates|"
    r"argued|argues|suggested|suggests|noted|notes|stated|states|concluded|concludes|"
    r"observed|observes|claimed|claims|proposed|proposes|established|establishes)\s+(?:that\s+)?",
    re.IGNORECASE,
)
_WINDOW = 1100
# Words of the judge's own phrasing, not of the source.
_REASON_WORDS = frozenset(
    {"evide", "claim", "state", "expli", "direc", "suppo", "contr", "refut", "which", "that"}
)
_PASSAGES = 3
_REFERENCES_HEAD = re.compile(r"^\s*(?:references|bibliography|works cited)\b", re.IGNORECASE)


def claim_content(claim: str) -> str:
    """The claim without who said it: "According to X (2017), Y" is checked as "Y"."""
    content = _REPORTING.sub("", _ATTRIBUTION.sub("", claim), count=1).strip()
    if not content or len(content) < 12:
        return claim
    return content[0].upper() + content[1:]


class SourceChecker:
    def __init__(self, reader: SourceReader, judge: Judge, *, workers: int = 4) -> None:
        self.reader = reader
        self.judge = judge
        self.workers = max(1, workers)

    def check(self, claim: str, links: list[Link]) -> list[SourceCheck]:
        return [self._one(claim, link) for link in links[:3]]

    def _one(self, claim: str, link: Link) -> SourceCheck:
        citation = link.citation
        base = SourceCheck(
            citation=citation.raw,
            title=citation.matched_title,
            url=citation.matched_url,
            cited_as=link.cited_as,
            status="unreadable",
            links=search_links(citation.matched_title or citation.raw),
        )
        if link.document is None and citation.status not in READABLE:
            if citation.status in {CitationStatus.NOT_FOUND, CitationStatus.INVALID}:
                return base.model_copy(
                    update={
                        "status": "missing",
                        "reason": "This cited source could not be found, so nothing backs "
                        "the claim.",
                    }
                )
            return base.model_copy(
                update={
                    "reason": "The cited source could not be reached, so the claim was not "
                    "read against it."
                }
            )
        source = link.document or self.reader.read(citation)
        if source is None:
            return base.model_copy(
                update={
                    "reason": "The source exists, but neither its text nor its abstract is "
                    "open to read."
                }
            )
        base = base.model_copy(
            update={
                "title": citation.matched_title or source.title,
                "url": source.url or citation.matched_url,
                "read": source.read,
                "via": source.via,
            }
        )
        if not self.judge.available():
            return base.model_copy(
                update={
                    "reason": "The source was found, but claim verdicts are paused, so it was "
                    "not read."
                }
            )
        content = claim_content(claim)
        passages = best_passages(content, source)
        judgments = self._judge_all(content, passages)
        return _decide(base, source, passages, judgments, content)

    def _judge_all(self, claim: str, passages: list[str]) -> list[Judgment]:
        def one(passage: str) -> Judgment:
            judged = self.judge.judge(claim, passage[:3500])
            if judged.stance is not Stance.NEUTRAL and _NOT_ABOUT.search(judged.explanation):
                return Judgment(Stance.NEUTRAL, 0.0, judged.explanation)
            return judged

        if len(passages) > 1 and getattr(self.judge, "concurrent", False):
            with ThreadPoolExecutor(max_workers=min(self.workers, len(passages))) as pool:
                return list(pool.map(one, passages))
        return [one(passage) for passage in passages]


def _decide(
    base: SourceCheck,
    source: SourceText,
    passages: list[str],
    judgments: list[Judgment],
    claim: str,
) -> SourceCheck:
    sided = [
        (judgment, passage)
        for judgment, passage in zip(judgments, passages, strict=True)
        if judgment.stance is not Stance.NEUTRAL and judgment.confidence >= 0.6
    ]
    support = [pair for pair in sided if pair[0].stance is Stance.SUPPORTS]
    refute = [pair for pair in sided if pair[0].stance is Stance.REFUTES]
    strongest = max(sided, key=lambda pair: pair[0].confidence, default=None)
    if strongest and (not support or not refute or _clear_winner(support, refute)):
        judgment, passage = strongest
        return base.model_copy(
            update={
                "status": "supports" if judgment.stance is Stance.SUPPORTS else "contradicts",
                "quote": best_sentence(claim, passage, reason=judgment.explanation),
                "reason": judgment.explanation or None,
                "confidence": judgment.confidence,
            }
        )
    scope = {
        "full text": "Its full text was read, and it does not say this.",
        "web page": "The page was read, and it does not say this.",
    }.get(
        source.read,
        "Its abstract does not say this. The full text is not open, so it may say it there.",
    )
    if support and refute:
        scope = "Parts of the source point both ways, so it neither backs nor contradicts this."
    return base.model_copy(update={"status": "not_found", "reason": scope})


def _clear_winner(support: list[tuple[Judgment, str]], refute: list[tuple[Judgment, str]]) -> bool:
    best_support = max(judgment.confidence for judgment, _ in support)
    best_refute = max(judgment.confidence for judgment, _ in refute)
    return abs(best_support - best_refute) >= 0.2


def best_passages(claim: str, source: SourceText, k: int = _PASSAGES) -> list[str]:
    """The abstract, then the windows of the text that share most of the claim's terms."""
    stems = _stems(claim)
    picked: list[str] = []
    if source.abstract:
        picked.append(source.abstract[:2500])
    scored: list[tuple[float, int, str]] = []
    for index, window in enumerate(_windows(source.body)):
        score = _score(window, stems)
        if score > 0:
            scored.append((score, index, window))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score_value, _index, window in scored[:k]:
        picked.append(window)
    if not picked and source.body:
        picked.append(source.body[:_WINDOW])
    return picked


def best_sentence(claim: str, passage: str, *, reason: str = "") -> str:
    """The sentence to quote: the one the judge's reason paraphrases, else the claim's."""
    stems = _stems(claim)
    because = _stems(reason) - _REASON_WORDS
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", passage) if part.strip()]
    if not sentences:
        return passage[:300]
    best = max(
        sentences, key=lambda sentence: 2 * _score(sentence, because) + _score(sentence, stems)
    )
    return best if len(best) <= 400 else best[:397].rsplit(" ", 1)[0] + "…"


def _windows(body: str) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    size = 0
    seen = 0
    for raw in re.split(r"\n+", body):
        paragraph = " ".join(raw.split())
        seen += len(raw) + 1
        if not paragraph:
            continue
        # The reference list at the end is not evidence for the claim. A
        # "References" entry in a contents list near the top is not it.
        if _REFERENCES_HEAD.match(paragraph) and len(paragraph) < 40 and seen > len(body) * 0.6:
            break
        current.append(paragraph)
        size += len(paragraph)
        if size >= _WINDOW:
            windows.append(" ".join(current))
            current, size = [], 0
    if current:
        windows.append(" ".join(current))
    # A very long paragraph is cut so a window stays readable.
    out: list[str] = []
    for window in windows:
        out.extend(
            window[start : start + 2 * _WINDOW] for start in range(0, len(window), 2 * _WINDOW)
        )
    return out


def _stems(claim: str) -> set[str]:
    return {term[:5] if len(term) > 5 else term for term in key_terms(claim)}


def _score(text: str, stems: set[str]) -> float:
    folded = text.casefold()
    hits = 0.0
    for stem in stems:
        found = len(re.findall(rf"\b{re.escape(stem)}", folded))
        if found:
            hits += 2 + min(found, 5) * 0.2
    return hits


def search_links(query: str) -> list[PageLink]:
    """Open the same work in Google Scholar and ResearchGate, where a reader can look further."""
    text = " ".join(query.split())[:200]
    if not text:
        return []
    return [
        PageLink(
            label="Google Scholar",
            url=f"https://scholar.google.com/scholar?q={quote_plus(text)}",
        ),
        PageLink(
            label="ResearchGate",
            url=f"https://www.researchgate.net/search/publication?q={quote_plus(text)}",
        ),
    ]


# How a claim's card names a paper the user attached rather than cited.
ATTACHED = "Your attached PDF"


def as_evidence(check: SourceCheck, citation: CitationResult) -> EvidenceItem | None:
    """A cited source that took a side is one independent source, like any other."""
    if check.status not in {"supports", "contradicts"}:
        return None
    ids = identifiers(citation)
    document_ids = [
        value
        for value in (
            f"doi:{ids.doi}" if ids.doi else None,
            f"arxiv:{ids.arxiv}" if ids.arxiv else None,
            f"pmid:{ids.pmid}" if ids.pmid else None,
        )
        if value
    ]
    return EvidenceItem(
        source="cited_source",
        title=check.title or citation.raw[:120],
        url=check.url,
        snippet=check.quote,
        reason=check.reason,
        score=check.confidence,
        stance=Stance.SUPPORTS.value if check.status == "supports" else Stance.REFUTES.value,
        family=f"cited:{document_ids[0] if document_ids else check.url or citation.raw[:80]}",
        document_ids=document_ids,
        speaker="Your attached PDF" if check.cited_as == ATTACHED else "The cited source",
    )
