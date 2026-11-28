"""The research literature, searched by the claim's terms.

OpenAlex indexes about 250 million works across every field, including what
Google Scholar finds, with open abstracts. arXiv covers preprints in physics,
mathematics, computer science, and statistics. Semantic Scholar is searched
too when an API key is set (without one, its shared budget refuses most
requests). Each paper is its own source; the same paper in two indexes is one.
"""

from __future__ import annotations

import re

from verifai.config import Settings
from verifai.http import PoliteHttp
from verifai.retrieval.textutil import document_ids, fetch_json, search_terms
from verifai.retrieval.types import Passage, RetrievalResult
from verifai.source_reader import abstract_from_index

# Words that state a relation, not a topic. Left in, they narrow a literature
# search to papers that happen to use the same verb.
_RELATION = frozenset(
    [
        "improve",
        "improves",
        "improved",
        "increase",
        "increases",
        "increased",
        "reduce",
        "reduces",
        "reduced",
        "decrease",
        "decreases",
        "decreased",
        "compared",
        "compare",
        "compares",
        "relative",
        "versus",
        "than",
        "leads",
        "lead",
        "cause",
        "causes",
        "caused",
        "causing",
        "show",
        "shows",
        "showed",
        "shown",
        "find",
        "finds",
        "found",
        "suggest",
        "suggests",
        "associated",
        "association",
        "effect",
        "effects",
        "affect",
        "affects",
        "risen",
        "rise",
        "rises",
        "rose",
        "fell",
        "fall",
        "falls",
        "since",
        "about",
        "approximately",
        "around",
        "times",
        "time",
        "long",
        "term",
        "terms",
        "make",
        "makes",
        "made",
        "use",
        "uses",
        "used",
        "using",
        "help",
        "helps",
        "helped",
        "become",
        "becomes",
        "became",
        "work",
        "works",
        "worked",
        "prevent",
        "prevents",
        "prevented",
        "higher",
        "lower",
        "greater",
        "less",
        "more",
        "most",
        "only",
        "all",
        "always",
        "never",
    ]
)


def topic_terms(query: str, *, limit: int = 6) -> list[str]:
    """The claim's subject words, for a literature search."""
    terms = [
        term
        for term in search_terms(query, limit=16)
        if term not in _RELATION and not any(char.isdigit() for char in term)
    ]
    return terms[:limit]


class OpenAlexSearchRetriever:
    name = "openalex_search"

    def __init__(self, http: PoliteHttp, settings: Settings, *, limit: int = 4) -> None:
        self.http = http
        self.settings = settings
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = topic_terms(query)
        if len(terms) < 2:
            return RetrievalResult(self.name)
        params: dict[str, object] = {
            # Title and abstract only: a full-text search spends the shared
            # budget much faster and matches passing mentions.
            "filter": f"title_and_abstract.search:{' '.join(terms)},has_abstract:true",
            "per-page": self.limit,
            "select": "id,doi,display_name,publication_year,abstract_inverted_index,"
            "primary_location,type,cited_by_count",
            "mailto": self.settings.mailto,
        }
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        body, error = fetch_json(
            self.http, "https://api.openalex.org/works", params, source=self.name
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        works = body.get("results")
        if not isinstance(works, list):
            return RetrievalResult(self.name)
        passages = [passage for work in works if (passage := _openalex_passage(work))]
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _openalex_passage(work: object) -> Passage | None:
    if not isinstance(work, dict):
        return None
    title = str(work.get("display_name") or "").strip()
    abstract = abstract_from_index(work.get("abstract_inverted_index"))
    if not title or not abstract:
        return None
    doi = str(work.get("doi") or "").removeprefix("https://doi.org/") or None
    year = work.get("publication_year") if isinstance(work.get("publication_year"), int) else None
    location = work.get("primary_location")
    source = location.get("source") if isinstance(location, dict) else None
    venue = source.get("display_name") if isinstance(source, dict) else None
    url = f"https://doi.org/{doi}" if doi else str(work.get("id") or "") or None
    return Passage(
        source="openalex_search",
        family="openalex",
        title=title,
        url=url,
        text=f"{title}\n{abstract}"[:3500],
        document_ids=document_ids(doi=doi) or (f"openalex:{work.get('id')}",),
        year=year,
        venue=venue if isinstance(venue, str) else None,
        publication_type=str(work.get("type") or "") or None,
    )


class ArxivSearchRetriever:
    name = "arxiv"

    def __init__(self, http: PoliteHttp, *, limit: int = 3) -> None:
        self.http = http
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = topic_terms(query, limit=5)
        if len(terms) < 2:
            return RetrievalResult(self.name)
        search = " AND ".join(f"all:{term}" for term in terms)
        result = self.http.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": search, "max_results": self.limit, "sortBy": "relevance"},
        )
        if not result.ok:
            return RetrievalResult(
                self.name, unavailable=True, error=result.error or f"HTTP {result.status_code}"
            )
        text = result.content.decode("utf-8", errors="replace")
        passages = [
            passage for entry in text.split("<entry>")[1:] if (passage := _arxiv_passage(entry))
        ]
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _arxiv_passage(entry: str) -> Passage | None:
    def field(name: str) -> str:
        found = re.search(rf"<{name}[^>]*>(.*?)</{name}>", entry, re.DOTALL)
        return " ".join(found.group(1).split()) if found else ""

    title, summary, link = field("title"), field("summary"), field("id")
    arxiv_id = re.search(r"abs/(\d{4}\.\d{4,5})", link)
    if not title or not summary or not arxiv_id:
        return None
    published = field("published")
    return Passage(
        source="arxiv",
        family="arxiv",
        title=title,
        url=f"https://arxiv.org/abs/{arxiv_id.group(1)}",
        text=f"{title}\n{summary}"[:3500],
        document_ids=(
            f"arxiv:{arxiv_id.group(1)}",
            f"doi:10.48550/arxiv.{arxiv_id.group(1)}",
        ),
        year=int(published[:4]) if published[:4].isdigit() else None,
        venue="arXiv",
        publication_type="preprint",
    )


class SemanticScholarSearchRetriever:
    name = "semantic_scholar_search"

    def __init__(self, http: PoliteHttp, api_key: str, *, limit: int = 3) -> None:
        self.http = http
        self.api_key = api_key
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = topic_terms(query, limit=8)
        if len(terms) < 2:
            return RetrievalResult(self.name)
        result = self.http.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": " ".join(terms),
                "limit": self.limit,
                "fields": "title,abstract,tldr,year,venue,externalIds,url",
            },
            headers={"x-api-key": self.api_key},
        )
        body = result.json_body() if result.ok else None
        if not isinstance(body, dict):
            return RetrievalResult(
                self.name, unavailable=True, error=result.error or f"HTTP {result.status_code}"
            )
        papers = body.get("data")
        if not isinstance(papers, list):
            return RetrievalResult(self.name)
        passages = [passage for paper in papers if (passage := _s2_passage(paper))]
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _s2_passage(paper: object) -> Passage | None:
    if not isinstance(paper, dict):
        return None
    title = str(paper.get("title") or "").strip()
    abstract = paper.get("abstract")
    tldr = paper.get("tldr")
    summary = abstract if isinstance(abstract, str) and abstract else None
    if summary is None and isinstance(tldr, dict) and isinstance(tldr.get("text"), str):
        summary = tldr["text"]
    if not title or not summary:
        return None
    ids = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
    doi = ids.get("DOI") if isinstance(ids, dict) else None
    pmid = ids.get("PubMed") if isinstance(ids, dict) else None
    year = paper.get("year") if isinstance(paper.get("year"), int) else None
    return Passage(
        source="semantic_scholar_search",
        family="semantic_scholar",
        title=title,
        url=f"https://doi.org/{doi}" if doi else str(paper.get("url") or "") or None,
        text=f"{title}\n{summary}"[:3500],
        document_ids=document_ids(doi=doi, pmid=str(pmid) if pmid else None),
        year=year,
        venue=str(paper.get("venue") or "") or None,
    )
