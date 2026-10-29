"""PubMed via NCBI E-utilities. Systematic reviews are requested first."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import document_ids, search_terms, year_of
from verifai.retrieval.types import Passage, RetrievalResult

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_REVIEW = "(systematic review[pt] OR meta-analysis[pt])"


class PubMedRetriever:
    name = "pubmed"

    def __init__(self, http: PoliteHttp, *, email: str, limit: int = 4) -> None:
        self.http = http
        self.email = email
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        terms = " ".join(search_terms(query, limit=10))
        if not terms:
            return RetrievalResult(self.name)
        review_ids, error = self._search(f"({terms}) AND {_REVIEW}")
        if error:
            return RetrievalResult(self.name, unavailable=True, error=error)
        kind = "systematic review" if review_ids else None
        ids = review_ids
        if not ids:
            ids, error = self._search(terms)
            if error:
                return RetrievalResult(self.name, unavailable=True, error=error)
        if not ids:
            return RetrievalResult(self.name)
        passages, error = self._fetch(ids[: self.limit], kind)
        if error:
            return RetrievalResult(self.name, unavailable=True, error=error)
        return RetrievalResult(self.name, passages=passages)

    def _search(self, term: str) -> tuple[list[str], str | None]:
        result = self.http.get(
            _ESEARCH,
            params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": self.limit,
                "sort": "relevance",
                "term": term,
                "tool": "verifai",
                "email": self.email,
            },
        )
        if result.error or result.status_code != 200:
            return [], result.error or f"HTTP {result.status_code}"
        body = result.json_body()
        if not isinstance(body, dict):
            return [], "PubMed search was not JSON"
        raw = body.get("esearchresult", {})
        id_list = raw.get("idlist") if isinstance(raw, dict) else None
        if not isinstance(id_list, list):
            return [], "PubMed search had no id list"
        return [str(item) for item in id_list if str(item).isdigit()], None

    def _fetch(self, ids: list[str], kind: str | None) -> tuple[list[Passage], str | None]:
        result = self.http.get(
            _EFETCH,
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
                "tool": "verifai",
                "email": self.email,
            },
        )
        if result.error or result.status_code != 200:
            return [], result.error or f"HTTP {result.status_code}"
        try:
            root = ET.fromstring(result.content)
        except ET.ParseError:
            return [], "PubMed abstracts were not XML"
        passages = [_article(node, kind) for node in root.iter("PubmedArticle")]
        return [item for item in passages if item is not None], None


def _article(node: ET.Element, kind: str | None) -> Passage | None:
    title = _text(node.find("./MedlineCitation/Article/ArticleTitle"))
    abstract = " ".join(
        piece.strip()
        for piece in (
            item.text or ""
            for item in node.findall("./MedlineCitation/Article/Abstract/AbstractText")
        )
        if piece.strip()
    )
    if not title and not abstract:
        return None
    pmid = _text(node.find("./MedlineCitation/PMID"))
    doi = ""
    for article_id in node.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi" and article_id.text:
            doi = article_id.text.strip()
    authors = tuple(
        " ".join(
            part
            for part in (
                _text(author.find("LastName")),
                _text(author.find("Initials")),
            )
            if part
        )
        for author in node.findall("./MedlineCitation/Article/AuthorList/Author")
    )
    authors = tuple(author for author in authors if author)
    venue = _text(node.find("./MedlineCitation/Article/Journal/Title"))
    year = year_of(_text(node.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")))
    types = [
        item.text.strip()
        for item in node.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
        if item.text
    ]
    publication = kind
    if any(
        "meta-analysis" in item.casefold() or "systematic review" in item.casefold()
        for item in types
    ):
        publication = "systematic review"
    pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
    return Passage(
        source="pubmed",
        family="pubmed",
        title=title or f"PMID {pmid}",
        url=pmid_url,
        text=abstract or title,
        document_ids=document_ids(doi=doi or None, pmid=pmid or None),
        authors=authors,
        year=year,
        venue=venue or None,
        publication_type=publication,
    )


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())
