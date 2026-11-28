"""Read the text of a cited work: full text where it is open, else its abstract.

The order follows what is free and dependable:

- arXiv: the paper's HTML full text, else its abstract.
- PubMed Central: the open-access full text from Europe PMC.
- A DOI: OpenAlex for the abstract and any open-access copy (a PDF is read
  with pypdf), then Semantic Scholar (with a key) and Crossref for the abstract.
- A PMID: the Europe PMC record, and its full text when there is one.
- A link: the page itself, HTML or PDF.

Google Scholar and ResearchGate are not read: neither has a public API, and
both refuse automated requests. OpenAlex and Semantic Scholar index the same
literature.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from rapidfuzz import fuzz

from verifai.config import Settings
from verifai.http import PoliteHttp
from verifai.net import public_url_problem
from verifai.retrieval.textutil import html_to_text
from verifai.schemas import CitationResult

_FULL_TEXT_LIMIT = 400_000
_PDF_PAGES = 40
_ARXIV_ID = re.compile(r"(?:arxiv[:./\s]*|abs/|html/|pdf/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_PMCID = re.compile(r"\bPMC\d{4,9}\b", re.IGNORECASE)
_PMID = re.compile(r"\bPMID:?\s*(\d{4,9})\b|pubmed\.ncbi\.nlm\.nih\.gov/(\d{4,9})", re.IGNORECASE)


@dataclass(frozen=True)
class SourceText:
    title: str | None
    url: str | None
    # "full text", "abstract", or "web page": how much of the work was read.
    read: str
    abstract: str
    body: str
    # Where the text came from: "arXiv", "Europe PMC", "OpenAlex", …
    via: str


@dataclass(frozen=True)
class Identifiers:
    doi: str | None = None
    arxiv: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    url: str | None = None


def identifiers(citation: CitationResult) -> Identifiers:
    """Every handle on the cited work the check found or the paste gave."""
    texts = [citation.raw, citation.matched_url or ""]
    texts.extend(item.url or "" for item in citation.evidence)
    joined = " ".join(texts)
    doi = next((item.doi for item in citation.evidence if item.doi), None)
    if doi is None and (found := _DOI.search(joined)):
        doi = found.group(1)
    if doi:
        doi = doi.strip().rstrip(".,;)").removeprefix("doi:").lower()
    arxiv_match = _ARXIV_ID.search(joined) or (
        re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi or "", re.IGNORECASE)
    )
    pmcid = _PMCID.search(joined)
    pmid_match = _PMID.search(joined)
    pmid = (pmid_match.group(1) or pmid_match.group(2)) if pmid_match else None
    if pmid is None and citation.kind == "pmid" and citation.raw.strip().isdigit():
        pmid = citation.raw.strip()
    url = None
    if citation.kind == "url":
        url = citation.raw.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
    return Identifiers(
        doi=doi,
        arxiv=arxiv_match.group(1) if arxiv_match else None,
        pmid=pmid,
        pmcid=pmcid.group(0).upper() if pmcid else None,
        url=url,
    )


class SourceReader:
    def __init__(self, http: PoliteHttp, settings: Settings) -> None:
        self.http = http
        self.settings = settings

    def read(self, citation: CitationResult) -> SourceText | None:
        ids = identifiers(citation)
        title = citation.matched_title
        if ids.arxiv:
            found = self._arxiv(ids.arxiv, title)
            if found:
                return found
        if ids.pmcid:
            found = self._europe_pmc_full(ids.pmcid, title, abstract="", url=None)
            if found:
                return found
        if ids.doi:
            found = self._doi(ids.doi, title)
            if found:
                return found
        if ids.pmid:
            found = self._pmid(ids.pmid, title)
            if found:
                return found
        if ids.url and not _is_doi_link(ids.url):
            return self._page(ids.url, title)
        if title and not (ids.doi or ids.arxiv or ids.pmid or ids.pmcid):
            # A journal without DOIs (JMLR, many conference papers): find it by title.
            work = self._work_by_title(title)
            if work is not None:
                return self._from_work(work, title, url=citation.matched_url)
        return None

    def _work_by_title(self, title: str) -> dict[str, object] | None:
        params: dict[str, object] = {
            "filter": "title.search:" + re.sub(r"[,:|]", " ", title)[:200],
            "per-page": 3,
            "mailto": self.settings.mailto,
        }
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        result = self.http.get("https://api.openalex.org/works", params=params)
        body = result.json_body() if result.ok else None
        works = body.get("results") if isinstance(body, dict) else None
        if not isinstance(works, list):
            return None
        wanted = _plain(title)
        for work in works:
            if (
                isinstance(work, dict)
                and fuzz.ratio(_plain(str(work.get("display_name") or "")), wanted) >= 90
            ):
                return work
        return None

    # arXiv ---------------------------------------------------------------

    def _arxiv(self, arxiv_id: str, title: str | None) -> SourceText | None:
        abstract, found_title = self._arxiv_abstract(arxiv_id)
        title = title or found_title
        page = f"https://arxiv.org/abs/{arxiv_id}"
        html = self.http.get(f"https://arxiv.org/html/{arxiv_id}")
        if html.ok and b"<html" in html.content[:2000].lower():
            body = html_to_text(html.content, limit=_FULL_TEXT_LIMIT)
            if len(body) > 3000:
                return SourceText(title, page, "full text", abstract, body, "arXiv")
        if abstract:
            return SourceText(title, page, "abstract", abstract, "", "arXiv")
        return None

    def _arxiv_abstract(self, arxiv_id: str) -> tuple[str, str | None]:
        result = self.http.get(
            "https://export.arxiv.org/api/query", params={"id_list": arxiv_id, "max_results": 1}
        )
        if not result.ok:
            return "", None
        text = result.content.decode("utf-8", errors="replace")
        entry = text.split("<entry>", 1)[-1] if "<entry>" in text else ""
        summary = re.search(r"<summary[^>]*>(.*?)</summary>", entry, re.DOTALL)
        heading = re.search(r"<title[^>]*>(.*?)</title>", entry, re.DOTALL)
        return (
            " ".join(summary.group(1).split()) if summary else "",
            " ".join(heading.group(1).split()) if heading else None,
        )

    # PubMed Central and Europe PMC ----------------------------------------

    def _europe_pmc_full(
        self, pmcid: str, title: str | None, *, abstract: str, url: str | None
    ) -> SourceText | None:
        result = self.http.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        )
        if not result.ok or not result.content:
            return None
        body = html_to_text(result.content, limit=_FULL_TEXT_LIMIT)
        if len(body) < 2000:
            return None
        page = url or f"https://europepmc.org/article/PMC/{pmcid}"
        return SourceText(title, page, "full text", abstract, body, "Europe PMC")

    def _pmid(self, pmid: str, title: str | None) -> SourceText | None:
        result = self.http.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"EXT_ID:{pmid} AND SRC:MED", "resultType": "core", "format": "json"},
        )
        body = result.json_body() if result.ok else None
        hits = (body.get("resultList") or {}).get("result") if isinstance(body, dict) else None
        if not isinstance(hits, list) or not hits or not isinstance(hits[0], dict):
            return None
        hit = hits[0]
        abstract = _strip_tags(str(hit.get("abstractText") or ""))
        title = title or str(hit.get("title") or "") or None
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        pmcid = hit.get("pmcid")
        if isinstance(pmcid, str) and pmcid:
            full = self._europe_pmc_full(pmcid, title, abstract=abstract, url=url)
            if full:
                return full
        if abstract:
            return SourceText(title, url, "abstract", abstract, "", "Europe PMC")
        return None

    # DOI ----------------------------------------------------------------

    def _doi(self, doi: str, title: str | None) -> SourceText | None:
        params: dict[str, str] = {"mailto": self.settings.mailto}
        if self.settings.openalex_api_key:
            params["api_key"] = self.settings.openalex_api_key
        result = self.http.get(f"https://api.openalex.org/works/doi:{doi}", params=params)
        work = result.json_body() if result.ok else None
        return self._from_work(
            work if isinstance(work, dict) else None, title, url=f"https://doi.org/{doi}", doi=doi
        )

    def _from_work(
        self,
        work: dict[str, object] | None,
        title: str | None,
        *,
        url: str | None,
        doi: str | None = None,
    ) -> SourceText | None:
        """Full text, or the abstract, of an OpenAlex work."""
        abstract = ""
        oa_pdf: str | None = None
        oa_page: str | None = None
        if isinstance(work, dict):
            abstract = abstract_from_index(work.get("abstract_inverted_index"))
            name = work.get("display_name")
            title = title or (name if isinstance(name, str) else None)
            ids = work.get("ids")
            pmcid = str(ids.get("pmcid") or "" if isinstance(ids, dict) else "").rsplit("/", 1)[-1]
            if pmcid.upper().startswith("PMC"):
                full = self._europe_pmc_full(pmcid.upper(), title, abstract=abstract, url=url)
                if full:
                    return full
            best = work.get("best_oa_location")
            if isinstance(best, dict):
                oa_pdf = best.get("pdf_url") if isinstance(best.get("pdf_url"), str) else None
                landing = best.get("landing_page_url")
                oa_page = landing if isinstance(landing, str) else None
        if oa_pdf:
            text = self._pdf(oa_pdf)
            if text:
                return SourceText(title, url, "full text", abstract, text, "an open-access PDF")
        if not abstract and doi:
            abstract = self._semantic_scholar_abstract(doi) or self._crossref_abstract(doi)
        if oa_page and not _is_doi_link(oa_page):
            page = self._fetch_page(oa_page, title)
            if page and len(page.body) > 6000:
                return SourceText(
                    title, url, "full text", abstract, page.body, "an open-access copy"
                )
        if abstract:
            return SourceText(title, url, "abstract", abstract, "", "OpenAlex")
        return None

    def _semantic_scholar_abstract(self, doi: str) -> str:
        if not self.settings.semantic_scholar_api_key:
            return ""
        result = self.http.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='/')}",
            params={"fields": "abstract,tldr"},
            headers={"x-api-key": self.settings.semantic_scholar_api_key},
        )
        body = result.json_body() if result.ok else None
        if not isinstance(body, dict):
            return ""
        abstract = body.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            return abstract.strip()
        tldr = body.get("tldr")
        text = tldr.get("text") if isinstance(tldr, dict) else None
        return text.strip() if isinstance(text, str) else ""

    def _crossref_abstract(self, doi: str) -> str:
        result = self.http.get(f"https://api.crossref.org/works/{doi}")
        body = result.json_body() if result.ok else None
        message = body.get("message") if isinstance(body, dict) else None
        abstract = message.get("abstract") if isinstance(message, dict) else None
        return _strip_tags(abstract) if isinstance(abstract, str) else ""

    # Pages and PDFs -------------------------------------------------------

    def _page(self, url: str, title: str | None, *, archived: bool = False) -> SourceText | None:
        found = self._fetch_page(url, title)
        if found is not None or archived:
            return found
        # A site that refuses automated readers, or answers with a bot check,
        # often has a copy in the Internet Archive. "id_" asks for the page as
        # it was served, without the archive's own banner.
        copy = self._fetch_page(f"https://web.archive.org/web/2id_/{url}", title)
        if copy is None:
            return None
        return SourceText(
            copy.title, url, copy.read, copy.abstract, copy.body, "the Internet Archive's copy"
        )

    def _fetch_page(self, url: str, title: str | None) -> SourceText | None:
        if public_url_problem(url):
            return None
        result = self.http.get(url, headers={"Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5"})
        if not result.ok or not result.content:
            return None
        kind = result.headers.get("content-type", "")
        if "pdf" in kind or result.content[:5] == b"%PDF-":
            text = _pdf_text(result.content)
            return SourceText(title, url, "full text", "", text, "the linked PDF") if text else None
        body = html_to_text(result.content, limit=_FULL_TEXT_LIMIT)
        if len(body) < 400:
            return None  # a bot check or a script-only shell, not the page
        host = (urlsplit(url).hostname or "").removeprefix("www.")
        return SourceText(title or body.split("\n", 1)[0][:140], url, "web page", "", body, host)

    def _pdf(self, url: str) -> str:
        if public_url_problem(url):
            return ""
        result = self.http.get(url, headers={"Accept": "application/pdf"})
        if not result.ok or result.content[:5] != b"%PDF-":
            return ""
        return _pdf_text(result.content)


def abstract_from_index(index: object) -> str:
    """OpenAlex stores an abstract as {word: [positions]}; put the words back in order."""
    if not isinstance(index, dict):
        return ""
    placed: dict[int, str] = {}
    for word, positions in index.items():
        if isinstance(positions, list):
            for position in positions:
                if isinstance(position, int):
                    placed[position] = str(word)
    return " ".join(placed[key] for key in sorted(placed))


def _pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages[:_PDF_PAGES]]
    except (PdfReadError, ValueError, KeyError, TypeError, OSError):
        return ""
    text = "\n".join(pages)
    # PDF lines break mid-sentence; keep paragraph breaks only.
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"(?<![.!?:])\n(?!\n)", " ", text)
    return text[:_FULL_TEXT_LIMIT] if len(text) > 1500 else ""


def _strip_tags(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _is_doi_link(url: str) -> bool:
    return (urlsplit(url).hostname or "").removeprefix("www.").endswith("doi.org")


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
