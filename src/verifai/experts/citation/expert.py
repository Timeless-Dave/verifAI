"""Citation expert.

DOIs are resolved at doi.org and cross-checked with OpenAlex. If doi.org is
down, the Crossref REST API fills that slot. Free-text references are
searched in Crossref, OpenAlex, and DataCite. DataCite is where many
preprints are registered when the other two indexes only return a later
reprint. URLs get a direct fetch, then the Wayback Machine when the fetch
does not confirm the page.

Semantic Scholar is included only when an API key is configured. It never
replaces the Crossref and OpenAlex pair.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import cast

from verifai.config import Settings
from verifai.evidence.base import EvidenceProvider, EvidenceRegistry
from verifai.experts.citation.decide import (
    decide_doi,
    decide_invalid,
    decide_isbn,
    decide_pmid,
    decide_reference,
    decide_url,
)
from verifai.experts.citation.extract import (
    ExtractedCitation,
    clean_doi,
    dangling_markers,
    extract_citations,
    isbn_valid,
    unresolvable_mentions,
)
from verifai.experts.citation.providers.base import ProviderOutcome
from verifai.experts.citation.providers.crossref import CrossrefProvider
from verifai.experts.citation.providers.datacite import DataCiteProvider
from verifai.experts.citation.providers.doi_org import DoiOrgProvider
from verifai.experts.citation.providers.openalex import OpenAlexProvider
from verifai.experts.citation.providers.openlibrary import OpenLibraryProvider
from verifai.experts.citation.providers.pubmed_id import PubMedIdProvider
from verifai.experts.citation.providers.semantic_scholar import SemanticScholarProvider
from verifai.experts.citation.providers.url_check import UrlLivenessProvider
from verifai.experts.citation.providers.wayback import WaybackProvider
from verifai.http import PoliteHttp
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    ExpertResult,
    Verdict,
    ordered_sources,
    source_label,
)

log = logging.getLogger(__name__)

MAX_CITATIONS = 25
_REQUIRED = ("doi.org", "crossref", "datacite", "openalex", "url", "wayback")


class CitationExpert:
    """Check whether citations in pasted text refer to real works or pages.

    The verdict is about the citations, not about whether those works support
    the paragraph around them. See the ``Expert`` protocol for the rule that
    no provider is the sole decider.
    """

    name = "citation"
    description = (
        "Checks whether DOIs, URLs, and bibliographic references exist "
        "by comparing them with records retrieved from independent sources."
    )

    def __init__(self, providers: list[EvidenceProvider], *, max_concurrency: int = 4) -> None:
        self.registry = EvidenceRegistry()
        self._providers: dict[str, object] = {}
        for provider in providers:
            self.registry.register(provider)
            self._providers[provider.name] = provider
        missing = [name for name in _REQUIRED if name not in self._providers]
        if missing:
            raise ValueError(
                "Citation checking requires doi.org, Crossref, DataCite, OpenAlex, "
                "a direct URL check, and the Wayback Machine, so no one source is "
                f"the sole decider. Missing: {', '.join(missing)}."
            )
        self.max_concurrency = max(1, max_concurrency)

    def check(self, text: str) -> ExpertResult:
        extracted = extract_citations(text or "")
        dangling = dangling_markers(text or "") + unresolvable_mentions(text or "")
        truncated = len(extracted) > MAX_CITATIONS
        extracted = extracted[:MAX_CITATIONS]
        if not extracted:
            return _summarize([], truncated=False, dangling=dangling)
        workers = min(self.max_concurrency, len(extracted))
        if workers == 1:
            checked = [self._check_one(item) for item in extracted]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                checked = list(pool.map(self._check_one, extracted))
        for item in checked:
            if item.sources_unavailable:
                log.info(
                    "citation %s unavailable: %s",
                    item.kind,
                    ", ".join(item.sources_unavailable),
                )
        return _summarize(_merge_same_paper(checked), truncated=truncated, dangling=dangling)

    def _check_one(self, citation: ExtractedCitation) -> CitationResult:
        if citation.problem:
            return decide_invalid(citation)
        if citation.kind == "pmc" and citation.pmc:
            pmid, answered = self._pubmed().pmid_for_pmc(citation.pmc)
            if pmid is None:
                outcome = ProviderOutcome("pubmed", unavailable=not answered)
                return decide_pmid(replace(citation, pmid=citation.pmc), outcome)
            return decide_pmid(replace(citation, pmid=pmid), self._pubmed().lookup_pmid(pmid))
        if citation.kind == "doi" and citation.doi:
            return self._check_doi(citation)
        if citation.kind == "url" and citation.url:
            return self._check_url(citation)
        if citation.kind == "isbn" and citation.isbn:
            if not isbn_valid(citation.isbn):
                return decide_isbn(citation, None)
            return decide_isbn(citation, self._openlibrary().lookup_isbn(citation.isbn))
        if citation.kind == "pmid" and citation.pmid:
            return decide_pmid(citation, self._pubmed().lookup_pmid(citation.pmid))
        return self._check_reference(citation)

    def _check_doi(self, citation: ExtractedCitation) -> CitationResult:
        assert citation.doi is not None
        outcomes: list[ProviderOutcome] = [self._doi().lookup_doi(citation.doi)]
        if outcomes[0].unavailable:
            outcomes.append(self._crossref().lookup_doi(citation.doi))
        outcomes.append(self._openalex().lookup_doi(citation.doi))
        scholar = self._scholar()
        if scholar is not None:
            outcomes.append(scholar.lookup_doi(citation.doi))
        return decide_doi(citation, outcomes)

    def _check_reference(self, citation: ExtractedCitation) -> CitationResult:
        title = citation.title
        authors = list(citation.surnames)
        year = citation.year
        venue = citation.venue
        raw = citation.raw

        def run(search: Callable[..., ProviderOutcome]) -> ProviderOutcome:
            return search(title=title, authors=authors, year=year, venue=venue, raw=raw)

        searches: list[Callable[..., ProviderOutcome]] = [
            self._crossref().search,
            self._openalex().search,
            self._datacite().search,
        ]
        scholar = self._scholar()
        if scholar is not None:
            searches.append(scholar.search)
        # Ask every index at once. A slow DataCite response must not wait
        # until Crossref and OpenAlex have already finished, or a short
        # timeout will discard the year-matching preprint.
        if len(searches) == 1:
            outcomes = [run(searches[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(searches)) as pool:
                outcomes = list(pool.map(run, searches))
        for outcome in outcomes:
            if outcome.unavailable and outcome.error:
                log.info("provider %s unavailable: %s", outcome.source, outcome.error)
        return decide_reference(citation, outcomes)

    def _check_url(self, citation: ExtractedCitation) -> CitationResult:
        assert citation.url is not None
        live = self._url().check(citation.url)
        confirmed = (
            live.blocked_reason is None
            and live.error is None
            and live.status_code is not None
            and 200 <= live.status_code < 400
        )
        wayback = None if confirmed or live.blocked_reason else self._wayback().lookup(citation.url)
        if not confirmed:
            return decide_url(citation, live, wayback)
        meta = self._url().page_meta(live.final_url or citation.url)
        if meta is not None and meta.doi:
            # The page names the work it holds. Check that work like any DOI,
            # retraction included, so a live link is not taken on trust.
            as_doi = replace(citation, doi=clean_doi(meta.doi))
            result = self._check_doi(as_doi)
            return result.model_copy(
                update={
                    "explanation": (
                        f"The page at {citation.url} identifies itself as DOI {as_doi.doi}. "
                        + result.explanation
                    ),
                    "matched_url": live.final_url or citation.url,
                    "sources_consulted": ordered_sources(["url", *result.sources_consulted]),
                    "sources_contributed": ordered_sources(["url", *result.sources_contributed]),
                }
            )
        return decide_url(citation, live, wayback, page_title=meta.title if meta else None)

    def _doi(self) -> DoiOrgProvider:
        return cast("DoiOrgProvider", self._providers["doi.org"])

    def _crossref(self) -> CrossrefProvider:
        return cast("CrossrefProvider", self._providers["crossref"])

    def _openalex(self) -> OpenAlexProvider:
        return cast("OpenAlexProvider", self._providers["openalex"])

    def _datacite(self) -> DataCiteProvider:
        return cast("DataCiteProvider", self._providers["datacite"])

    def _url(self) -> UrlLivenessProvider:
        return cast("UrlLivenessProvider", self._providers["url"])

    def _wayback(self) -> WaybackProvider:
        return cast("WaybackProvider", self._providers["wayback"])

    def _pubmed(self) -> PubMedIdProvider:
        return cast("PubMedIdProvider", self._providers["pubmed"])

    def _openlibrary(self) -> OpenLibraryProvider:
        return cast("OpenLibraryProvider", self._providers["openlibrary"])

    def _scholar(self) -> SemanticScholarProvider | None:
        provider = self._providers.get("semantic_scholar")
        if provider is None:
            return None
        return cast("SemanticScholarProvider", provider)


def build_citation_expert(http: PoliteHttp, settings: Settings) -> CitationExpert:
    providers: list[EvidenceProvider] = [
        DoiOrgProvider(http),
        CrossrefProvider(http),
        DataCiteProvider(http),
        OpenAlexProvider(http, mailto=settings.mailto, api_key=settings.openalex_api_key),
        UrlLivenessProvider(http),
        WaybackProvider(http),
        OpenLibraryProvider(http),
        PubMedIdProvider(http, email=settings.mailto),
    ]
    if settings.semantic_scholar_api_key:
        providers.append(SemanticScholarProvider(http, api_key=settings.semantic_scholar_api_key))
    return CitationExpert(providers, max_concurrency=settings.max_concurrency)


def _summarize(
    citations: list[CitationResult], *, truncated: bool, dangling: list[str] | None = None
) -> ExpertResult:
    dangling_note = ""
    if dangling:
        marks = ", ".join(dangling)
        dangling_note = (
            f" The text points to {marks}, with no reference or title to look up, "
            "so there is nothing to check."
        )
    if not citations:
        return ExpertResult(
            expert="citation",
            verdict=Verdict.UNVERIFIABLE,
            verdict_scope="citations",
            confidence=0.0,
            explanation=(
                "No DOIs, URLs, or bibliographic references were found. "
                "A bare author-year mention such as “(Smith et al., 2021)” is skipped, "
                "because it is not specific enough to search without calling a real paper fabricated. "
                "No providers were queried."
            )
            + dangling_note,
            evidence=[],
            sources_consulted=[],
            sources_contributed=[],
            sources_unavailable=[],
            citations=[],
        )

    counts: Counter[CitationStatus] = Counter(item.status for item in citations)
    total = len(citations)
    verdict, lead = _verdict(counts, total)
    pieces = [
        (
            f"Checked {total} citation{'s' if total != 1 else ''}: "
            f"{counts[CitationStatus.VERIFIED]} verified, "
            f"{counts[CitationStatus.PARTIALLY_MATCHES]} partially matching, "
            f"{counts[CitationStatus.NOT_FOUND]} not found, "
            f"{counts[CitationStatus.RETRACTED]} retracted, "
            f"{counts[CitationStatus.INVALID]} invalid, "
            f"{counts[CitationStatus.WRONG_WORK]} pointing to a different paper, "
            f"{counts[CitationStatus.UNREACHABLE]} unreachable."
        )
    ]
    if dangling_note:
        pieces.append(dangling_note.strip())
    if truncated:
        pieces.append(f"Only the first {MAX_CITATIONS} citations were checked.")
    pieces.append(lead)
    unavailable = ordered_sources(name for item in citations for name in item.sources_unavailable)
    if unavailable:
        labels = ", ".join(source_label(name) for name in unavailable)
        pieces.append(
            f"Unavailable for at least one citation: {labels}. "
            "Those checks use only the sources that answered, at lower confidence."
        )
    pieces.append(
        "This check only asks whether each citation exists and matches the details "
        "printed with it. It does not decide whether the source supports the surrounding claim."
    )
    confidence = round(sum(item.confidence for item in citations) / total, 2)
    evidence = [entry for item in citations for entry in item.evidence]
    return ExpertResult(
        expert="citation",
        verdict=verdict,
        verdict_scope="citations",
        confidence=confidence,
        explanation=" ".join(pieces),
        evidence=evidence,
        sources_consulted=ordered_sources(
            name for item in citations for name in item.sources_consulted
        ),
        sources_contributed=ordered_sources(
            name for item in citations for name in item.sources_contributed
        ),
        sources_unavailable=unavailable,
        citations=citations,
    )


def _verdict(counts: Counter[CitationStatus], total: int) -> tuple[Verdict, str]:
    if counts[CitationStatus.VERIFIED] == total:
        return Verdict.SUPPORTED, "Every citation matched a retrieved record."
    wrong = counts[CitationStatus.INVALID] + counts[CitationStatus.WRONG_WORK]
    if counts[CitationStatus.NOT_FOUND] + wrong == total:
        return Verdict.UNSUPPORTED, "None of the citations matched a retrieved record."
    if counts[CitationStatus.UNREACHABLE] == total:
        return (
            Verdict.UNVERIFIABLE,
            "None of the checks could be completed because the sources did not answer.",
        )
    if counts[CitationStatus.RETRACTED] == total:
        return Verdict.UNSUPPORTED, "Every citation points to a retracted work."
    if counts[CitationStatus.NOT_FOUND] or wrong:
        return (
            Verdict.MIXED,
            (
                "At least one citation looks fabricated, points to a different paper, "
                "or could not be found."
            ),
        )
    if counts[CitationStatus.RETRACTED]:
        return Verdict.MIXED, "At least one citation points to a retracted work."
    return (
        Verdict.MIXED,
        "Some citations only partially matched, or could not be reached.",
    )


def _merge_same_paper(results: list[CitationResult]) -> list[CitationResult]:
    """A PMID and a PMC id for the same paper are one citation, cited twice."""
    kept: list[CitationResult] = []
    by_record: dict[str, int] = {}
    for item in results:
        key = item.matched_url if item.kind in {"pmid", "pmc"} and item.matched_url else None
        if key is None or key not in by_record:
            if key is not None:
                by_record[key] = len(kept)
            kept.append(item)
            continue
        first = kept[by_record[key]]
        kept[by_record[key]] = first.model_copy(
            update={"occurrences": first.occurrences + item.occurrences}
        )
    return kept
