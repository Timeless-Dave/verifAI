"""PubMed by PMID: does this record exist, and has it been retracted?"""

from __future__ import annotations

from verifai.experts.citation.providers.base import ProviderOutcome, Record
from verifai.experts.citation.textnorm import canonical_doi, surname_from_display
from verifai.http import RETRY_STATUSES, PoliteHttp

_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PubMedIdProvider:
    name = "pubmed"

    def __init__(self, http: PoliteHttp, *, email: str) -> None:
        self.http = http
        self.email = email

    def lookup_pmid(self, pmid: str) -> ProviderOutcome:
        result = self.http.get(
            _ESUMMARY,
            params={
                "db": "pubmed",
                "id": pmid,
                "retmode": "json",
                "tool": "verifai",
                "email": self.email,
            },
        )
        if result.error is not None or result.status_code in RETRY_STATUSES or not result.ok:
            return ProviderOutcome(
                self.name, unavailable=True, error=result.error or f"HTTP {result.status_code}"
            )
        body = result.json_body()
        summary = body.get("result", {}).get(pmid) if isinstance(body, dict) else None
        if not isinstance(summary, dict):
            return ProviderOutcome(self.name, unavailable=True, error="PubMed sent no summary")
        if summary.get("error") or not summary.get("title"):
            return ProviderOutcome(self.name, unavailable=False, records=[])
        return ProviderOutcome(self.name, unavailable=False, records=[_record(pmid, summary)])

    def pmid_for_pmc(self, pmc: str) -> tuple[str | None, bool]:
        """The PMID behind a PMC id, and whether PubMed answered at all."""
        number = pmc.upper().removeprefix("PMC")
        result = self.http.get(
            _ESUMMARY,
            params={
                "db": "pmc",
                "id": number,
                "retmode": "json",
                "tool": "verifai",
                "email": self.email,
            },
        )
        if result.error is not None or not result.ok:
            return None, False
        body = result.json_body()
        summary = body.get("result", {}).get(number) if isinstance(body, dict) else None
        if not isinstance(summary, dict) or summary.get("error"):
            return None, True
        for item in summary.get("articleids") or []:
            if isinstance(item, dict) and item.get("idtype") == "pmid" and item.get("value"):
                return str(item["value"]), True
        return None, True


def _record(pmid: str, summary: dict[str, object]) -> Record:
    authors = [
        str(item.get("name"))
        for item in summary.get("authors") or []  # type: ignore[attr-defined]
        if isinstance(item, dict) and item.get("name")
    ]
    pubdate = str(summary.get("pubdate") or "")
    year = int(pubdate[:4]) if pubdate[:4].isdigit() else None
    doi = next(
        (
            str(item.get("value"))
            for item in summary.get("articleids") or []  # type: ignore[attr-defined]
            if isinstance(item, dict) and item.get("idtype") == "doi"
        ),
        None,
    )
    pubtypes = [str(item) for item in summary.get("pubtype") or []]  # type: ignore[attr-defined]
    retracted = "Retracted Publication" in pubtypes
    notices = ["PubMed lists it as a Retracted Publication."] if retracted else []
    venue = summary.get("fulljournalname") or summary.get("source")
    return Record(
        source="pubmed",
        title=str(summary.get("title")).rstrip("."),
        authors=authors,
        surnames=[surname_from_display(name.split(" ")[0]) for name in authors],
        year=year,
        years=[year] if year else [],
        venue=str(venue) if venue else None,
        doi=canonical_doi(doi) if doi else None,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        record_id=f"pmid:{pmid}",
        notices=notices,
        retracted=retracted,
    )
