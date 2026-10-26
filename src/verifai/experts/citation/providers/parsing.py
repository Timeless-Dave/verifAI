"""Turn Crossref-like JSON and OpenAlex works into records."""

from __future__ import annotations

from typing import Any

from verifai.experts.citation.providers.base import Record
from verifai.experts.citation.textnorm import canonical_doi, surname_from_display


def record_from_crossref_like(source: str, payload: object) -> Record | None:
    if not isinstance(payload, dict):
        return None
    title = _first_str(payload.get("title"))
    doi = _first_str(payload.get("DOI") or payload.get("doi"))
    if not title and not doi:
        return None
    authors, surnames = _crossref_authors(payload.get("author") or payload.get("authors"))
    years = _years(payload)
    venue = _first_str(payload.get("container-title") or payload.get("container_title"))
    if venue is None:
        venue = _first_str(payload.get("short-container-title"))
    url = _first_str(payload.get("URL") or payload.get("url"))
    if url is None and doi:
        url = f"https://doi.org/{doi}"
    return Record(
        source=source,
        title=title,
        authors=authors,
        surnames=surnames,
        year=years[0] if years else None,
        years=years,
        venue=venue,
        doi=canonical_doi(doi),
        url=url,
        record_id=canonical_doi(doi),
    )


def record_from_openalex(work: object) -> Record | None:
    if not isinstance(work, dict):
        return None
    title = work.get("display_name") or work.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    authors: list[str] = []
    surnames: list[str] = []
    for authorship in work.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        raw = authorship.get("raw_author_name") or ""
        if not raw and isinstance(authorship.get("author"), dict):
            raw = authorship["author"].get("display_name") or ""
        raw = str(raw).strip()
        if not raw:
            continue
        authors.append(raw)
        surnames.append(surname_from_display(raw))
    year = work.get("publication_year")
    if not isinstance(year, int):
        year = None
    venue = None
    location = work.get("primary_location") or {}
    if isinstance(location, dict):
        source = location.get("source") or {}
        if isinstance(source, dict) and isinstance(source.get("display_name"), str):
            venue = source["display_name"]
    doi = work.get("doi") if isinstance(work.get("doi"), str) else None
    url = doi if doi and doi.startswith("http") else (f"https://doi.org/{doi}" if doi else None)
    if url is None and isinstance(work.get("id"), str):
        url = work["id"]
    return Record(
        source="openalex",
        title=title.strip(),
        authors=authors,
        surnames=surnames,
        year=year,
        years=[year] if year else [],
        venue=venue,
        doi=canonical_doi(doi),
        url=url,
        record_id=work.get("id") if isinstance(work.get("id"), str) else None,
    )


def record_from_semantic_scholar(paper: object) -> Record | None:
    if not isinstance(paper, dict):
        return None
    title = paper.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    authors: list[str] = []
    surnames: list[str] = []
    for author in paper.get("authors") or []:
        if not isinstance(author, dict):
            continue
        name = author.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        authors.append(name.strip())
        surnames.append(surname_from_display(name))
    year = paper.get("year") if isinstance(paper.get("year"), int) else None
    venue = (
        paper.get("venue") if isinstance(paper.get("venue"), str) and paper.get("venue") else None
    )
    raw_external = paper.get("externalIds")
    external = raw_external if isinstance(raw_external, dict) else {}
    raw_doi = external.get("DOI")
    doi = raw_doi if isinstance(raw_doi, str) else None
    url = paper.get("url") if isinstance(paper.get("url"), str) else None
    if url is None and doi:
        url = f"https://doi.org/{doi}"
    return Record(
        source="semantic_scholar",
        title=title.strip(),
        authors=authors,
        surnames=surnames,
        year=year,
        years=[year] if year else [],
        venue=venue,
        doi=canonical_doi(doi),
        url=url,
        record_id=paper.get("paperId") if isinstance(paper.get("paperId"), str) else None,
    )


def record_from_datacite(item: object) -> Record | None:
    """Parse one DataCite JSON:API resource.

    Venue comes from ``container.title`` only. The publisher field is often
    the repository (``arXiv``) and is not the venue a citation names.
    """
    if not isinstance(item, dict):
        return None
    payload = item.get("attributes") if isinstance(item.get("attributes"), dict) else item
    if not isinstance(payload, dict):
        return None
    title = None
    titles = payload.get("titles")
    if isinstance(titles, list):
        for entry in titles:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("title"), str)
                and entry["title"].strip()
            ):
                title = entry["title"].strip()
                break
    doi = payload.get("doi") if isinstance(payload.get("doi"), str) else None
    if not title and not doi:
        return None
    authors, surnames = _datacite_creators(payload.get("creators"))
    year = _datacite_year(payload.get("publicationYear"))
    venue = None
    container = payload.get("container")
    if (
        isinstance(container, dict)
        and isinstance(container.get("title"), str)
        and container["title"].strip()
    ):
        venue = container["title"].strip()
    url = (
        payload.get("url")
        if isinstance(payload.get("url"), str) and payload["url"].strip()
        else None
    )
    if url is None and doi:
        url = f"https://doi.org/{doi}"
    return Record(
        source="datacite",
        title=title,
        authors=authors,
        surnames=surnames,
        year=year,
        years=[year] if year else [],
        venue=venue,
        doi=canonical_doi(doi),
        url=url,
        record_id=canonical_doi(doi),
    )


def crossref_message(data: object) -> dict[Any, Any] | None:
    if not isinstance(data, dict):
        return None
    message = data.get("message")
    if isinstance(message, dict):
        return message
    if "DOI" in data or "title" in data:
        return data
    return None


def _datacite_creators(value: object) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    surnames: list[str] = []
    if not isinstance(value, list):
        return authors, surnames
    for entry in value:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("familyName") or "").strip()
        given = str(entry.get("givenName") or "").strip()
        name = str(entry.get("name") or "").strip()
        if family:
            authors.append(f"{given} {family}".strip())
            surnames.append(surname_from_display(family))
        elif "," in name:
            last, _, rest = name.partition(",")
            authors.append(f"{rest.strip()} {last.strip()}".strip())
            surnames.append(surname_from_display(last.strip()))
        elif name:
            authors.append(name)
            surnames.append(surname_from_display(name))
    return authors, surnames


def _datacite_year(value: object) -> int | None:
    if isinstance(value, int) and 1000 <= value <= 2100:
        return value
    if isinstance(value, str) and value.isdigit():
        year = int(value)
        if 1000 <= year <= 2100:
            return year
    return None


def _crossref_authors(value: object) -> tuple[list[str], list[str]]:
    authors: list[str] = []
    surnames: list[str] = []
    if not isinstance(value, list):
        return authors, surnames
    for entry in value:
        if not isinstance(entry, dict):
            continue
        family = str(entry.get("family") or "").strip()
        given = str(entry.get("given") or "").strip()
        literal = str(entry.get("literal") or entry.get("name") or "").strip()
        if family:
            authors.append(f"{given} {family}".strip())
            surnames.append(family)
        elif literal:
            authors.append(literal)
            surnames.append(surname_from_display(literal))
    return authors, surnames


def _years(payload: dict[Any, Any]) -> list[int]:
    years: list[int] = []
    for key in ("published-print", "published-online", "issued", "published", "created"):
        year = _year_from_date(payload.get(key))
        if year and year not in years:
            years.append(year)
    return years


def _year_from_date(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    parts = value.get("date-parts") or value.get("date_parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        year = parts[0][0]
        if isinstance(year, int) and 1000 <= year <= 2100:
            return year
    return None


def _first_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None
