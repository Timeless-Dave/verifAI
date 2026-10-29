"""HTTP fixtures shaped like Crossref, doi.org, OpenAlex, and the Wayback Machine."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

import httpx

ATTENTION_TITLE = "Attention Is All You Need"
ATTENTION_DOI = "10.48550/arxiv.1706.03762"
JINEK_TITLE = "A Programmable Dual-RNA-Guided DNA Endonuclease in Adaptive Bacterial Immunity"
JINEK_DOI = "10.1126/science.1225829"
PLOS_DOI = "10.1371/journal.pone.0000217"
PLOS_TITLE = "Probing the Magnetosphere Using Chemical Releases"
FAKE_DOI = "10.1234/verifai.fabricated.2021"
WIKI_URL = "https://en.wikipedia.org/wiki/VerifAI_missing_citation_page_9f3c"

ATTENTION_AUTHORS = [
    "Ashish Vaswani",
    "Noam Shazeer",
    "Niki Parmar",
    "Jakob Uszkoreit",
    "Llion Jones",
    "Aidan Gomez",
    "Lukasz Kaiser",
    "Illia Polosukhin",
]


def install(
    router: Any,
    *,
    doi_org_down: bool = False,
    openalex_down: bool = False,
    crossref_down: bool = False,
    datacite_down: bool = False,
    attention_indexes_reprint_only: bool = False,
) -> None:
    """Register mocked bibliographic and archive routes. URL targets are added by each test."""

    def doi_org(request: httpx.Request) -> httpx.Response:
        if doi_org_down:
            return httpx.Response(503, text="doi.org down")
        doi = unquote(request.url.path).strip("/")
        record = _known_doi(doi)
        if record is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, json=_csl(record))

    def crossref(request: httpx.Request) -> httpx.Response:
        if crossref_down:
            return httpx.Response(503, text="crossref down")
        path = unquote(request.url.path).rstrip("/")
        if path == "/works":
            items = _search_items(request)
            if attention_indexes_reprint_only:
                items = [item for item in items if ATTENTION_DOI not in item.get("DOI", "").lower()]
            return httpx.Response(200, json={"status": "ok", "message": {"items": items}})
        doi = path.split("/works/", 1)[-1]
        record = _known_doi(doi)
        if record is None:
            return httpx.Response(404, text="Resource not found.")
        return httpx.Response(
            200, json={"status": "ok", "message-type": "work", "message": _crossref_item(record)}
        )

    def openalex(request: httpx.Request) -> httpx.Response:
        if openalex_down:
            return httpx.Response(503, text="openalex down")
        url = unquote(str(request.url))
        if "/works/doi:" in url:
            doi = url.split("/works/doi:", 1)[1].split("?", 1)[0]
            record = _known_doi(doi)
            if record is None:
                return httpx.Response(404, text="missing")
            return httpx.Response(200, json=_openalex(record))
        found = _search_records(_filter_title(request))
        if attention_indexes_reprint_only:
            found = [item for item in found if item["doi"].lower() != ATTENTION_DOI]
        return httpx.Response(
            200,
            json={"meta": {"count": len(found)}, "results": [_openalex(item) for item in found]},
        )

    def datacite(request: httpx.Request) -> httpx.Response:
        if datacite_down:
            return httpx.Response(503, text="datacite down")
        query = request.url.params.get("query", "")
        found = _search_records(_datacite_title(query))
        year = _datacite_year(query)
        if year is not None:
            found = [item for item in found if item["year"] == year]
        return httpx.Response(
            200,
            json={"meta": {"total": len(found)}, "data": [_datacite_item(item) for item in found]},
        )

    def wayback(request: httpx.Request) -> httpx.Response:
        target = request.url.params.get("url", "")
        if "archived-page" in target:
            return httpx.Response(
                200,
                json={
                    "url": target,
                    "archived_snapshots": {
                        "closest": {
                            "available": True,
                            "url": "http://web.archive.org/web/20200102030405/https://example.com/archived-page",
                            "timestamp": "20200102030405",
                            "status": "200",
                        }
                    },
                },
            )
        return httpx.Response(200, json={"url": target, "archived_snapshots": {}})

    router.get(host="doi.org").mock(side_effect=doi_org)
    router.get(host="api.crossref.org").mock(side_effect=crossref)
    router.get(host="api.openalex.org").mock(side_effect=openalex)
    router.get(host="api.datacite.org").mock(side_effect=datacite)
    router.get(host="archive.org").mock(side_effect=wayback)
    router.get(url__startswith=WIKI_URL).mock(return_value=httpx.Response(404, text="missing"))

    # Claim retrievers stay offline. The specific Wikipedia page route above
    # is registered first, so the dead citation URL is still a 404.
    def empty(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    for host in (
        "en.wikipedia.org",
        "www.wikidata.org",
        "factchecktools.googleapis.com",
        "eutils.ncbi.nlm.nih.gov",
        "www.ebi.ac.uk",
        "api.gdeltproject.org",
    ):
        router.get(host=host).mock(side_effect=empty)


def _datacite_title(query: str) -> str:
    marker = 'titles.title:"'
    if marker not in query:
        return ""
    return query.split(marker, 1)[1].split('"', 1)[0]


def _datacite_year(query: str) -> int | None:
    marker = "publicationYear:"
    if marker not in query:
        return None
    raw = query.split(marker, 1)[1].split("AND", 1)[0].strip()
    return int(raw) if raw.isdigit() else None


def _filter_title(request: httpx.Request) -> str:
    raw = str(request.url.params.get("filter", ""))
    if "title.search:" in raw:
        return raw.split("title.search:", 1)[1]
    return str(request.url.params.get("query.title", ""))


def _search_items(request: httpx.Request) -> list[dict[str, Any]]:
    return [
        _crossref_item(record)
        for record in _search_records(request.url.params.get("query.title", ""))
    ]


def _search_records(title: str) -> list[dict[str, Any]]:
    folded = title.casefold()
    if "attention is all you need" in folded:
        modern = _record(
            "10.65215/ysbyhc05",
            ATTENTION_TITLE,
            ATTENTION_AUTHORS[:2],
            2025,
            None,
        )
        original = _record(
            ATTENTION_DOI,
            ATTENTION_TITLE,
            ATTENTION_AUTHORS,
            2017,
            "Advances in Neural Information Processing Systems",
        )
        return [original, modern]
    if "endonuclease" in folded or "programmable" in folded:
        return [
            _record(
                JINEK_DOI,
                JINEK_TITLE,
                [
                    "Martin Jinek",
                    "Krzysztof Chylinski",
                    "Ines Fonfara",
                    "Michael Hauer",
                    "Jennifer Doudna",
                    "Emmanuelle Charpentier",
                ],
                2012,
                "Science",
            )
        ]
    if "sourdough" in folded:
        return [
            _record(
                "10.1111/epic.12046",
                "Scaling is Like Making Sourdough: Finding Sourdough Starters to Help Your Research Scale",
                ["Karyn Georgilis"],
                2020,
                "Epic Journal",
            )
        ]
    return []


def _known_doi(doi: str) -> dict[str, Any] | None:
    lowered = doi.lower()
    if FAKE_DOI in lowered:
        return None
    if PLOS_DOI in lowered:
        return _record(PLOS_DOI, PLOS_TITLE, ["J. Hunter Waite"], 2007, "PLOS ONE")
    if JINEK_DOI in lowered:
        return _record(
            JINEK_DOI,
            JINEK_TITLE,
            ["Martin Jinek", "Krzysztof Chylinski", "Jennifer Doudna"],
            2012,
            "Science",
        )
    if ATTENTION_DOI in lowered:
        return _record(
            ATTENTION_DOI,
            ATTENTION_TITLE,
            ATTENTION_AUTHORS,
            2017,
            "Advances in Neural Information Processing Systems",
        )
    return None


def _record(
    doi: str, title: str, authors: list[str], year: int, venue: str | None
) -> dict[str, Any]:
    return {"doi": doi, "title": title, "authors": authors, "year": year, "venue": venue}


def _crossref_item(record: dict[str, Any]) -> dict[str, Any]:
    item = {
        "DOI": record["doi"],
        "title": [record["title"]],
        "author": [
            {
                "given": name.split()[0],
                "family": name.split()[-1],
                "sequence": "first" if index == 0 else "additional",
            }
            for index, name in enumerate(record["authors"])
        ],
        "issued": {"date-parts": [[record["year"]]]},
        "published-print": {"date-parts": [[record["year"]]]},
        "URL": f"https://doi.org/{record['doi']}",
    }
    if record["venue"]:
        item["container-title"] = [record["venue"]]
    return item


def _csl(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "DOI": record["doi"],
        "title": record["title"],
        "author": [
            {"given": name.split()[0], "family": name.split()[-1]} for name in record["authors"]
        ],
        "issued": {"date-parts": [[record["year"]]]},
        "container-title": record["venue"],
        "URL": f"https://doi.org/{record['doi']}",
    }


def _datacite_item(record: dict[str, Any]) -> dict[str, Any]:
    creators = []
    for name in record["authors"]:
        parts = name.split()
        family = parts[-1]
        given = " ".join(parts[:-1])
        creators.append(
            {"name": f"{family}, {given}".strip(", "), "givenName": given, "familyName": family}
        )
    return {
        "id": f"https://doi.org/{record['doi']}",
        "attributes": {
            "doi": record["doi"],
            "titles": [{"title": record["title"]}],
            "creators": creators,
            "publicationYear": record["year"],
            "publisher": "Test publisher",
            "container": {"title": record["venue"]} if record["venue"] else {},
            "url": f"https://doi.org/{record['doi']}",
        },
    }


def _openalex(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/W{abs(hash(record['doi'])) % 10_000_000}",
        "doi": f"https://doi.org/{record['doi']}",
        "display_name": record["title"],
        "publication_year": record["year"],
        "authorships": [
            {"raw_author_name": name, "author": {"display_name": name}}
            for name in record["authors"]
        ],
        "primary_location": {
            "source": {"display_name": record["venue"]} if record["venue"] else None,
            "landing_page_url": f"https://doi.org/{record['doi']}",
        },
    }
