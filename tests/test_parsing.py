from verifai.experts.citation.providers.parsing import (
    record_from_crossref_like,
    record_from_datacite,
    record_from_openalex,
)


def test_crossref_and_doi_org_shapes():
    crossref = record_from_crossref_like(
        "crossref",
        {
            "DOI": "10.1126/science.1225829",
            "title": ["A Programmable Dual-RNA–Guided DNA Endonuclease"],
            "author": [{"given": "Martin", "family": "Jinek", "sequence": "first"}],
            "published-print": {"date-parts": [[2012, 8, 17]]},
            "issued": {"date-parts": [[2012, 6, 28]]},
            "container-title": ["Science"],
            "URL": "https://doi.org/10.1126/science.1225829",
        },
    )
    assert crossref is not None
    assert crossref.year == 2012
    assert 2012 in crossref.years
    assert crossref.surnames == ["Jinek"]
    assert crossref.venue == "Science"
    assert crossref.doi == "10.1126/science.1225829"

    csl = record_from_crossref_like(
        "doi.org",
        {
            "DOI": "10.1126/science.1225829",
            "title": "A Programmable Dual-RNA–Guided DNA Endonuclease",
            "author": [{"given": "Martin", "family": "Jinek"}],
            "issued": {"date-parts": [[2012]]},
            "container-title": "Science",
        },
    )
    assert csl is not None
    assert csl.title is not None
    assert csl.title.startswith("A Programmable")
    assert csl.venue == "Science"


def test_openalex_shape():
    record = record_from_openalex(
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1126/science.1225829",
            "display_name": "A paper",
            "publication_year": 2012,
            "authorships": [
                {"raw_author_name": "Martin Jinek", "author": {"display_name": "Martin Jínek"}}
            ],
            "primary_location": {"source": {"display_name": "Science"}},
        }
    )
    assert record is not None
    assert record.surnames == ["Jinek"]
    assert record.venue == "Science"
    assert record.doi == "10.1126/science.1225829"
    assert record.url == "https://doi.org/10.1126/science.1225829"


def test_datacite_uses_container_not_publisher_as_venue():
    record = record_from_datacite(
        {
            "attributes": {
                "doi": "10.48550/arxiv.1706.03762",
                "publicationYear": 2017,
                "publisher": "arXiv",
                "container": {},
                "url": "https://arxiv.org/abs/1706.03762",
                "titles": [{"title": "Attention Is All You Need"}],
                "creators": [
                    {"givenName": "Ashish", "familyName": "Vaswani", "name": "Vaswani, Ashish"},
                    {"name": "Polosukhin, Illia"},
                ],
            }
        }
    )
    assert record is not None
    assert record.source == "datacite"
    assert record.year == 2017
    assert record.venue is None
    assert record.surnames == ["Vaswani", "Polosukhin"]
    assert record.authors[0] == "Ashish Vaswani"
    assert record.doi == "10.48550/arxiv.1706.03762"
    assert record.url == "https://arxiv.org/abs/1706.03762"
