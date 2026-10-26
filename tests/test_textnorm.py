from verifai.experts.citation.textnorm import (
    canonical_doi,
    format_people,
    surname_from_display,
    surnames_match,
)


def test_surnames_keep_particles_and_drop_suffixes() -> None:
    assert surname_from_display("Ludwig van Beethoven") == "van Beethoven"
    assert surname_from_display("Maria de la Cruz") == "de la Cruz"
    assert surname_from_display("John Smith Jr.") == "Smith"
    assert surname_from_display("Jr.") == "Jr."


def test_surname_comparison_and_people_list() -> None:
    assert surnames_match("", "Smith") is False
    assert surnames_match("Jo", "Joseph") is False
    assert surnames_match("Kaiser", "Lukasz Kaiser") is True
    assert format_people([]) == ""
    assert format_people(["A", "", "B", "C", "D", "E"]) == "A, B, C, D, et al."


def test_doi_canonicalization() -> None:
    assert canonical_doi(None) is None
    assert canonical_doi("   ") is None
    assert canonical_doi("https://dx.doi.org/10.1/ABC") == "10.1/abc"
    assert canonical_doi("doi: 10.1/ABC") == "10.1/abc"
