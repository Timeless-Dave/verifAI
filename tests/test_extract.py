from verifai.experts.citation.extract import clean_doi, clean_url, extract_citations
from verifai.sample import EXAMPLE_TEXT


def test_example_text_finds_real_and_fake_citations():
    found = extract_citations(EXAMPLE_TEXT)
    kinds = [(item.kind, item.doi or item.title or item.url) for item in found]
    assert [kind for kind, _ in kinds] == [
        "reference",
        "reference",
        "doi",
        "reference",
        "doi",
        "url",
    ]
    assert found[0].year == 2017
    assert found[0].surnames[0] == "Vaswani"
    assert "Polosukhin" in found[0].surnames
    assert found[0].title == "Attention is all you need"
    assert found[0].venue.startswith("Advances in Neural Information Processing Systems")
    assert found[1].surnames == ["Jinek"]
    assert found[1].et_al is True
    assert found[1].year == 2012
    assert found[1].venue == "Science"
    assert "endonuclease" in found[1].title.lower()
    assert found[2].doi == "10.1371/journal.pone.0000217"
    assert found[3].surnames == ["Smith", "Doe"]
    assert "sourdough" in found[3].title.lower()
    assert found[4].doi == "10.1234/verifai.fabricated.2021"
    assert found[5].url.startswith("https://en.wikipedia.org/")


def test_doi_url_is_not_also_a_plain_url():
    found = extract_citations("See https://doi.org/10.1371/journal.pone.0000217 for the paper.")
    assert len(found) == 1
    assert found[0].kind == "doi"


def test_trailing_punctuation_is_stripped():
    assert clean_doi("10.1126/science.1225829.") == "10.1126/science.1225829"
    assert clean_doi("doi:10.1126/science.1225829)") == "10.1126/science.1225829"
    assert clean_url("https://example.com/paper).") == "https://example.com/paper"
    found = extract_citations("The DOI is 10.1126/science.1225829.")
    assert found[0].doi == "10.1126/science.1225829"


def test_wrapped_reference_is_joined():
    text = (
        "Vaswani, A., Shazeer, N., Parmar, N.,\n"
        "Uszkoreit, J. (2017). Attention is all you need.\n"
        "Advances in Neural Information Processing Systems, 30.\n"
    )
    found = extract_citations(text)
    assert len(found) == 1
    assert found[0].title == "Attention is all you need"
    assert found[0].year == 2017
    assert "Vaswani" in found[0].surnames


def test_bare_author_year_is_ignored():
    text = "As shown before (Smith et al., 2021) the effect was small. See also Jones (1999)."
    assert extract_citations(text) == []


def test_ordinary_prose_is_not_a_citation():
    text = (
        "In March 2020 the lab closed. Smith said the samples were fine. "
        "They reopened in June, after lunch, before the vote on the annual budget proposal.\n\n"
        "From June, 2020, the office, a large team moved buildings and stayed home for months."
    )
    assert extract_citations(text) == []


def test_duplicate_doi_is_reported_once():
    text = "10.1126/science.1225829 and again https://doi.org/10.1126/science.1225829."
    found = extract_citations(text)
    assert len(found) == 1


def test_doi_keeps_bibliography_metadata_from_the_same_line():
    text = (
        "Smith, J. & Doe, A. (2019). Quantum entanglement in household sourdough starters. "
        "Journal of Improbable Culinary Physics, 4(2), 11-19. https://doi.org/10.1126/science.1225829"
    )
    found = extract_citations(text)
    assert len(found) == 1
    assert found[0].kind == "doi"
    assert found[0].doi == "10.1126/science.1225829"
    assert found[0].surnames == ["Smith", "Doe"]
    assert found[0].year == 2019
    assert "sourdough" in found[0].title.lower()
