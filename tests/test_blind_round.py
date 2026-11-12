"""Regressions from the blind live test of 50 new claims."""

import httpx

from verifai.claims import screen_paste
from verifai.retrieval.wikidata import WikidataRetriever, _pick


def test_sentences_about_proper_nouns_are_claims_not_titles() -> None:
    claims, _ignored = screen_paste(
        "Germany won the 2014 FIFA World Cup.\n"
        "William Shakespeare wrote Don Quixote.\n"
        "The Titanic sank in the North Atlantic Ocean.\n"
        "Deep Residual Learning for Image Recognition.\n"
        "Attention Is All You Need."
    )
    assert claims == [
        "Germany won the 2014 FIFA World Cup.",
        "William Shakespeare wrote Don Quixote.",
        "The Titanic sank in the North Atlantic Ocean.",
    ]


def _hit(qid: str, label: str, description: str, match: str | None = None) -> dict[str, object]:
    return {
        "id": qid,
        "label": label,
        "description": description,
        "match": {"text": match or label},
    }


def test_the_item_a_name_means_in_the_claim() -> None:
    nile = [
        _hit("Q660835", "Nile", "American death metal band"),
        _hit("Q3392", "Nile", "major river in northeastern Africa"),
    ]
    assert _pick(nile, "Nile", "The Nile flows into the sea.", {"Q660835": 20, "Q3392": 250}) == (
        "Q3392"
    )
    apple = [
        _hit("Q20056642", "Apple Music", "Internet online music service by Apple"),
        _hit("Q89", "apple", "edible fruit of the apple tree"),
        _hit("Q312", "Apple Inc.", "American technology company"),
    ]
    reach = {"Q20056642": 60, "Q89": 300, "Q312": 200}
    assert _pick(apple, "Apple", "Apple was founded in 1976.", reach) == "Q312"
    napoleon = [
        _hit("Q517", "Napoleon", "French Emperor", match="Napoléon Bonaparte"),
        _hit("Q947316", "Napoleon XIV", "American musician", match="Napoleon Bonaparte"),
    ]
    reach = {"Q517": 250, "Q947316": 15}
    assert _pick(napoleon, "Napoleon Bonaparte", "Napoleon Bonaparte was born.", reach) == "Q517"
    jordan = [
        _hit("Q41421", "Michael Jordan", "American basketball player"),
        _hit(
            "Q3308285", "Michael I. Jordan", "American computer scientist", match="Michael Jordan"
        ),
    ]
    claim = "Michael Jordan is a professor of computer science."
    assert _pick(jordan, "Michael Jordan", claim, {"Q41421": 200, "Q3308285": 30}) == "Q3308285"
    assert _pick([], "Nothing", claim, {}) is None


def test_one_oversized_item_does_not_sink_the_others(polite_http, http_mock):
    bolt = {
        "labels": {"en": {"value": "Usain Bolt"}},
        "descriptions": {"en": {"value": "Jamaican sprinter"}},
        "claims": {},
    }

    def api(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params["action"] == "wbsearchentities":
            found = {"Usain Bolt": "Q1189", "Jamaica": "Q766"}[params["search"]]
            return httpx.Response(200, json={"search": [{"id": found}]})
        if params["props"] == "sitelinks":
            return httpx.Response(200, json={"entities": {}})
        if params["ids"] == "Q766":
            return httpx.Response(200, content=b"x" * 3_000_000)
        return httpx.Response(200, json={"entities": {"Q1189": bolt}})

    http_mock.get("https://www.wikidata.org/w/api.php").mock(side_effect=api)
    found = WikidataRetriever(polite_http).retrieve("Usain Bolt is from Jamaica.")
    assert not found.unavailable
    assert [passage.title for passage in found.passages] == ["Usain Bolt"]


def test_an_attribute_phrase_falls_back_to_its_object(polite_http, http_mock):
    gold = {
        "labels": {"en": {"value": "gold"}},
        "descriptions": {"en": {"value": "chemical element"}},
        "claims": {
            "P246": [
                {"mainsnak": {"datavalue": {"type": "string", "value": "Au"}}, "rank": "normal"}
            ]
        },
    }

    def api(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params["action"] == "wbsearchentities":
            hits = [{"id": "Q897"}] if params["search"] == "gold" else []
            return httpx.Response(200, json={"search": hits})
        if params["props"] in {"sitelinks", "labels"}:
            return httpx.Response(200, json={"entities": {}})
        return httpx.Response(200, json={"entities": {"Q897": gold}})

    http_mock.get("https://www.wikidata.org/w/api.php").mock(side_effect=api)
    found = WikidataRetriever(polite_http).retrieve("The chemical symbol for gold is Ag.")
    assert found.passages[0].text == "gold (chemical element). element symbol: Au."
