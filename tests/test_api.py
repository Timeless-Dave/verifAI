from tests.fakes import install
from verifai.sample import EXAMPLE_TEXT


def test_homepage_and_form_and_json(client, http_mock):
    install(http_mock)
    home = client.get("/")
    assert home.status_code == 200
    assert "Do these references exist?" in home.text
    assert "Crossref" in home.text
    assert "OpenAlex" in home.text
    assert "if one is down" in home.text

    empty = client.post("/check", data={"text": "   "})
    assert empty.status_code == 200
    assert "Paste some text first" in empty.text

    page = client.post("/check", data={"text": EXAMPLE_TEXT})
    assert page.status_code == 200
    assert "Verified" in page.text
    assert "Attention Is All You Need" in page.text
    assert "Crossref" in page.text and "OpenAlex" in page.text

    partial = client.post("/check", data={"text": EXAMPLE_TEXT}, headers={"HX-Request": "true"})
    assert partial.status_code == 200
    assert "<html" not in partial.text.lower()
    assert partial.text.count("Verified") == 3
    assert partial.text.count("Not found") == 3
    assert partial.text.count("Partially matches") == 0

    api = client.post("/api/v1/check", json={"text": EXAMPLE_TEXT})
    assert api.status_code == 200
    body = api.json()
    assert body["results"][0]["verdict"] == "mixed"
    assert body["results"][0]["verdict_scope"] == "citations"
    assert "sole decider" in body["results"][0]["evidence_principle"]
    statuses = [item["status"] for item in body["results"][0]["citations"]]
    assert statuses.count("verified") == 3
    assert statuses.count("not_found") == 3
    assert "crossref" in body["results"][0]["sources_consulted"]
    assert "openalex" in body["results"][0]["sources_consulted"]

    missing = client.post("/api/v1/check", json={})
    assert missing.status_code == 422
    blank = client.post("/api/v1/check", json={"text": "  "})
    assert blank.status_code == 422

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert "crossref" in payload["providers"]
    assert "openalex" in payload["providers"]
    assert "doi.org" in payload["providers"]
    assert "wayback" in payload["providers"]
    assert "datacite" in payload["providers"]
    assert "google_fact_check" not in payload["providers"]
