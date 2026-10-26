import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings

from verifai.config import get_settings
from verifai.experts.citation.providers.doi_org import DoiOrgProvider
from verifai.experts.citation.providers.wayback import WaybackProvider
from verifai.http import PoliteHttp
from verifai.main import create_app, run


def test_run_binds_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        del app
        captured.update(kwargs)

    monkeypatch.setattr("verifai.main.uvicorn.run", fake_run)
    get_settings.cache_clear()
    try:
        run()
    finally:
        get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8741


def test_unexpected_error_hides_the_traceback() -> None:
    application = create_app(make_settings())

    @application.get("/boom")
    def boom() -> None:
        raise RuntimeError("disk secret")

    def explode(text: str) -> None:
        del text
        raise RuntimeError("disk secret")

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/boom")
        application.state.router.experts[0].check = explode
        page = client.post("/check", data={"text": "https://example.com/paper"})
        api = client.post("/api/v1/check", json={"text": "https://example.com/paper"})
    assert response.status_code == 500
    assert response.json()["detail"] == "The check failed before it could finish."
    assert "disk secret" not in response.text
    assert response.headers["x-request-id"]
    assert page.status_code == 200
    assert "failed before it could finish" in page.text
    assert "disk secret" not in page.text
    assert api.status_code == 500
    assert "disk secret" not in api.text


def test_doi_org_and_wayback_report_bad_payloads() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(url__startswith="https://doi.org/").mock(
            side_effect=[
                httpx.Response(503, text="down"),
                httpx.Response(401, text="no"),
                httpx.Response(200, json=["nope"]),
                httpx.Response(200, json={"title": "", "DOI": ""}),
            ]
        )
        router.get("https://archive.org/wayback/available").mock(
            side_effect=[
                httpx.Response(500, text="archive down"),
                httpx.Response(401, text="no"),
                httpx.Response(200, text="not-json"),
                httpx.Response(
                    200,
                    json={
                        "archived_snapshots": {
                            "closest": {
                                "available": True,
                                "url": "https://web.archive.org/web/20200101/https://example.com",
                                "timestamp": "20200101000000",
                            }
                        }
                    },
                ),
            ]
        )
        http = PoliteHttp(make_settings())
        doi = DoiOrgProvider(http)
        wayback = WaybackProvider(http)
        try:
            down = doi.lookup_doi("10.1/down")
            denied = doi.lookup_doi("10.1/denied")
            not_json = doi.lookup_doi("10.1/list")
            no_record = doi.lookup_doi("10.1/empty")
            archive_down = wayback.lookup("https://example.com/a")
            archive_denied = wayback.lookup("https://example.com/b")
            archive_garbage = wayback.lookup("https://example.com/c")
            archive_hit = wayback.lookup("https://example.com/d")
        finally:
            http.close()
    assert down.unavailable and denied.unavailable and not_json.unavailable
    assert "not JSON" in (not_json.error or "")
    assert no_record.unavailable and "no title" in (no_record.error or "")
    assert archive_down.unavailable and archive_denied.unavailable and archive_garbage.unavailable
    assert archive_hit.snapshot_url is not None
    assert archive_hit.timestamp == "20200101000000"
