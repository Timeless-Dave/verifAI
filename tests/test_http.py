import httpx
import respx
from tests.conftest import make_settings

from verifai.http import PoliteHttp, request_timeout


def test_user_agent_retries_and_cache():
    with respx.mock(assert_all_called=False) as respx_mock:
        _assert_user_agent_retries_and_cache(respx_mock)


def _assert_user_agent_retries_and_cache(respx_mock):
    route = respx_mock.get("https://api.crossref.org/works").mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    settings = make_settings(http_retries=1, cache_ttl_seconds=60)
    http = PoliteHttp(settings)
    try:
        first = http.get("https://api.crossref.org/works")
        second = http.get("https://api.crossref.org/works")
    finally:
        http.close()
    assert first.status_code == 200
    assert first.json_body() == {"ok": True}
    assert second.from_cache is True
    assert route.call_count == 2
    agent = route.calls[0].request.headers["user-agent"]
    assert "VerifAI/" in agent
    assert "mailto:tests@example.com" in agent


def test_read_timeout_waits_longer_than_connect():
    timeout = request_timeout(20)
    assert timeout.read == 20
    assert timeout.connect == 5
    assert timeout.pool == 5
    short = request_timeout(2)
    assert short.read == 2
    assert short.connect == 2
