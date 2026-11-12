import logging
import threading

import pytest
import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings
from tests.fakes import install

from verifai.main import create_app
from verifai.web.throttle import CheckSlots, RateLimiter, Throttled


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_per_minute_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(per_minute=2, per_hour=0, clock=clock)
    limiter.admit("a")
    clock.now += 10
    limiter.admit("a")
    with pytest.raises(Throttled) as refused:
        limiter.admit("a")
    assert refused.value.status == 429
    assert refused.value.retry_after == 50
    assert "2 checks per minute" in refused.value.message
    limiter.admit("b")
    clock.now += 50
    limiter.admit("a")


def test_hourly_window_and_refusals_are_not_counted() -> None:
    clock = Clock()
    limiter = RateLimiter(per_minute=0, per_hour=1, clock=clock)
    limiter.admit("a")
    for _ in range(3):
        with pytest.raises(Throttled) as refused:
            limiter.admit("a")
    assert refused.value.retry_after == 3600
    assert "60 minutes" in refused.value.message
    clock.now += 3600
    limiter.admit("a")


def test_disabled_limiter_admits_everything() -> None:
    limiter = RateLimiter(per_minute=0, per_hour=0)
    assert limiter.enabled is False
    for _ in range(100):
        limiter.admit("a")


def test_client_table_is_capped() -> None:
    limiter = RateLimiter(per_minute=1, per_hour=0, max_clients=2, clock=Clock())
    limiter.admit("a")
    limiter.admit("b")
    limiter.admit("c")
    limiter.admit("a")  # "a" was evicted, so it starts fresh


def test_slots_refuse_when_every_slot_is_held() -> None:
    slots = CheckSlots(1, wait_seconds=0.0)
    with slots.hold():
        with pytest.raises(Throttled) as refused, slots.hold():
            pass
        assert refused.value.status == 503
    with slots.hold():
        pass


def test_slots_let_a_waiting_check_through() -> None:
    slots = CheckSlots(1, wait_seconds=5.0)
    entered = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with slots.hold():
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=worker)
    thread.start()
    entered.wait(5)
    release.set()
    with slots.hold():
        pass
    thread.join(5)


def test_api_and_form_report_the_limit(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    app = create_app(make_settings(rate_limit_per_minute=1))
    with TestClient(app) as client:
        first = client.post("/api/v1/check", json={"text": "10.1038/nature14539"})
        assert first.status_code == 200
        second = client.post("/api/v1/check", json={"text": "10.1038/nature14539"})
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) > 0
        assert "1 check per minute" in second.json()["detail"]
        form = client.post(
            "/check", data={"text": "10.1038/nature14539"}, headers={"HX-Request": "true"}
        )
        assert form.status_code == 200
        assert "1 check per minute" in form.text
        assert client.get("/health").status_code == 200


def test_busy_server_returns_503(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    app = create_app(make_settings(max_concurrent_checks=1, check_queue_seconds=0))
    with TestClient(app) as client, app.state.check_slots.hold():
        busy = client.post("/api/v1/check", json={"text": "10.1038/nature14539"})
    assert busy.status_code == 503
    assert "busy" in busy.json()["detail"]


def test_public_host_without_limits_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="verifai.main"):
        create_app(make_settings(host="0.0.0.0"))  # noqa: S104
    assert "no rate limit" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="verifai.main"):
        create_app(make_settings(host="0.0.0.0", rate_limit_per_minute=5))  # noqa: S104
    assert "no rate limit" not in caplog.text


def test_run_trusts_only_the_configured_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr("verifai.main.uvicorn.run", lambda _app, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(
        "verifai.main.get_settings", lambda: make_settings(forwarded_allow_ips="172.17.0.1")
    )
    from verifai.main import run

    run()
    assert seen["proxy_headers"] is True
    assert seen["forwarded_allow_ips"] == "172.17.0.1"


def test_a_platform_ip_header_is_used_only_when_configured(http_mock: respx.MockRouter) -> None:
    install(http_mock)
    trusting = create_app(make_settings(rate_limit_per_minute=1, client_ip_header="x-real-ip"))
    with TestClient(trusting) as client:
        body = {"text": "10.1038/nature14539"}
        assert (
            client.post("/api/v1/check", json=body, headers={"x-real-ip": "1.1.1.1"}).status_code
            == 200
        )
        assert (
            client.post("/api/v1/check", json=body, headers={"x-real-ip": "2.2.2.2"}).status_code
            == 200
        )
        assert (
            client.post("/api/v1/check", json=body, headers={"x-real-ip": "1.1.1.1"}).status_code
            == 429
        )
    ignoring = create_app(make_settings(rate_limit_per_minute=1))
    with TestClient(ignoring) as client:
        assert (
            client.post("/api/v1/check", json=body, headers={"x-real-ip": "3.3.3.3"}).status_code
            == 200
        )
        assert (
            client.post("/api/v1/check", json=body, headers={"x-real-ip": "4.4.4.4"}).status_code
            == 429
        )
