from __future__ import annotations

from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient

from verifai.config import Settings
from verifai.experts.citation import build_citation_expert
from verifai.http import PoliteHttp
from verifai.main import create_app


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "mailto": "tests@example.com",
        "http_retries": 0,
        "http_backoff_seconds": 0.0,
        "cache_ttl_seconds": 0,
        "negative_cache_ttl_seconds": 0,
        "http_timeout": 2.0,
        "http_min_interval_seconds": 0.0,
        "max_concurrency": 1,
        "semantic_scholar_api_key": None,
        "openalex_api_key": None,
    }
    values.update(overrides)
    return Settings.model_validate(values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def http_mock():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def polite_http(settings, http_mock):
    client = PoliteHttp(settings)
    yield client
    client.close()


@pytest.fixture
def expert(settings, polite_http):
    return build_citation_expert(polite_http, settings)


@pytest.fixture
def client(settings, http_mock):
    application = create_app(settings)
    with TestClient(application) as test_client:
        yield test_client
