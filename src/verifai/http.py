"""Polite HTTP: identifiable User-Agent, timeouts, retries, and a short cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from verifai.config import Settings

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
CACHEABLE_STATUSES = frozenset({200, 201, 204, 301, 302, 303, 307, 308, 401, 403, 404, 410})


@dataclass
class HttpResult:
    url: str
    status_code: int | None
    content: bytes
    headers: dict[str, str]
    error: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300

    def json_body(self) -> dict[str, Any] | list[Any] | None:
        if not self.content:
            return None
        try:
            data = json.loads(self.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(data, (dict, list)):
            return data
        return None


@dataclass
class _CacheEntry:
    expires: float
    result: HttpResult


class TTLCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> HttpResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires <= now:
                self._data.pop(key, None)
                return None
            cached = entry.result
            return HttpResult(
                url=cached.url,
                status_code=cached.status_code,
                content=cached.content,
                headers=dict(cached.headers),
                error=cached.error,
                from_cache=True,
            )

    def set(self, key: str, result: HttpResult, ttl: float) -> None:
        if ttl <= 0:
            return
        with self._lock:
            self._data[key] = _CacheEntry(expires=time.monotonic() + ttl, result=result)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class PoliteHttp:
    """Shared client for every evidence provider.

    Responses of 429 and 5xx are retried. 404s are cached briefly so a
    fabricated DOI is not requested again on every keystroke-equivalent
    recheck. Transport failures are not cached.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = TTLCache()
        self.client = httpx.Client(
            timeout=request_timeout(settings.http_timeout),
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            max_redirects=5,
            limits=httpx.Limits(max_connections=12, max_keepalive_connections=6),
        )

    def close(self) -> None:
        self.client.close()

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResult:
        accept = (headers or {}).get("Accept", "")
        key = _cache_key("GET", url, params, accept)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._with_retries("GET", url, params=params, headers=headers)
        self._maybe_cache(key, result)
        return result

    def probe(self, url: str) -> HttpResult:
        """GET just enough of a URL to learn its status, without saving a large body."""
        key = _cache_key("PROBE", url, None, "")
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._probe_with_retries(url)
        self._maybe_cache(key, result)
        return result

    def _maybe_cache(self, key: str, result: HttpResult) -> None:
        if result.error is not None or result.status_code not in CACHEABLE_STATUSES:
            return
        if len(result.content) > 1_500_000:
            return
        ttl = (
            self.settings.negative_cache_ttl_seconds
            if result.status_code in {404, 410}
            else self.settings.cache_ttl_seconds
        )
        self.cache.set(key, result, ttl)

    def _with_retries(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> HttpResult:
        attempts = self.settings.http_retries + 1
        last: HttpResult | None = None
        for attempt in range(attempts):
            last = self._request(method, url, params=params, headers=headers)
            retry = last.error is not None or last.status_code in RETRY_STATUSES
            if not retry or attempt + 1 >= attempts:
                return last
            self._sleep(attempt)
        assert last is not None
        return last

    def _probe_with_retries(self, url: str) -> HttpResult:
        attempts = self.settings.http_retries + 1
        last: HttpResult | None = None
        for attempt in range(attempts):
            last = self._stream_get(url)
            retry = last.error is not None or last.status_code in RETRY_STATUSES
            if not retry or attempt + 1 >= attempts:
                return last
            self._sleep(attempt)
        assert last is not None
        return last

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> HttpResult:
        try:
            response = self.client.request(
                method,
                url,
                params=_clean_params(params),
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return HttpResult(
                url=url,
                status_code=None,
                content=b"",
                headers={},
                error=type(exc).__name__ + ": " + str(exc),
            )
        return _from_response(response)

    def _stream_get(self, url: str) -> HttpResult:
        try:
            with self.client.stream("GET", url, headers={"Accept": "*/*"}) as response:
                chunks = bytearray()
                for chunk in response.iter_bytes(512):
                    chunks.extend(chunk)
                    if len(chunks) >= 512:
                        break
                return _from_response(response, content=bytes(chunks))
        except httpx.HTTPError as exc:
            return HttpResult(
                url=url,
                status_code=None,
                content=b"",
                headers={},
                error=type(exc).__name__ + ": " + str(exc),
            )

    def _sleep(self, attempt: int) -> None:
        delay = self.settings.http_backoff_seconds * (2**attempt)
        if delay > 0:
            time.sleep(delay)


def request_timeout(read_seconds: float) -> httpx.Timeout:
    """Allow a slow index to answer, and fail fast when the host does not connect.

    DataCite's title query for a heavily reprinted paper can take longer than
    a few seconds. A short read timeout drops that answer and leaves only the
    later reprint from the other indexes.
    """
    read = max(0.1, read_seconds)
    quick = min(5.0, read)
    return httpx.Timeout(connect=quick, read=read, write=min(10.0, read), pool=quick)


def _from_response(response: httpx.Response, content: bytes | None = None) -> HttpResult:
    body = response.content if content is None else content
    return HttpResult(
        url=str(response.url),
        status_code=response.status_code,
        content=body,
        headers={key.lower(): value for key, value in response.headers.items()},
        error=None,
    )


def _clean_params(params: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not params:
        return None
    cleaned = {key: value for key, value in params.items() if value is not None and value != ""}
    return cleaned or None


def _cache_key(method: str, url: str, params: Mapping[str, Any] | None, accept: str) -> str:
    payload = json.dumps(
        {"m": method, "u": url, "p": _clean_params(params) or {}, "a": accept},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
