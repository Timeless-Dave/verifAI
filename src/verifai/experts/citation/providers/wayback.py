"""Wayback Machine availability API, used when a direct fetch cannot confirm a URL."""

from __future__ import annotations

from verifai.experts.citation.providers.base import WaybackHit
from verifai.http import RETRY_STATUSES, PoliteHttp


class WaybackProvider:
    name = "wayback"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def lookup(self, url: str) -> WaybackHit:
        result = self.http.get("https://archive.org/wayback/available", params={"url": url})
        if result.error is not None or result.status_code is None or result.status_code in RETRY_STATUSES:
            return WaybackHit(unavailable=True, snapshot_url=None, timestamp=None, error=result.error or f"HTTP {result.status_code}")
        if result.status_code != 200:
            return WaybackHit(
                unavailable=True,
                snapshot_url=None,
                timestamp=None,
                error=f"HTTP {result.status_code}",
            )
        body = result.json_body()
        if not isinstance(body, dict):
            return WaybackHit(unavailable=True, snapshot_url=None, timestamp=None, error="Wayback response was not JSON")
        closest = (body.get("archived_snapshots") or {}).get("closest") if isinstance(body.get("archived_snapshots"), dict) else None
        if isinstance(closest, dict) and closest.get("available") and closest.get("url"):
            timestamp = closest.get("timestamp")
            return WaybackHit(
                unavailable=False,
                snapshot_url=str(closest["url"]),
                timestamp=str(timestamp) if timestamp else None,
                error=None,
            )
        return WaybackHit(unavailable=False, snapshot_url=None, timestamp=None, error=None)
