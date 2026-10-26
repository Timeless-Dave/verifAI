"""Direct liveness check for an http(s) URL."""

from __future__ import annotations

from verifai.experts.citation.providers.base import UrlCheck
from verifai.http import PoliteHttp
from verifai.net import public_url_problem


class UrlLivenessProvider:
    name = "url"

    def __init__(self, http: PoliteHttp) -> None:
        self.http = http

    def check(self, url: str) -> UrlCheck:
        problem = public_url_problem(url)
        if problem:
            return UrlCheck(status_code=None, final_url=None, error=None, blocked_reason=problem)
        result = self.http.probe(url)
        return UrlCheck(
            status_code=result.status_code,
            final_url=result.url,
            error=result.error,
            blocked_reason=None,
        )
