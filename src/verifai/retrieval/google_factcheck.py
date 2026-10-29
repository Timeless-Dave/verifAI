"""Google Fact Check Tools API. Every review it returns is still one source."""

from __future__ import annotations

from typing import Any

from verifai.http import PoliteHttp
from verifai.retrieval.textutil import fetch_json, stance_from_rating
from verifai.retrieval.types import Passage, RetrievalResult

_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class GoogleFactCheckRetriever:
    name = "google_factcheck"

    def __init__(self, http: PoliteHttp, api_key: str, *, limit: int = 5) -> None:
        self.http = http
        self.api_key = api_key
        self.limit = limit

    def retrieve(self, query: str) -> RetrievalResult:
        text = " ".join((query or "").split())[:200]
        if not text:
            return RetrievalResult(self.name)
        body, error = fetch_json(
            self.http,
            _API,
            {
                "query": text,
                "key": self.api_key,
                "languageCode": "en",
                "pageSize": self.limit,
            },
            source=self.name,
        )
        if body is None:
            return RetrievalResult(self.name, unavailable=True, error=error)
        passages: list[Passage] = []
        for claim in _claims(body):
            passages.extend(_reviews(claim))
            if len(passages) >= self.limit:
                break
        return RetrievalResult(self.name, passages=passages[: self.limit])


def _claims(body: dict[str, Any]) -> list[dict[str, Any]]:
    claims = body.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def _reviews(claim: dict[str, Any]) -> list[Passage]:
    reviews = claim.get("claimReview")
    if not isinstance(reviews, list):
        return []
    claimant = str(claim.get("claimant") or "").strip()
    claim_text = str(claim.get("text") or "").strip()
    passages: list[Passage] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        rating = str(review.get("textualRating") or "").strip()
        stance, confidence = stance_from_rating(rating)
        publisher = review.get("publisher")
        speaker = ""
        if isinstance(publisher, dict):
            speaker = str(publisher.get("name") or publisher.get("site") or "").strip()
        title = str(review.get("title") or claim_text or "Fact check")
        url = review.get("url")
        who = speaker or claimant or "a fact checker"
        text = f"{who} rated “{claim_text or title}” as {rating or 'unrated'}."
        passages.append(
            Passage(
                source="google_factcheck",
                family="google_factcheck",
                title=title,
                url=url if isinstance(url, str) else None,
                text=text,
                speaker=speaker or claimant or None,
                native_stance=stance,
                native_confidence=confidence,
                rating=rating or None,
            )
        )
    return passages
