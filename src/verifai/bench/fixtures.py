"""Recorded passages for an offline run.

The retriever implements the same protocol as Wikipedia or PubMed. Experts
still call ``check_claim``, which still calls ``corroborate``. Only the
network fetch is replaced. A passage may carry ``native_stance``, which the
existing judge path treats as a published rating and does not send to a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from verifai.corroboration import Stance
from verifai.retrieval.types import Passage, RetrievalResult


def normalize_claim(text: str) -> str:
    return " ".join(text.split())


class FixtureRetriever:
    name = "fixture"

    def __init__(self, passages: dict[str, tuple[Passage, ...]]) -> None:
        self._passages = passages

    def retrieve(self, query: str) -> RetrievalResult:
        found = self._passages.get(normalize_claim(query), ())
        return RetrievalResult(self.name, passages=list(found))


def load_passage_fixtures(path: Path) -> dict[str, tuple[Passage, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} is not a JSON object of claim text to passages.")
    loaded: dict[str, tuple[Passage, ...]] = {}
    for claim, rows in raw.items():
        if not isinstance(claim, str) or not isinstance(rows, list):
            raise TypeError(f"{path} has a bad entry for {claim!r}.")
        passages = tuple(_passage(row, claim, path) for row in rows)
        loaded[normalize_claim(claim)] = passages
    return loaded


def _passage(row: object, claim: str, path: Path) -> Passage:
    if not isinstance(row, dict):
        raise TypeError(f"{path} passage for {claim!r} is not an object.")
    source = row.get("source")
    text = row.get("text")
    title = row.get("title")
    if not isinstance(source, str) or not isinstance(text, str) or not isinstance(title, str):
        raise TypeError(f"{path} passage for {claim!r} needs source, title, and text.")
    family = row.get("family")
    if not isinstance(family, str) or not family:
        family = source
    document_ids = row.get("document_ids", [])
    if not isinstance(document_ids, list) or not all(
        isinstance(item, str) for item in document_ids
    ):
        raise TypeError(f"{path} document_ids for {claim!r} must be a list of strings.")
    stance = _stance(row.get("native_stance"), claim, path)
    confidence = row.get("native_confidence")
    native_confidence = float(confidence) if isinstance(confidence, (int, float)) else None
    url = row.get("url")
    return Passage(
        source=source,
        family=family,
        title=title,
        url=url if isinstance(url, str) else None,
        text=text,
        document_ids=tuple(item for item in document_ids if isinstance(item, str)),
        native_stance=stance,
        native_confidence=native_confidence,
        rating=str(row.get("rating") or "") or None,
    )


def _stance(raw: object, claim: str, path: Path) -> Stance | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(f"{path} native_stance for {claim!r} must be a string.")
    try:
        return Stance(raw.strip().casefold())
    except ValueError as exc:
        raise ValueError(f"{path} has unknown stance {raw!r} for {claim!r}.") from exc
