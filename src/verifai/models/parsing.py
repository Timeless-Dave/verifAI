"""Turn a local model's text into a stance. Invalid output is neutral."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from verifai.corroboration import Stance


@dataclass(frozen=True)
class Judgment:
    stance: Stance
    confidence: float
    explanation: str


_STANCES = {
    "supports": Stance.SUPPORTS,
    "support": Stance.SUPPORTS,
    "supported": Stance.SUPPORTS,
    "entailment": Stance.SUPPORTS,
    "entails": Stance.SUPPORTS,
    "yes": Stance.SUPPORTS,
    "true": Stance.SUPPORTS,
    "refutes": Stance.REFUTES,
    "refute": Stance.REFUTES,
    "refuted": Stance.REFUTES,
    "contradicts": Stance.REFUTES,
    "contradiction": Stance.REFUTES,
    "no": Stance.REFUTES,
    "false": Stance.REFUTES,
    "neutral": Stance.NEUTRAL,
    "unrelated": Stance.NEUTRAL,
    "unknown": Stance.NEUTRAL,
}


def judgment_prompt(claim: str, passage: str) -> str:
    return (
        "You are a careful claim checker. Decide whether the evidence supports, "
        "refutes, or does not address the claim. Use only the evidence. "
        'Reply with one JSON object and no other text: {"stance":"supports"|"refutes"|'
        '"neutral","confidence":0.0,"explanation":"one sentence"}\n'
        f"Claim: {claim.strip()[:600]}\n"
        f"Evidence: {passage.strip()[:3500]}"
    )


def parse_judgment(raw: str) -> Judgment:
    data = _extract_object(raw)
    if data is None:
        return Judgment(Stance.NEUTRAL, 0.0, "The local model did not return JSON.")
    stance = _STANCES.get(str(data.get("stance", "")).strip().lower(), Stance.NEUTRAL)
    confidence = _confidence(
        data.get("confidence"), default=0.5 if stance is not Stance.NEUTRAL else 0.0
    )
    explanation = str(data.get("explanation", "")).strip()[:400]
    return Judgment(stance, confidence, explanation)


def map_nli_rows(rows: object) -> Judgment:
    if isinstance(rows, dict):
        items = [rows]
    elif isinstance(rows, list):
        items = [row for row in rows if isinstance(row, dict)]
    else:
        items = []
    if not items:
        return Judgment(Stance.NEUTRAL, 0.0, "The NLI model returned no label.")
    best = max(items, key=_score)
    label = str(best.get("label", ""))
    score = _score(best)
    return map_nli_label(label, score)


def map_nli_label(label: str, score: float) -> Judgment:
    folded = label.strip().lower().replace("-", " ")
    if "entail" in folded or folded in {"supports", "support", "yes", "1"}:
        stance = Stance.SUPPORTS
    elif "contradict" in folded or folded in {"refutes", "refute", "no", "0"}:
        stance = Stance.REFUTES
    else:
        stance = Stance.NEUTRAL
    confidence = _confidence(score, default=0.0)
    if confidence < 0.55:
        stance = Stance.NEUTRAL
    explanation = f"NLI label {label.strip() or 'unknown'}."
    return Judgment(stance, confidence, explanation)


def _extract_object(raw: str) -> dict[str, Any] | None:
    starts = [index for index in (raw.find("{"), raw.find("[")) if index >= 0]
    if not starts:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[min(starts) :])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _confidence(value: object, *, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = default
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return round(number, 2)


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", 0))
    except (TypeError, ValueError):
        return 0.0
