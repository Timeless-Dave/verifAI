"""Central corroboration rule for claim verdicts.

A supported or unsupported verdict needs at least two independent sources
that agree. Google Fact Check reviews collapse to one source, however many
claim reviews came back. The same paper found in two indexes is one source.
One source against one source is not agreement. The verdict then stays
unverifiable, which this project reports as insufficient evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from verifai.schemas import Verdict, source_label

GOOGLE_FAMILY = "google_factcheck"


class Stance(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SourceVote:
    """One source's stance on a claim.

    ``document_ids`` tie the same work together across indexes (``doi:…``,
    ``pmid:…``). Votes that share an id are not independent.
    """

    source: str
    family: str
    stance: Stance
    confidence: float
    document_ids: tuple[str, ...] = ()
    url: str | None = None
    title: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class IndependentSource:
    key: str
    stance: Stance
    confidence: float
    sources: tuple[str, ...]
    titles: tuple[str, ...]


@dataclass(frozen=True)
class Corroboration:
    verdict: Verdict
    confidence: float
    explanation: str
    support: tuple[IndependentSource, ...]
    refute: tuple[IndependentSource, ...]


def corroborate(votes: list[SourceVote]) -> Corroboration:
    """Turn source stances into one verdict. Neutral votes do not count."""
    support, refute = _tally(votes)
    verdict, confidence, explanation = _decide(votes, support, refute)
    return Corroboration(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        support=support,
        refute=refute,
    )


def _tally(
    votes: list[SourceVote],
) -> tuple[tuple[IndependentSource, ...], tuple[IndependentSource, ...]]:
    groups = _group([vote for vote in votes if vote.stance is not Stance.NEUTRAL])
    support = tuple(group for group in groups if group.stance is Stance.SUPPORTS)
    refute = tuple(group for group in groups if group.stance is Stance.REFUTES)
    return support, refute


def _group(votes: list[SourceVote]) -> list[IndependentSource]:
    clusters: list[list[SourceVote]] = []
    for vote in votes:
        keys = set(_keys(vote))
        matched = [cluster for cluster in clusters if keys & _cluster_keys(cluster)]
        if not matched:
            clusters.append([vote])
            continue
        primary = matched[0]
        primary.append(vote)
        for extra in matched[1:]:
            primary.extend(extra)
            clusters.remove(extra)
    groups = [group for members in clusters if (group := _reduce(members)) is not None]
    groups.sort(key=lambda group: group.key)
    return groups


def _cluster_keys(members: list[SourceVote]) -> set[str]:
    keys: set[str] = set()
    for member in members:
        keys.update(_keys(member))
    return keys


def _keys(vote: SourceVote) -> tuple[str, ...]:
    if _is_google(vote):
        return ("family:google_factcheck",)
    documents = tuple(f"doc:{item}" for item in vote.document_ids if item)
    if documents:
        return documents
    family = vote.family or vote.source
    return (f"family:{family}",)


def _is_google(vote: SourceVote) -> bool:
    return vote.family == GOOGLE_FAMILY or vote.source == "google_factcheck"


def _reduce(members: list[SourceVote]) -> IndependentSource | None:
    stances = {member.stance for member in members}
    if len(stances) != 1:
        return None
    stance = next(iter(stances))
    confidence = sum(_bounded(member.confidence) for member in members) / len(members)
    sources = tuple(dict.fromkeys(member.source for member in members))
    titles = tuple(dict.fromkeys(member.title for member in members if member.title))
    return IndependentSource(
        key=_group_key(members),
        stance=stance,
        confidence=confidence,
        sources=sources,
        titles=titles,
    )


def _group_key(members: list[SourceVote]) -> str:
    if any(_is_google(member) for member in members):
        return "family:google_factcheck"
    documents = tuple(
        dict.fromkeys(item for member in members for item in member.document_ids if item)
    )
    if documents:
        return f"doc:{documents[0]}"
    family = members[0].family or members[0].source
    return f"family:{family}"


def _decide(
    votes: list[SourceVote],
    support: tuple[IndependentSource, ...],
    refute: tuple[IndependentSource, ...],
) -> tuple[Verdict, float, str]:
    n_support = len(support)
    n_refute = len(refute)
    if n_support >= 2 and n_refute == 0:
        return (
            Verdict.SUPPORTED,
            _confidence(support, dissent=False),
            _agreed("supported", support),
        )
    if n_refute >= 2 and n_support == 0:
        return (
            Verdict.UNSUPPORTED,
            _confidence(refute, dissent=False),
            _agreed("refuted", refute),
        )
    if n_support >= 2 and n_refute >= 2:
        return (
            Verdict.MIXED,
            _confidence(support + refute, dissent=True),
            _mixed(support, refute),
        )
    if n_support >= 2 and n_refute == 1:
        return (
            Verdict.SUPPORTED,
            _confidence(support, dissent=True),
            _agreed("supported", support) + " " + _dissent(refute),
        )
    if n_refute >= 2 and n_support == 1:
        return (
            Verdict.UNSUPPORTED,
            _confidence(refute, dissent=True),
            _agreed("refuted", refute) + " " + _dissent(support),
        )
    return Verdict.UNVERIFIABLE, 0.0, _insufficient(votes, support, refute)


def _confidence(groups: tuple[IndependentSource, ...], *, dissent: bool) -> float:
    mean = sum(group.confidence for group in groups) / len(groups)
    cap = 0.72 if len(groups) == 2 else 0.88
    if dissent:
        mean *= 0.85
        cap = min(cap, 0.6)
    return round(min(cap, max(0.0, mean)), 2)


def _agreed(word: str, groups: tuple[IndependentSource, ...]) -> str:
    count = len(groups)
    noun = "source" if count == 1 else "sources"
    return f"{count} independent {noun} agree that this claim is {word}: {_name_list(groups)}."


def _dissent(groups: tuple[IndependentSource, ...]) -> str:
    return (
        f"{_name_list(groups)} disagrees. That dissent does not outnumber the sources that agree."
    )


def _mixed(
    support: tuple[IndependentSource, ...],
    refute: tuple[IndependentSource, ...],
) -> str:
    return (
        "Independent sources disagree, and both sides are corroborated. "
        f"Supporting: {_name_list(support)}. Refuting: {_name_list(refute)}."
    )


def _insufficient(
    votes: list[SourceVote],
    support: tuple[IndependentSource, ...],
    refute: tuple[IndependentSource, ...],
) -> str:
    directional = [vote for vote in votes if vote.stance is not Stance.NEUTRAL]
    google_only = bool(directional) and all(_is_google(vote) for vote in directional)
    if google_only and not support and not refute:
        return (
            "This stays unverified: insufficient evidence. Google Fact Check "
            "reviews do not agree with each other, and no other source judged "
            "the claim. Google Fact Check counts as one source at most."
        )
    if google_only:
        return (
            "This stays unverified: insufficient evidence. Google Fact Check "
            "is one source, even when several reviews agree. A verdict needs "
            "at least two independent sources that agree."
        )
    present = support + refute
    if len(present) == 1:
        return (
            "This stays unverified: insufficient evidence. Only "
            f"{_name_list(present)} speaks to this claim. A verdict needs at "
            "least two independent sources that agree."
        )
    if present:
        return (
            "This stays unverified: insufficient evidence. The sources that "
            "answered do not agree, and neither side has two independent sources."
        )
    if directional:
        return (
            "This stays unverified: insufficient evidence. The retrieved "
            "sources conflict internally, so none of them counts as agreement."
        )
    return (
        "This stays unverified: insufficient evidence. No source judged the "
        "claim as supported or refuted."
    )


def _name_list(groups: tuple[IndependentSource, ...]) -> str:
    names = [_describe(group) for group in groups]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _describe(group: IndependentSource) -> str:
    labels = [source_label(name) for name in group.sources]
    if len(labels) == 1:
        label = labels[0]
    elif len(labels) == 2:
        label = f"{labels[0]} and {labels[1]}"
    else:
        label = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    if group.titles:
        title = group.titles[0]
        if len(title) > 80:
            title = title[:77] + "..."
        return f"{label} (“{title}”)"
    return label


def _bounded(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
