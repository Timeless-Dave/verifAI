"""Turn provider outcomes into one citation status.

Crossref, OpenAlex, and DataCite are peers for free-text references.
Semantic Scholar joins them when a key is configured. A hit from only one
of them can still verify a citation when another is down or has no
comparable record, at lower confidence. None of them is allowed to decide
alone when another also returned a competing record. A year-matching
record outranks a later reprint of the same title.
"""

from __future__ import annotations

from dataclasses import dataclass

from verifai.experts.citation.extract import ExtractedCitation
from verifai.experts.citation.match import Assessment, Query, assess, title_scores
from verifai.experts.citation.providers.base import ProviderOutcome, Record, UrlCheck, WaybackHit
from verifai.experts.citation.textnorm import canonical_doi, format_people
from verifai.schemas import (
    CitationResult,
    CitationStatus,
    EvidenceItem,
    Link,
    ordered_sources,
    source_label,
)

CONF_REFERENCE_VERIFIED_MULTI = 0.95
CONF_REFERENCE_VERIFIED_SINGLE_DOWN = 0.82
CONF_REFERENCE_VERIFIED_SINGLE = 0.78
CONF_REFERENCE_PARTIAL_MULTI = 0.74
CONF_REFERENCE_PARTIAL_SINGLE_DOWN = 0.63
CONF_REFERENCE_PARTIAL_SINGLE = 0.60
CONF_NOT_FOUND = 0.86
CONF_NOT_FOUND_PARTIAL_OUTAGE = 0.55
CONF_UNREACHABLE = 0.70

CONF_DOI_VERIFIED_MULTI = 0.97
CONF_DOI_VERIFIED_SINGLE_DOWN = 0.83
CONF_DOI_VERIFIED_SINGLE = 0.88
CONF_DOI_PARTIAL = 0.80
CONF_DOI_NOT_FOUND = 0.91
CONF_DOI_NOT_FOUND_CONFLICT = 0.72
CONF_DOI_NOT_FOUND_PARTIAL_OUTAGE = 0.58

CONF_URL_LIVE = 0.92
CONF_URL_ARCHIVED = 0.74
CONF_URL_ARCHIVED_UNCONFIRMED = 0.70
CONF_URL_NOT_FOUND = 0.80
CONF_URL_BLOCKED = 0.90

_SOURCE_PRIORITY = {"doi.org": 0, "crossref": 1, "openalex": 2, "semantic_scholar": 3}


@dataclass
class _Scored:
    record: Record
    assessment: Assessment


def decide_reference(
    citation: ExtractedCitation, outcomes: list[ProviderOutcome]
) -> CitationResult:
    consulted = ordered_sources([item.source for item in outcomes])
    unavailable = ordered_sources([item.source for item in outcomes if item.unavailable])
    contributed = ordered_sources([item.source for item in outcomes if item.contributed])
    query = _query(citation)

    if not query.title:
        return _result(
            citation,
            status=CitationStatus.UNREACHABLE,
            confidence=CONF_UNREACHABLE,
            explanation="This reference has no title, so it was not searched.",
            consulted=consulted,
            contributed=[],
            unavailable=unavailable,
        )

    scored = _score_all(query, outcomes)
    if not scored:
        return _not_found_or_unreachable(
            citation, query, outcomes, consulted, contributed, unavailable
        )

    scored.sort(key=_rank, reverse=True)
    best = scored[0]
    supporters = _supporters(best, scored)
    contradictors = _contradictors(best, scored, supporters)
    support_sources = ordered_sources(
        [best.record.source, *[item.record.source for item in supporters]]
    )
    band = _band(len(support_sources), any_down=bool(unavailable))

    status = best.assessment.status
    mismatches = list(best.assessment.mismatches)
    if contradictors and not supporters:
        status = CitationStatus.PARTIALLY_MATCHES
        winner = source_label(best.record.source)
        other = source_label(contradictors[0].record.source)
        mismatches.append(
            f"{winner} and {other} point at different works, so this is not treated as verified."
        )
        band = "single_unconfirmed"

    confidence = _reference_confidence(status, band)
    notes = _edition_notes(best.record, scored)
    explanation = _reference_explanation(
        status=status,
        record=best.record,
        support_sources=support_sources,
        outcomes=outcomes,
        notes=notes,
    )
    evidence_records = [best.record, *[item.record for item in supporters]]
    return _result(
        citation,
        status=status,
        confidence=confidence,
        explanation=explanation,
        mismatches=mismatches,
        consulted=consulted,
        contributed=contributed,
        unavailable=unavailable,
        evidence=_evidence(evidence_records, best.assessment.title_blend),
        links=_links(evidence_records),
        matched=best.record,
    )


def decide_doi(citation: ExtractedCitation, outcomes: list[ProviderOutcome]) -> CitationResult:
    consulted = ordered_sources([item.source for item in outcomes])
    unavailable = ordered_sources([item.source for item in outcomes if item.unavailable])
    contributed = ordered_sources([item.source for item in outcomes if item.contributed])
    doi_org = next((item for item in outcomes if item.source == "doi.org"), None)
    records = [record for item in outcomes for record in item.records]
    registry_miss = doi_org is not None and doi_org.contributed and not doi_org.records

    if registry_miss:
        return _doi_not_registered(citation, outcomes, records, consulted, contributed, unavailable)

    if not records:
        if contributed:
            down = _down_phrase(outcomes)
            explanation = f"No source that answered has a record for DOI {citation.doi}."
            if down:
                explanation += f" {down} Confidence is lower because a source was missing."
            confidence = CONF_DOI_NOT_FOUND_PARTIAL_OUTAGE if unavailable else CONF_DOI_NOT_FOUND
            return _result(
                citation,
                status=CitationStatus.NOT_FOUND,
                confidence=confidence,
                explanation=explanation,
                consulted=consulted,
                contributed=contributed,
                unavailable=unavailable,
            )
        return _result(
            citation,
            status=CitationStatus.UNREACHABLE,
            confidence=CONF_UNREACHABLE,
            explanation=_outage_sentence(outcomes),
            consulted=consulted,
            contributed=contributed,
            unavailable=unavailable,
        )

    primary = min(records, key=lambda record: _SOURCE_PRIORITY.get(record.source, 9))
    status = CitationStatus.VERIFIED
    mismatches: list[str] = []
    if citation.title and primary.title:
        assessment = assess(_query(citation), primary)
        if assessment is None:
            status = CitationStatus.PARTIALLY_MATCHES
            mismatches.append(
                f"The DOI is registered to “{primary.title}” ({_byline(primary)}), "
                f"which does not match the cited title “{citation.title}”."
            )
        elif assessment.status != CitationStatus.VERIFIED:
            status = CitationStatus.PARTIALLY_MATCHES
            mismatches.extend(assessment.mismatches)
    for other in records:
        if other is primary or not other.title or not primary.title:
            continue
        if canonical_doi(other.doi) and canonical_doi(other.doi) == canonical_doi(primary.doi):
            continue
        if title_scores(primary.title, other.title).blend < 0.60:
            status = CitationStatus.PARTIALLY_MATCHES
            mismatches.append(
                f"{source_label(primary.source)} and {source_label(other.source)} "
                "disagree about what this DOI refers to."
            )
            break

    positive_sources = ordered_sources([record.source for record in records])
    if status == CitationStatus.VERIFIED and len(positive_sources) >= 2:
        confidence = CONF_DOI_VERIFIED_MULTI
    elif status == CitationStatus.VERIFIED and unavailable:
        confidence = CONF_DOI_VERIFIED_SINGLE_DOWN
    elif status == CitationStatus.VERIFIED:
        confidence = CONF_DOI_VERIFIED_SINGLE
    else:
        confidence = CONF_DOI_PARTIAL

    explanation = _doi_explanation(
        citation=citation,
        primary=primary,
        positive_sources=positive_sources,
        outcomes=outcomes,
        status=status,
    )
    return _result(
        citation,
        status=status,
        confidence=confidence,
        explanation=explanation,
        mismatches=_unique(mismatches),
        consulted=consulted,
        contributed=contributed,
        unavailable=unavailable,
        evidence=_evidence(records, None),
        links=_links(records),
        matched=primary,
    )


def decide_url(
    citation: ExtractedCitation,
    live: UrlCheck,
    wayback: WaybackHit | None,
) -> CitationResult:
    if live.blocked_reason:
        return _result(
            citation,
            status=CitationStatus.UNREACHABLE,
            confidence=CONF_URL_BLOCKED,
            explanation=live.blocked_reason,
            consulted=["url"],
            contributed=[],
            unavailable=[],
        )

    consulted = ["url"]
    contributed: list[str] = []
    unavailable: list[str] = []
    live_failed = live.error is not None or live.status_code is None or live.status_code >= 500
    if live_failed:
        unavailable.append("url")
    else:
        contributed.append("url")

    if live.status_code is not None and 200 <= live.status_code < 400:
        final = live.final_url or citation.url
        return _result(
            citation,
            status=CitationStatus.VERIFIED,
            confidence=CONF_URL_LIVE,
            explanation=f"The URL responded with HTTP {live.status_code}.",
            consulted=consulted,
            contributed=contributed,
            unavailable=unavailable,
            links=[Link(label="Fetched URL", url=final or citation.url or "")],
            matched_url=final,
        )

    # Direct fetch did not confirm the page. Wayback is the fallback, not a
    # second opinion we always wait on for pages that already load.
    consulted.append("wayback")
    snapshot = None
    if wayback is None or wayback.unavailable:
        unavailable.append("wayback")
    else:
        contributed.append("wayback")
        snapshot = wayback.snapshot_url

    if snapshot:
        when = _format_timestamp(wayback.timestamp if wayback else None)
        if live.status_code in {404, 410}:
            explanation = (
                f"The URL returned HTTP {live.status_code}, but the Wayback Machine "
                f"has a snapshot from {when}, so the address did point at a real page."
            )
            confidence = CONF_URL_ARCHIVED
        elif live.status_code in {401, 403}:
            explanation = (
                f"The host refused the fetch (HTTP {live.status_code}), but the Wayback "
                f"Machine has a snapshot from {when}."
            )
            confidence = CONF_URL_ARCHIVED_UNCONFIRMED
        else:
            detail = live.error or (
                f"HTTP {live.status_code}" if live.status_code else "a network error"
            )
            explanation = f"The live fetch failed ({detail}), but the Wayback Machine has a snapshot from {when}."
            confidence = CONF_URL_ARCHIVED_UNCONFIRMED
        return _result(
            citation,
            status=CitationStatus.VERIFIED,
            confidence=confidence,
            explanation=explanation,
            consulted=ordered_sources(consulted),
            contributed=ordered_sources(contributed),
            unavailable=ordered_sources(unavailable),
            links=[Link(label="Wayback snapshot", url=snapshot)],
            matched_url=snapshot,
        )

    if live.status_code in {401, 403}:
        return _result(
            citation,
            status=CitationStatus.UNREACHABLE,
            confidence=0.60,
            explanation=(
                f"The host responded HTTP {live.status_code} and the Wayback Machine has no snapshot. "
                "That is a blocked fetch, not evidence the link was fabricated."
            ),
            consulted=ordered_sources(consulted),
            contributed=ordered_sources(contributed),
            unavailable=ordered_sources(unavailable),
        )

    if live.status_code in {404, 410} or (
        live.status_code is not None
        and 400 <= live.status_code < 500
        and live.status_code not in {401, 403}
    ):
        confidence = (
            CONF_URL_NOT_FOUND if "wayback" in contributed else CONF_NOT_FOUND_PARTIAL_OUTAGE
        )
        extra = ""
        if "wayback" in unavailable:
            extra = " The Wayback Machine did not answer, so this is less certain."
        return _result(
            citation,
            status=CitationStatus.NOT_FOUND,
            confidence=confidence,
            explanation=(
                f"The URL returned HTTP {live.status_code}, and the Wayback Machine has no snapshot."
                + extra
            ),
            consulted=ordered_sources(consulted),
            contributed=ordered_sources(contributed),
            unavailable=ordered_sources(unavailable),
        )

    return _result(
        citation,
        status=CitationStatus.UNREACHABLE,
        confidence=CONF_UNREACHABLE,
        explanation=_url_outage_sentence(live, wayback),
        consulted=ordered_sources(consulted),
        contributed=ordered_sources(contributed),
        unavailable=ordered_sources(unavailable),
    )


def _doi_not_registered(
    citation: ExtractedCitation,
    outcomes: list[ProviderOutcome],
    records: list[Record],
    consulted: list[str],
    contributed: list[str],
    unavailable: list[str],
) -> CitationResult:
    explanation = f"doi.org reports that {citation.doi} is not a registered DOI."
    if records:
        explanation += (
            f" {source_label(records[0].source)} still returned a record; the registry "
            "answer is the one used here, so this is treated as not found."
        )
        confidence = CONF_DOI_NOT_FOUND_CONFLICT
    else:
        others = [
            source_label(item.source)
            for item in outcomes
            if item.source != "doi.org" and item.contributed
        ]
        if others:
            explanation += f" {join_and(others)} also had no record."
        if unavailable:
            explanation += (
                f" {join_and([source_label(name) for name in unavailable])} did not answer."
            )
        confidence = CONF_DOI_NOT_FOUND_PARTIAL_OUTAGE if unavailable else CONF_DOI_NOT_FOUND
    return _result(
        citation,
        status=CitationStatus.NOT_FOUND,
        confidence=confidence,
        explanation=explanation,
        consulted=consulted,
        contributed=contributed,
        unavailable=unavailable,
        evidence=_evidence(records, None),
        links=_links(records),
    )


def _doi_explanation(
    *,
    citation: ExtractedCitation,
    primary: Record,
    positive_sources: list[str],
    outcomes: list[ProviderOutcome],
    status: CitationStatus,
) -> str:
    labels = [source_label(name) for name in positive_sources]
    lead = f"DOI {citation.doi} resolves to “{primary.title or 'an untitled work'}” ({_byline(primary)})."
    down = _down_phrase(outcomes)
    if status != CitationStatus.VERIFIED:
        tail = f"Confirmed as a registered DOI by {join_and(labels)}, but the citation details do not match the record."
    elif len(positive_sources) >= 2:
        tail = f"Confirmed by {join_and(labels)}."
    elif down:
        tail = f"Confirmed by {join_and(labels)}. {down}"
    else:
        missing = [
            source_label(item.source)
            for item in outcomes
            if item.contributed and item.source not in positive_sources
        ]
        if missing:
            tail = (
                f"Confirmed by {join_and(labels)}. {join_and(missing)} has no copy of this DOI, "
                "so the cross-check is missing and confidence is lower."
            )
        else:
            tail = f"Confirmed by {join_and(labels)}."
    return f"{lead} {tail}"


def _not_found_or_unreachable(
    citation: ExtractedCitation,
    query: Query,
    outcomes: list[ProviderOutcome],
    consulted: list[str],
    contributed: list[str],
    unavailable: list[str],
) -> CitationResult:
    if not contributed:
        return _result(
            citation,
            status=CitationStatus.UNREACHABLE,
            confidence=CONF_UNREACHABLE,
            explanation=_outage_sentence(outcomes),
            consulted=consulted,
            contributed=contributed,
            unavailable=unavailable,
        )
    closest, closest_score = _closest(query, outcomes)
    title = query.title or citation.raw
    explanation = f"No record in the sources that answered is close enough to “{title}”."
    if closest is not None and closest.title and closest_score >= 0.35:
        year = closest.year or "year unknown"
        percent = round(closest_score * 100)
        explanation += (
            f" Closest was “{closest.title}” ({source_label(closest.source)}, {year}), "
            f"at {percent}% title similarity."
        )
    if unavailable:
        explanation += (
            f" {join_and([source_label(name) for name in unavailable])} did not answer, "
            "so this is less certain."
        )
    return _result(
        citation,
        status=CitationStatus.NOT_FOUND,
        confidence=CONF_NOT_FOUND_PARTIAL_OUTAGE if unavailable else CONF_NOT_FOUND,
        explanation=explanation,
        consulted=consulted,
        contributed=contributed,
        unavailable=unavailable,
    )


def _reference_explanation(
    *,
    status: CitationStatus,
    record: Record,
    support_sources: list[str],
    outcomes: list[ProviderOutcome],
    notes: list[str],
) -> str:
    quoted = f"“{record.title}” ({_byline(record)})"
    labels = [source_label(name) for name in support_sources]
    if status == CitationStatus.VERIFIED:
        sentence = f"Matched {quoted}."
    else:
        sentence = f"Closest record is {quoted}, but at least one cited detail does not match."
    if len(labels) >= 2:
        follow = f"Corroborated by {join_and(labels)}."
    else:
        down = [item for item in outcomes if item.unavailable]
        silent = [
            source_label(item.source)
            for item in outcomes
            if item.contributed and item.source not in support_sources
        ]
        if down:
            follow = f"Found in {labels[0]}. {_down_phrase(outcomes)} This rests on one index."
        elif silent:
            follow = (
                f"Found in {labels[0]}. {join_and(silent)} answered but did not corroborate "
                "this record, so confidence is lower."
            )
        else:
            follow = f"Found in {labels[0]}."
    extra = (" " + " ".join(notes)) if notes else ""
    return f"{sentence} {follow}{extra}"


def _score_all(query: Query, outcomes: list[ProviderOutcome]) -> list[_Scored]:
    scored: list[_Scored] = []
    for outcome in outcomes:
        for record in outcome.records:
            assessment = assess(query, record)
            if assessment is not None:
                scored.append(_Scored(record, assessment))
    return scored


def _rank(item: _Scored) -> tuple[int, float, int]:
    status_rank = 2 if item.assessment.status == CitationStatus.VERIFIED else 1
    return (status_rank, item.assessment.title_blend, -len(item.assessment.mismatches))


def _supporters(best: _Scored, scored: list[_Scored]) -> list[_Scored]:
    found: list[_Scored] = []
    for item in scored:
        if item.record is best.record or item.record.source == best.record.source:
            continue
        if _same_work(best.record, item.record):
            found.append(item)
    return found


def _contradictors(
    best: _Scored, scored: list[_Scored], supporters: list[_Scored]
) -> list[_Scored]:
    supporter_ids = {id(item) for item in supporters}
    found: list[_Scored] = []
    for item in scored:
        if item.record.source == best.record.source or id(item) in supporter_ids:
            continue
        if item.assessment.title_blend >= max(0.86, best.assessment.title_blend - 0.04):
            found.append(item)
    return found


def _same_work(left: Record, right: Record) -> bool:
    left_doi = canonical_doi(left.doi)
    right_doi = canonical_doi(right.doi)
    if left_doi and right_doi and left_doi == right_doi:
        return True
    return bool(left.title and right.title and title_scores(left.title, right.title).blend >= 0.90)


def _edition_notes(best: Record, scored: list[_Scored]) -> list[str]:
    notes: list[str] = []
    seen: set[tuple[str, int]] = set()
    for item in scored:
        if item.record is best or not item.record.title or not best.title:
            continue
        if item.record.year is None or best.year is None or item.record.year == best.year:
            continue
        if title_scores(item.record.title, best.title).blend < 0.94:
            continue
        key = (item.record.source, item.record.year)
        if key in seen:
            continue
        seen.add(key)
        notes.append(
            f"{source_label(item.record.source)} also lists this title with year {item.record.year}."
        )
    return notes


def _closest(query: Query, outcomes: list[ProviderOutcome]) -> tuple[Record | None, float]:
    best: Record | None = None
    best_score = 0.0
    if not query.title:
        return None, 0.0
    for outcome in outcomes:
        for record in outcome.records:
            if not record.title:
                continue
            score = title_scores(query.title, record.title).blend
            if score > best_score:
                best = record
                best_score = score
    return best, best_score


def _reference_confidence(status: CitationStatus, band: str) -> float:
    table = {
        (CitationStatus.VERIFIED, "multi"): CONF_REFERENCE_VERIFIED_MULTI,
        (CitationStatus.VERIFIED, "single_down"): CONF_REFERENCE_VERIFIED_SINGLE_DOWN,
        (CitationStatus.VERIFIED, "single_unconfirmed"): CONF_REFERENCE_VERIFIED_SINGLE,
        (CitationStatus.PARTIALLY_MATCHES, "multi"): CONF_REFERENCE_PARTIAL_MULTI,
        (CitationStatus.PARTIALLY_MATCHES, "single_down"): CONF_REFERENCE_PARTIAL_SINGLE_DOWN,
        (CitationStatus.PARTIALLY_MATCHES, "single_unconfirmed"): CONF_REFERENCE_PARTIAL_SINGLE,
    }
    return table.get((status, band), CONF_REFERENCE_PARTIAL_SINGLE)


def _band(n_support: int, *, any_down: bool) -> str:
    if n_support >= 2:
        return "multi"
    if any_down:
        return "single_down"
    return "single_unconfirmed"


def _query(citation: ExtractedCitation) -> Query:
    return Query(
        title=citation.title,
        surnames=list(citation.surnames),
        year=citation.year,
        venue=citation.venue,
        et_al=citation.et_al or len(citation.surnames) <= 1,
    )


def _byline(record: Record) -> str:
    people = format_people(record.authors, limit=3)
    bits = [bit for bit in (people, str(record.year) if record.year else None, record.venue) if bit]
    return ", ".join(bits) if bits else "no author, year, or venue on the record"


def _evidence(records: list[Record], score: float | None) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    seen: set[tuple[str, str | None]] = set()
    for record in records:
        key = (record.source, canonical_doi(record.doi) or (record.title or "").casefold())
        if key in seen:
            continue
        seen.add(key)
        items.append(
            EvidenceItem(
                source=record.source,
                title=record.title,
                url=record.url,
                authors=list(record.authors),
                year=record.year,
                venue=record.venue,
                doi=record.doi,
                snippet=_byline(record),
                score=round(score, 2) if score is not None else None,
            )
        )
    return items


def _links(records: list[Record]) -> list[Link]:
    links: list[Link] = []
    seen: set[str] = set()

    def add(label: str, url: str | None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        links.append(Link(label=label, url=url))

    for record in records:
        if record.doi:
            add("DOI record", f"https://doi.org/{record.doi}")
        if record.url and record.url.rstrip("/") != f"https://doi.org/{record.doi or ''}".rstrip(
            "/"
        ):
            add(source_label(record.source), record.url)
    return links


def _result(
    citation: ExtractedCitation,
    *,
    status: CitationStatus,
    confidence: float,
    explanation: str,
    mismatches: list[str] | None = None,
    consulted: list[str],
    contributed: list[str],
    unavailable: list[str],
    evidence: list[EvidenceItem] | None = None,
    links: list[Link] | None = None,
    matched: Record | None = None,
    matched_url: str | None = None,
) -> CitationResult:
    return CitationResult(
        raw=citation.raw,
        kind=citation.kind,  # type: ignore[arg-type]
        status=status,
        confidence=round(confidence, 2),
        explanation=explanation,
        mismatches=mismatches or [],
        sources_consulted=ordered_sources(consulted),
        sources_contributed=ordered_sources(contributed),
        sources_unavailable=ordered_sources(unavailable),
        evidence=evidence or [],
        links=links or [],
        matched_title=matched.title if matched else None,
        matched_authors=format_people(matched.authors, limit=4) if matched else None,
        matched_year=matched.year if matched else None,
        matched_venue=matched.venue if matched else None,
        matched_url=matched_url or (matched.url if matched else None),
    )


def _down_phrase(outcomes: list[ProviderOutcome]) -> str:
    parts = [
        f"{source_label(item.source)} was unavailable ({item.error or 'no response'})"
        for item in outcomes
        if item.unavailable
    ]
    if not parts:
        return ""
    return join_and(parts) + "."


def _outage_sentence(outcomes: list[ProviderOutcome]) -> str:
    parts = [
        f"{source_label(item.source)} ({item.error or 'no response'})"
        for item in outcomes
        if item.unavailable
    ]
    if not parts:
        return "The bibliographic sources did not answer, so this citation was not judged real or fabricated."
    return f"{join_and(parts)} did not answer, so this citation was not judged real or fabricated."


def _url_outage_sentence(live: UrlCheck, wayback: WaybackHit | None) -> str:
    live_bit = live.error or (f"HTTP {live.status_code}" if live.status_code else "no response")
    if wayback and wayback.unavailable:
        wayback_bit = wayback.error or "no response"
        return (
            f"Direct fetch failed ({live_bit}) and the Wayback Machine did not answer "
            f"({wayback_bit}), so this URL was not judged real or fabricated."
        )
    return (
        f"Direct fetch failed ({live_bit}) and the Wayback Machine has no snapshot, "
        "so this URL could not be confirmed."
    )


def _format_timestamp(timestamp: str | None) -> str:
    if timestamp and len(timestamp) >= 8 and timestamp[:8].isdigit():
        return f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    return "an unknown date"


def join_and(items: list[str]) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
