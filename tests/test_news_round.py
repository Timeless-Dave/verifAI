"""Newsrooms, government sites, current facts, relative dates, and corrections."""

from __future__ import annotations

import threading
from datetime import date

import httpx

from verifai.corroboration import SourceVote, Stance, corroborate
from verifai.dates import pin_dates
from verifai.experts.claim_check import check_claim, date_only, grounded, spelling_note
from verifai.experts.router import domains_for
from verifai.models.judge import HostedJudge
from verifai.models.parsing import Judgment, correction_prompt, parse_correction
from verifai.retrieval.types import Passage, RetrievalResult
from verifai.retrieval.wikidata import _best, _statement_text
from verifai.schemas import CombinedClaim, Verdict
from verifai.web.view import claim_card


def test_relative_dates_are_pinned_to_the_calendar() -> None:
    day = date(2026, 11, 13)
    pinned = pin_dates("Jensen Huang resigned today.", day)
    assert pinned.dated == "Jensen Huang resigned on 13 November 2026."
    assert pinned.search == "Jensen Huang resigned."
    assert pinned.readings == (("today", "on 13 November 2026"),)
    assert pin_dates("The Fed cut rates yesterday.", day).dated.endswith("on 12 November 2026.")
    assert "week of 9 November 2026" in pin_dates("It launched this week.", day).dated
    assert pin_dates("Inflation fell last month.", day).dated == "Inflation fell in October 2026."
    assert pin_dates("Sales rose last year.", day).dated == "Sales rose in 2025."
    # A newspaper and a TV show are names, not dates.
    named = pin_dates("USA Today covered the Today show.", day)
    assert named.readings == ()


def test_a_correction_must_come_from_the_claim_or_its_sources() -> None:
    sources = ["Jensen Huang stepped down as chief executive on November 12, 2026."]
    assert grounded("Jensen Huang resigned on 12 November 2026.", "Jensen Huang resigned.", sources)
    # A date or a successor the sources never gave is the model's memory.
    assert not grounded("Jensen Huang resigned on 3 March 2026.", "Jensen Huang resigned.", sources)
    assert not grounded("Lisa Su replaced Jensen Huang.", "Jensen Huang resigned.", sources)


def test_a_date_only_correction_is_the_right_event_on_another_day() -> None:
    claim = "Jensen Huang resigned on 13 November 2026."
    assert date_only(claim, "Jensen Huang resigned on 12 November 2026.")
    assert not date_only(claim, claim)
    assert not date_only("The Eiffel Tower is in Rome.", "The Eiffel Tower is in Paris, France.")


def test_a_misspelled_name_is_pointed_out() -> None:
    article = Passage("wikipedia", "wikipedia", "Barack Obama", None, "x", about="Barak Obama")
    note = spelling_note("Barak Obama was born in Hawaii.", [article])
    assert note == (
        "Check the spelling: the sources write “Barack”, so it reads "
        "“Barack Obama was born in Hawaii.”"
    )
    assert spelling_note("Barack Obama was born in Hawaii.", [article]) is None


def test_the_correction_prompt_and_reply() -> None:
    prompt = correction_prompt("Nvidia is based in Seattle.", ["Nvidia is based in Santa Clara."])
    assert "reply NONE" in prompt and "Santa Clara" in prompt
    claim = "Nvidia is based in Seattle."
    assert parse_correction("Nvidia is based in Santa Clara.", claim) == (
        "Nvidia is based in Santa Clara."
    )
    assert parse_correction("NONE", claim) is None
    assert parse_correction("None.", claim) is None
    assert parse_correction(claim, claim) is None


class _Judge:
    """Refutes every passage, writes corrections, and records concurrency."""

    name = "test"
    concurrent = True

    def __init__(self, correction: str | None) -> None:
        self.correction = correction
        self.threads: set[int] = set()
        self.corrected_with: list[str] = []

    def available(self) -> bool:
        return True

    def judge(self, claim: str, passage: str) -> Judgment:
        del claim, passage
        self.threads.add(threading.get_ident())
        return Judgment(Stance.REFUTES, 0.9, "It names a different city.")

    def correct(self, claim: str, evidence: list[str]) -> str | None:
        del claim
        self.corrected_with = evidence
        return self.correction


class _Fixed:
    def __init__(self, name: str, passages: list[Passage]) -> None:
        self.name = name
        self.passages = passages

    def retrieve(self, query: str) -> RetrievalResult:
        del query
        return RetrievalResult(self.name, passages=self.passages)


def _sources() -> list[_Fixed]:
    text = "Nvidia is an American technology company based in Santa Clara, California."
    return [
        _Fixed("wikipedia", [Passage("wikipedia", "wikipedia", "Nvidia", None, text)]),
        _Fixed("wikidata", [Passage("wikidata", "wikidata", "Nvidia", None, text)]),
    ]


def test_a_refuted_claim_says_what_the_sources_say_instead() -> None:
    judge = _Judge("Nvidia is based in Santa Clara, California.")
    result = check_claim(
        expert="general",
        scope="claim",
        claim="Nvidia is based in Seattle.",
        retrievers=_sources(),
        judge=judge,
        workers=4,
    )
    assert result.verdict is Verdict.UNSUPPORTED
    assert result.correction == "Nvidia is based in Santa Clara, California."
    assert judge.corrected_with and "Santa Clara" in judge.corrected_with[0]


def test_a_correction_the_sources_do_not_back_is_dropped() -> None:
    result = check_claim(
        expert="general",
        scope="claim",
        claim="Nvidia is based in Seattle.",
        retrievers=_sources(),
        judge=_Judge("Nvidia is based in Austin, Texas."),
        workers=4,
    )
    assert result.correction is None


def test_a_relative_date_is_judged_as_a_calendar_date() -> None:
    text = "Jensen Huang stepped down as chief executive of Nvidia on November 12, 2026."
    sources = [
        _Fixed("wikipedia", [Passage("wikipedia", "wikipedia", "Jensen Huang", None, text)]),
        _Fixed("wikidata", [Passage("wikidata", "wikidata", "Jensen Huang", None, text)]),
    ]
    result = check_claim(
        expert="news",
        scope="news",
        claim="Jensen Huang resigned today.",
        retrievers=sources,
        judge=_Judge("Jensen Huang resigned on 12 November 2026."),
        workers=1,
    )
    assert "Dates were read from today" in result.explanation
    assert "Right event, wrong date" in result.explanation
    assert result.correction == "Jensen Huang resigned on 12 November 2026."


def test_a_hosted_judge_reads_passages_in_parallel() -> None:
    judge = _Judge(None)
    many = [
        Passage("wikipedia", "wikipedia", f"Nvidia {n}", None, "Nvidia moved from Seattle.")
        for n in range(8)
    ]
    check_claim(
        expert="general",
        scope="claim",
        claim="Nvidia is based in Seattle.",
        retrievers=[_Fixed("wikipedia", many)],
        judge=judge,
        workers=4,
    )
    assert len(judge.threads) > 1
    assert HostedJudge.concurrent is True


def test_an_off_topic_passage_is_not_sent_to_the_model() -> None:
    judge = _Judge(None)
    stray = Passage("wikipedia", "wikipedia", "Iron Maiden", None, "A heavy metal band.")
    check_claim(
        expert="general",
        scope="claim",
        claim="Nvidia is based in Seattle.",
        retrievers=[_Fixed("wikipedia", [stray])],
        judge=judge,
        workers=1,
    )
    assert judge.threads == set()


def test_a_headline_needs_only_one_shared_term() -> None:
    headline = Passage(
        "major_news",
        "outlet:apnews.com",
        "Democrats win the US House and Senate",
        None,
        "Democrats win the US House and Senate\n(Headline from AP News.)",
    )
    judge = _Judge(None)
    result = check_claim(
        expert="news",
        scope="news",
        claim="Democrats take full control of Congress.",
        retrievers=[_Fixed("major_news", [headline])],
        judge=judge,
        workers=1,
    )
    assert result.evidence[0].stance == "refutes"


def test_newsrooms_are_named_in_the_explanation() -> None:
    votes = [
        SourceVote("major_news", "outlet:npr.org", Stance.SUPPORTS, 0.9, title="A", speaker="NPR"),
        SourceVote(
            "major_news", "outlet:nbcnews.com", Stance.SUPPORTS, 0.9, title="B", speaker="NBC News"
        ),
    ]
    explanation = corroborate(votes).explanation
    assert "NPR (“A”)" in explanation and "NBC News (“B”)" in explanation
    assert "Major news outlets" not in explanation


def test_current_affairs_go_to_the_news_sources() -> None:
    for claim in (
        "Leo XIV is the current pope.",
        "Russia invaded Ukraine in 2022.",
        "The FDA approved Wegovy for weight loss.",
        "Democrats win full control of Congress.",
    ):
        assert "news" in domains_for(claim), claim


def _statement(value: str, start: str | None, end: str | None = None) -> dict[str, object]:
    def time(stamp: str) -> list[dict[str, object]]:
        return [{"datavalue": {"type": "time", "value": {"time": stamp, "precision": 11}}}]

    qualifiers: dict[str, object] = {}
    if start:
        qualifiers["P580"] = time(start)
    if end:
        qualifiers["P582"] = time(end)
    return {
        "rank": "normal",
        "mainsnak": {"datavalue": {"type": "string", "value": value}},
        "qualifiers": qualifiers,
    }


def test_wikidata_lists_what_holds_now_first() -> None:
    statements = [
        _statement("George Washington", "+1789-04-30T00:00:00Z", "+1797-03-04T00:00:00Z"),
        _statement("Joe Biden", "+2021-01-20T00:00:00Z", "+2025-01-20T00:00:00Z"),
        _statement("Donald Trump", "+2025-01-20T00:00:00Z"),
    ]
    ordered = [_statement_text(item, {}) for item in _best(statements)]
    assert ordered[0] == "Donald Trump (from 20 January 2025, to the present)"
    assert ordered[1].startswith("Joe Biden")
    assert ordered[-1].startswith("George Washington")


def test_the_card_shows_the_correction() -> None:
    claim = CombinedClaim(
        claim="Nvidia is based in Seattle.",
        verdict=Verdict.UNSUPPORTED,
        confidence=0.8,
        explanation="Refuted. Right event, wrong date: the sources report it, but on another day.",
        correction="Nvidia is based in Santa Clara, California.",
    )
    card = claim_card(claim)
    assert card.correction == "Nvidia is based in Santa Clara, California."
    assert card.correction_label == "What the sources say"
    assert "Right event, wrong date" in card.flags
    spelled = claim_card(
        claim.model_copy(update={"correction": "Check the spelling: the sources write “Barack”"})
    )
    assert spelled.correction_label == "Check the spelling"
    assert spelled.correction == "the sources write “Barack”"


def test_reasoning_models_are_switched_off_their_own_way() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"stance":"neutral","confidence":0.5}'}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    HostedJudge("https://ai.example/v1", "@cf/google/gemma-4-26b-a4b-it", "k", client=client).judge(
        "c", "p"
    )
    HostedJudge("https://ai.example/v1", "@cf/qwen/qwen3-30b-a3b-fp8", "k", client=client).judge(
        "c", "p"
    )
    gemma, qwen = sent
    assert gemma["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in qwen
    messages = qwen["messages"]
    assert isinstance(messages, list) and messages[0]["content"].endswith("/no_think")


def test_a_spent_daily_allowance_pauses_verdicts_until_midnight() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "errors": [{"message": "you have used up your daily free allocation", "code": 4006}]
            },
        )

    now = [86_400 * 100 + 3_600.0]
    judge = HostedJudge(
        "https://ai.example/v1",
        "@cf/google/gemma-4-26b-a4b-it",
        "k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: now[0],
    )
    assert judge.available()
    assert judge.judge("c", "p").stance is Stance.NEUTRAL
    assert not judge.available()
    # Paused claims do not keep spending requests.
    judge.judge("c", "p")
    assert calls == 1
    now[0] = 86_400 * 101 + 1
    assert judge.available()
