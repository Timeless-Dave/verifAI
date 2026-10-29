from verifai.claims import extract_claims
from verifai.corroboration import Stance
from verifai.experts.claim_check import NOT_MEDICAL_ADVICE
from verifai.experts.general import GeneralExpert
from verifai.experts.health import HealthExpert
from verifai.experts.news import NewsExpert
from verifai.models.judge import NullJudge
from verifai.models.parsing import Judgment
from verifai.orchestrate import combine_claim, rollup
from verifai.retrieval.types import Passage, RetrievalResult
from verifai.schemas import EvidenceItem, ExpertResult, Verdict


class ScriptedRetriever:
    def __init__(self, name: str, result: RetrievalResult) -> None:
        self.name = name
        self.result = result

    def retrieve(self, query: str) -> RetrievalResult:
        del query
        return self.result


class ScriptedJudge:
    name = "script"

    def __init__(self, stance: Stance, confidence: float = 0.8) -> None:
        self.stance = stance
        self.confidence = confidence
        self._on = True

    def available(self) -> bool:
        return self._on

    def judge(self, claim: str, passage: str) -> Judgment:
        del claim, passage
        return Judgment(self.stance, self.confidence, "The passage addresses the claim.")


def _passage(
    source: str,
    family: str,
    title: str,
    *,
    text: str | None = None,
    document_ids: tuple[str, ...] = (),
    speaker: str | None = None,
    publication_type: str | None = None,
    native_stance: Stance | None = None,
    native_confidence: float | None = None,
    rating: str | None = None,
) -> Passage:
    return Passage(
        source=source,
        family=family,
        title=title,
        url=f"https://example.com/{source}",
        text=text or f"{title} says the claim holds.",
        document_ids=document_ids,
        speaker=speaker,
        publication_type=publication_type,
        native_stance=native_stance,
        native_confidence=native_confidence,
        rating=rating,
    )


def test_extract_claims_skips_references_and_links():
    text = (
        "The Eiffel Tower is 330 metres tall and stands in Paris.\n"
        "Smith, J. (2019). A made up paper. Journal of Nowhere.\n"
        "See https://example.com/only-a-link\n"
        "How tall is it really?"
    )
    assert extract_claims(text) == ["The Eiffel Tower is 330 metres tall and stands in Paris."]


def test_general_needs_two_sources_and_links_them():
    wiki = RetrievalResult(
        "wikipedia",
        passages=[_passage("wikipedia", "wikipedia", "Eiffel Tower")],
    )
    data = RetrievalResult(
        "wikidata",
        passages=[_passage("wikidata", "wikidata", "Eiffel Tower")],
    )
    expert = GeneralExpert(
        [ScriptedRetriever("wikipedia", wiki), ScriptedRetriever("wikidata", data)],
        ScriptedJudge(Stance.SUPPORTS),
        google_configured=False,
    )
    result = expert.check("The Eiffel Tower is 330 metres tall and stands in Paris.")
    assert result.verdict is Verdict.SUPPORTED
    assert result.verdict_scope == "claim"
    assert {item.url for item in result.evidence} == {
        "https://example.com/wikipedia",
        "https://example.com/wikidata",
    }
    assert "not configured" in result.explanation
    assert expert.registry.names() == ["wikipedia", "wikidata"]


def test_google_alone_cannot_support_a_claim():
    reviews = RetrievalResult(
        "google_factcheck",
        passages=[
            _passage(
                "google_factcheck",
                "google_factcheck",
                "Review",
                native_stance=Stance.REFUTES,
                native_confidence=0.9,
                rating="False",
                speaker="Desk",
            )
        ],
    )
    expert = GeneralExpert(
        [ScriptedRetriever("google_factcheck", reviews)],
        NullJudge(),
        google_configured=True,
    )
    result = expert.check("The bridge opened in 1889 and carried trains.")
    assert result.verdict is Verdict.UNVERIFIABLE
    assert "one source" in result.explanation
    assert result.evidence[0].speaker == "Desk"
    assert result.evidence[0].url == "https://example.com/google_factcheck"


def test_pages_are_listed_when_no_model_is_loaded():
    pages = RetrievalResult(
        "page",
        passages=[_passage("page", "page:example.com", "A page", text="A long passage.")],
    )
    down = RetrievalResult("wikipedia", unavailable=True, error="timeout")
    expert = GeneralExpert(
        [ScriptedRetriever("wikipedia", down), ScriptedRetriever("page", pages)],
        NullJudge(),
        google_configured=True,
    )
    result = expert.check("The council funded the bridge repair this year.")
    assert result.verdict is Verdict.UNVERIFIABLE
    assert result.evidence[0].stance == "neutral"
    assert "No local model" in result.explanation
    assert "Wikipedia" in result.explanation
    assert "wikipedia" in result.sources_unavailable


def test_health_disclaimer_and_distinct_papers():
    first = RetrievalResult(
        "pubmed",
        passages=[
            _passage(
                "pubmed",
                "pubmed",
                "Review A",
                document_ids=("pmid:1",),
                publication_type="systematic review",
            )
        ],
    )
    second = RetrievalResult(
        "europepmc",
        passages=[_passage("europepmc", "europepmc", "Review B", document_ids=("pmid:2",))],
    )
    expert = HealthExpert(
        [ScriptedRetriever("pubmed", first), ScriptedRetriever("europepmc", second)],
        ScriptedJudge(Stance.REFUTES),
    )
    result = expert.check("Vitamin D supplements prevent fractures in older adults.")
    assert result.verdict is Verdict.UNSUPPORTED
    assert NOT_MEDICAL_ADVICE in result.explanation
    assert "Systematic reviews" in result.explanation


def test_the_same_paper_from_two_indexes_is_not_enough():
    ids = ("pmid:9", "doi:10.1000/same")
    expert = HealthExpert(
        [
            ScriptedRetriever(
                "pubmed",
                RetrievalResult(
                    "pubmed",
                    passages=[
                        _passage(
                            "pubmed", "pubmed", "Shared", text="Same abstract.", document_ids=ids
                        )
                    ],
                ),
            ),
            ScriptedRetriever(
                "europepmc",
                RetrievalResult(
                    "europepmc",
                    passages=[
                        _passage(
                            "europepmc",
                            "europepmc",
                            "Shared",
                            text="Same abstract.",
                            document_ids=ids,
                        )
                    ],
                ),
            ),
        ],
        ScriptedJudge(Stance.SUPPORTS),
    )
    result = expert.check("This drug cures the common cold in a week.")
    assert result.verdict is Verdict.UNVERIFIABLE
    assert NOT_MEDICAL_ADVICE in result.explanation


def test_news_shows_who_said_what():
    outlets = RetrievalResult(
        "gdelt",
        passages=[
            _passage("gdelt", "outlet:alpha.news", "Alpha headline", speaker="alpha.news"),
            _passage("gdelt", "outlet:beta.news", "Beta headline", speaker="beta.news"),
        ],
    )
    expert = NewsExpert(
        [ScriptedRetriever("gdelt", outlets)],
        ScriptedJudge(Stance.SUPPORTS),
        google_configured=False,
    )
    result = expert.check("The president signed the election bill on Tuesday.")
    assert result.verdict is Verdict.SUPPORTED
    assert result.verdict_scope == "news"
    speakers = {item.speaker for item in result.evidence}
    assert speakers == {"alpha.news", "beta.news"}
    assert all(item.url for item in result.evidence)


def test_parallel_retrieve_still_corroborates():
    expert = GeneralExpert(
        [
            ScriptedRetriever(
                "wikipedia",
                RetrievalResult("wikipedia", passages=[_passage("wikipedia", "wikipedia", "A")]),
            ),
            ScriptedRetriever(
                "wikidata",
                RetrievalResult("wikidata", passages=[_passage("wikidata", "wikidata", "B")]),
            ),
        ],
        ScriptedJudge(Stance.SUPPORTS),
        google_configured=True,
        max_concurrency=2,
    )
    assert expert.check("The river flooded the town in 2011.").verdict is Verdict.SUPPORTED


def test_combine_two_experts_and_rollup():
    health = ExpertResult(
        expert="health",
        verdict=Verdict.UNVERIFIABLE,
        verdict_scope="health",
        confidence=0.0,
        explanation="thin",
        evidence=[
            EvidenceItem(
                source="pubmed",
                title="One study",
                url="https://pubmed.ncbi.nlm.nih.gov/1/",
                stance="supports",
                family="pubmed",
                document_ids=["pmid:1"],
                score=0.8,
            )
        ],
    )
    news = ExpertResult(
        expert="news",
        verdict=Verdict.UNVERIFIABLE,
        verdict_scope="news",
        confidence=0.0,
        explanation="thin",
        evidence=[
            EvidenceItem(
                source="gdelt",
                title="A desk",
                url="https://alpha.news/story",
                stance="supports",
                family="outlet:alpha.news",
                speaker="alpha.news",
                score=0.8,
            )
        ],
    )
    combined = combine_claim("The vaccine ended the outbreak.", [health, news])
    assert combined.verdict is Verdict.SUPPORTED
    assert NOT_MEDICAL_ADVICE in combined.explanation
    assert combined.experts == ["health", "news"]
    many_verdict, _confidence, explanation = rollup(
        [combined, combined.model_copy(update={"verdict": Verdict.UNSUPPORTED, "claim": "other"})],
        [],
    )
    assert many_verdict is Verdict.MIXED
    assert "same verdict" in explanation
    unsupported, _, text = rollup(
        [
            combined.model_copy(update={"verdict": Verdict.UNSUPPORTED}),
            combined.model_copy(update={"verdict": Verdict.UNSUPPORTED, "claim": "second"}),
        ],
        [],
    )
    assert unsupported is Verdict.UNSUPPORTED
    assert "refuted" in text
    none_verdict, _, none_text = rollup([], [])
    assert none_verdict is Verdict.UNVERIFIABLE
    assert "No checkable claim" in none_text
