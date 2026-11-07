import json
from pathlib import Path

from verifai.experts.base import Expert
from verifai.experts.router import Router
from verifai.orchestrate import run_check
from verifai.schemas import ExpertResult, Verdict


class Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name

    def check(self, text: str) -> ExpertResult:
        raise AssertionError(text)


def test_only_citation_expert_always_runs() -> None:
    citation: Expert = Named("citation")
    router = Router([citation])
    assert router.select("no citations here") == [citation]
    assert router.select("see 10.1000/xyz123") == [citation]


def test_routes_a_claim_to_the_matching_experts() -> None:
    experts: list[Expert] = [Named(name) for name in ("citation", "general", "health", "news")]
    router = Router(experts)
    health = "Vitamin D supplements do not prevent fractures in older adults."
    news = "The president won the election by a wide margin last night."
    both = "The president said the vaccine caused the outbreak during the election."
    general = "The Eiffel Tower is 330 metres tall and stands in Paris."
    doi = "A DOI: 10.1000/xyz123"
    cited = "Vitamin D supplements do not prevent fractures. See 10.1000/xyz123."
    assert [expert.name for expert in router.select(health)] == ["health"]
    assert [expert.name for expert in router.select(news)] == ["news"]
    assert [expert.name for expert in router.select(both)] == ["health", "news"]
    assert [expert.name for expert in router.select(general)] == ["general"]
    assert [expert.name for expert in router.select(doi)] == ["citation"]
    assert [expert.name for expert in router.select(cited)] == ["citation", "health"]
    assert router.select("") == []


def test_stub_keeps_domain_experts_off_citation_text() -> None:
    citation: Expert = Named("citation")
    health: Expert = Named("health")
    router = Router([citation, health])
    assert router.select("A DOI: 10.1000/xyz123") == [citation]
    assert router.select("How does this symptom work?") == [health]


class Recording:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.seen: list[str] = []

    def check(self, text: str) -> ExpertResult:
        self.seen.append(text)
        return ExpertResult(
            expert=self.name,
            verdict=Verdict.UNVERIFIABLE,
            verdict_scope=self.name,
            confidence=0.0,
            explanation=self.name,
        )


def test_citation_pastes_never_reach_the_claim_experts() -> None:
    experts = [Recording(name) for name in ("citation", "general", "health", "news")]
    router = Router(list[Expert](experts))
    rows = (Path(__file__).parents[1] / "src/verifai/bench/data/citations.jsonl").read_text()
    for line in rows.splitlines():
        text = json.loads(line)["text"]
        bundle = run_check(router, text)
        assert [result.expert for result in bundle.results] == ["citation"], text
        assert bundle.combined == [], text
    assert all(not expert.seen for expert in experts[1:])
    assert len(experts[0].seen) == len(rows.splitlines())


def test_prose_beside_references_is_still_checked() -> None:
    general = Recording("general")
    health = Recording("health")
    router = Router([Recording("citation"), general, health])
    text = (
        "1. Vaswani, A., Shazeer, N., Parmar, N.,\n"
        "Uszkoreit, J. (2017). Attention is all you need.\n"
        "Advances in Neural Information Processing Systems, 30.\n"
        "2. Jinek et al., 2012, Science, A programmable dual-RNA-guided DNA endonuclease\n"
        "\n"
        "The measles vaccine is about 97 percent effective after two doses.\n"
        "The Eiffel Tower is 330 metres tall and stands in Paris. See 10.1000/xyz123.\n"
    )
    bundle = run_check(router, text)
    assert [claim.claim for claim in bundle.combined] == [
        "The measles vaccine is about 97 percent effective after two doses.",
        "The Eiffel Tower is 330 metres tall and stands in Paris.",
    ]
    assert health.seen == ["The measles vaccine is about 97 percent effective after two doses."]
    assert general.seen == ["The Eiffel Tower is 330 metres tall and stands in Paris."]
