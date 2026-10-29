from verifai.experts.base import Expert
from verifai.experts.router import Router
from verifai.schemas import ExpertResult


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
