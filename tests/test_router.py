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


def test_stub_keeps_domain_experts_off_citation_text() -> None:
    citation: Expert = Named("citation")
    health: Expert = Named("health")
    router = Router([citation, health])
    assert router.select("A DOI: 10.1000/xyz123") == [citation]
    assert router.select("How does this symptom work?") == [health]
