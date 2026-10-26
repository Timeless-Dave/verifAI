from verifai.experts.router import Router


class Named:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name

    def check(self, text: str):
        raise AssertionError(text)


def test_only_citation_expert_always_runs():
    citation = Named("citation")
    router = Router([citation])
    assert router.select("no citations here") == [citation]
    assert router.select("see 10.1000/xyz123") == [citation]


def test_stub_keeps_domain_experts_off_citation_text():
    citation = Named("citation")
    health = Named("health")
    router = Router([citation, health])
    assert router.select("A DOI: 10.1000/xyz123") == [citation]
    assert router.select("How does this symptom work?") == [health]
