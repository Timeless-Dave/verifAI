import pytest

from verifai.evidence.base import EvidenceRegistry


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name


def test_registry_keeps_order_and_refuses_a_duplicate() -> None:
    registry = EvidenceRegistry()
    registry.register(_Provider("crossref"))
    registry.register(_Provider("openalex"))
    assert "crossref" in registry
    assert registry.names() == ["crossref", "openalex"]
    assert registry.get("openalex").name == "openalex"
    assert len(registry) == 2
    assert [provider.name for provider in registry] == ["crossref", "openalex"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_Provider("crossref"))
