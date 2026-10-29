"""Send each claim to the experts that match it.

The citation expert still runs on the whole paste when it looks like a
reference list. Health, news, and general experts run on the claims
themselves. A paste can go to more than one expert.
"""

from __future__ import annotations

import re

from verifai.claims import extract_claims
from verifai.experts.base import Expert

_CITATION_SIGNAL = re.compile(
    r"\b10\.\d{4,9}/|https?://|\(\s*(?:19|20)\d{2}\s*\)|\bet\s+al\.?",
    re.IGNORECASE,
)
_HEALTH = re.compile(
    r"\b(vaccin\w*|cancers?|symptoms?|diseases?|treatments?|clinical|trials?|"
    r"placebos?|doses?|drugs?|medications?|viruses?|covid|diabetes|cholesterol|"
    r"supplements?|meta-analys\w*|systematic reviews?|patients?|diagnos\w*|"
    r"therap\w*|mortality|side effects?|infections?|antibiotics?|"
    r"blood pressure|heart disease|tumou?rs?)\b",
    re.IGNORECASE,
)
_NEWS = re.compile(
    r"\b(elections?|presidents?|prime ministers?|parliaments?|senators?|congress|"
    r"campaigns?|ballots?|votes?|voted|ministers?|governments?|protests?|"
    r"wars?|invasions?|sanctions?|democrats?|republicans?|politic\w*|scandals?|"
    r"indict\w*|white house|kremlin|governors?|legislation)\b",
    re.IGNORECASE,
)


def domains_for(text: str) -> set[str]:
    """Expert names, other than citation, that should see this text."""
    found: set[str] = set()
    if _HEALTH.search(text or ""):
        found.add("health")
    if _NEWS.search(text or ""):
        found.add("news")
    if not found and extract_claims(text):
        found.add("general")
    return found


class Router:
    def __init__(self, experts: list[Expert]) -> None:
        self.experts = list(experts)

    def select(self, text: str) -> list[Expert]:
        """Choose experts for this paste or for one extracted claim."""
        text = text or ""
        if len(self.experts) == 1 and self.experts[0].name == "citation":
            return list(self.experts)
        has_citation = bool(_CITATION_SIGNAL.search(text))
        wanted = domains_for(text)
        return [
            expert
            for expert in self.experts
            if (expert.name == "citation" and has_citation) or expert.name in wanted
        ]
