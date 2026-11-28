"""Send each claim to the experts that match it.

The citation expert still runs on the whole paste when it looks like a
reference list. Health, news, and general experts run on the claims
themselves. A paste can go to more than one expert.
"""

from __future__ import annotations

import re

from verifai import claims
from verifai.experts.base import Expert

_CITATION_SIGNAL = re.compile(
    r"\b10\.\d{4,9}/|https?://|\(\s*(?:19|20)\d{2}\s*\)|\bet\s+al\.?|"
    # Identifiers a reference list carries on their own: PubMed, PMC, arXiv, ISBN.
    r"\bPMID\b|\bPMC\d{4,}|\barXiv\b|\bISBN\b|"
    # A Vancouver entry: "1. Doll R, Hill AB." or "[1] Devlin, J."
    r"^\s*(?:\[\d{1,3}\]|\d{1,3}[.)])\s+[A-Z][\w'’-]+,?\s+[A-Z]{1,3}\b",
    re.IGNORECASE | re.MULTILINE,
)
_HEALTH = re.compile(
    r"\b(vaccin\w*|cancers?|symptoms?|diseases?|treatments?|clinical|trials?|"
    r"placebos?|doses?|drugs?|medications?|viruses?|covid|diabetes|cholesterol|"
    r"supplements?|meta-analys\w*|systematic reviews?|patients?|diagnos\w*|"
    r"therap\w*|mortality|side effects?|infections?|antibiotics?|"
    r"blood pressure|heart disease|tumou?rs?|vitamins?|common cold|colds|flu|influenza|"
    r"immun\w*|megadoses?|smoking|cigarettes?|autism|nutrition|obesity)\b",
    re.IGNORECASE,
)
_NEWS = re.compile(
    r"\b(elections?|presidents?|prime ministers?|parliaments?|senators?|congress|"
    r"campaigns?|ballots?|votes?|voted|ministers?|governments?|protests?|"
    r"wars?|invasions?|sanctions?|democrats?|republicans?|politic\w*|scandals?|"
    r"indict\w*|white house|kremlin|governors?|legislation|"
    # Who holds an office now, and what just happened: newsrooms know first.
    r"current(?:ly)?|incumbent|midterms?|speaker|popes?|resign\w*|elected|"
    r"invade[sd]?|inaugurat\w*|impeach\w*|tariffs?|executive orders?|"
    r"supreme court|federal|fda|cdc|nih|recalls?|approved)\b",
    re.IGNORECASE,
)


# Research and academic writing: the literature knows these first.
_RESEARCH = re.compile(
    r"\b(?:study|studies|research\w*|participants?|students?|learners?|pupils?|teachers?|"
    r"achievement|academic|learning|cognitive|cognition|interventions?|meta-analys\w*|"
    r"systematic reviews?|experiments?|hypothes\w*|theory|theories|algorithms?|neural|"
    r"networks?|machine learning|deep learning|artificial intelligence|transformers?|"
    r"datasets?|benchmarks?|statistic\w*|significant(?:ly)?|correlat\w*|regression|"
    r"samples?|surveys?|economic\w*|economy|inflation|gdp|polic(?:y|ies)|climate|"
    r"emissions?|carbon|species|ecosystems?|evolution\w*|genes?|genetic\w*|proteins?|"
    r"cells?|neurons?|brain|quantum|particles?|physics|chemistry|psycholog\w*|"
    r"sociolog\w*|linguistic\w*|mindset|motivation|well-?being|productivity|literacy|"
    r"curricul\w*|pedagog\w*|attention mechanisms?|recurren\w*|convolution\w*|"
    r"language models?|accuracy|outperform\w*|empirical\w*|evidence suggests|retention|"
    r"recall|memory|repetition|spaced|practice|scholars?|literature|peer-reviewed)\b",
    re.IGNORECASE,
)


def domains_for(text: str) -> set[str]:
    """Expert names, other than citation, that should see this text."""
    found: set[str] = set()
    if _HEALTH.search(text or ""):
        found.add("health")
    elif _RESEARCH.search(text or ""):
        # Health claims already go to PubMed and Europe PMC.
        found.add("research")
    if _NEWS.search(text or ""):
        found.add("news")
    if not found and claims.extract_claims(text):
        found.add("general")
    return found


class Router:
    def __init__(self, experts: list[Expert], *, source_checker: object | None = None) -> None:
        self.experts = list(experts)
        # Reads a claim against the work the paste cites for it (verifai.source_check).
        self.source_checker = source_checker

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
