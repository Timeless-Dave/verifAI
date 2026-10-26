"""Stub router.

Later versions will send a claim to health, news, and general experts as
well as the citation expert. Several experts may run on one claim. Their
results stay separate so no single expert is the sole decider.
"""

from __future__ import annotations

import re

from verifai.experts.base import Expert

# A light signal, not the real extractor. The citation expert decides what
# it can actually parse.
_CITATION_SIGNAL = re.compile(
    r"\b10\.\d{4,9}/|https?://|\(\s*(?:19|20)\d{2}\s*\)|\bet\s+al\.?",
    re.IGNORECASE,
)


class Router:
    def __init__(self, experts: list[Expert]) -> None:
        self.experts = list(experts)

    def select(self, text: str) -> list[Expert]:
        """Choose experts for this paste.

        With only the citation expert registered, it always runs, so a paste
        with no citations still gets an explicit "nothing to check" result.
        Once other experts exist, the citation expert runs only when the
        text looks like it contains citations.
        """
        text = text or ""
        has_signal = bool(_CITATION_SIGNAL.search(text))
        chosen: list[Expert] = []
        for expert in self.experts:
            if expert.name == "citation" and (has_signal or len(self.experts) == 1):
                chosen.append(expert)
            elif expert.name != "citation" and not has_signal:
                # Placeholder branch for future domain experts. None are
                # registered in this version, so this does not fire yet.
                chosen.append(expert)
        return chosen
