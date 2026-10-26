"""Common expert interface.

Input is the pasted claim, chatbot answer, or link. Output is a structured
:class:`~verifai.schemas.ExpertResult`.

Verdicts are computed only from evidence retrieved during the check, never
from a language model's memory.

Evidence providers are independent and pluggable. A verdict aggregates every
provider that answered, records which sources were consulted and which ones
contributed, and must still be produced when any single provider is
unavailable. No provider — including a future Google Fact Check Tools
provider — is allowed to be the sole decider. The citation expert cross-checks
Crossref and OpenAlex rather than trusting either index alone, and also asks
DataCite so a preprint those indexes miss is not judged only by a later
reprint. A verdict is still produced when any one of them is down.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verifai.schemas import ExpertResult


@runtime_checkable
class Expert(Protocol):
    """Domain checker: pasted text in, structured evidence-backed result out.

    Verdicts aggregate evidence from multiple independent sources, record
    which sources were consulted and which ones contributed, and must still
    be produced when any single provider is unavailable. No provider —
    including a future Google Fact Check Tools provider — is the sole
    decider. The citation expert cross-checks Crossref and OpenAlex rather
    than trusting either index alone, and also asks DataCite so a preprint
    those indexes miss is not judged only by a later reprint. A verdict is
    still produced when any one of them is down.
    """

    name: str
    description: str

    def check(self, text: str) -> ExpertResult:
        """Check ``text`` and return a verdict derived only from retrieved evidence."""
        ...
