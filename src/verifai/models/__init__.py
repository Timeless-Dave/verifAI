"""Local model backends. No commercial model API is called."""

from verifai.models.judge import Judge, build_judge
from verifai.models.parsing import Judgment

__all__ = ["Judge", "Judgment", "build_judge"]
