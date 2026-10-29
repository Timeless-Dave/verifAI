"""Pluggable local judges: MLX, Ollama, llama.cpp, and a small NLI model.

Weights are optional. When none are installed the judge reports that it is
unavailable and returns a neutral stance, so a check can still list the
pages it retrieved.
"""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Callable
from typing import Any, Protocol, cast

import httpx

from verifai.config import Settings
from verifai.corroboration import Stance
from verifai.models.parsing import Judgment, judgment_prompt, map_nli_rows, parse_judgment

Complete = Callable[[str], str]


class Judge(Protocol):
    name: str

    def available(self) -> bool:
        """Whether this backend can judge a passage right now."""

    def judge(self, claim: str, passage: str) -> Judgment:
        """Judge ``passage`` against ``claim`` using only that passage."""


class NullJudge:
    """Used when no local model is installed. It never invents a stance."""

    name = "none"

    def available(self) -> bool:
        return False

    def judge(self, claim: str, passage: str) -> Judgment:
        del claim, passage
        return Judgment(
            Stance.NEUTRAL,
            0.0,
            "No local model is loaded, so this passage was not judged.",
        )


class GenerativeJudge:
    def __init__(self, name: str, complete: Complete | None) -> None:
        self.name = name
        self._complete = complete

    def available(self) -> bool:
        return self._complete is not None

    def judge(self, claim: str, passage: str) -> Judgment:
        if self._complete is None:
            return Judgment(Stance.NEUTRAL, 0.0, f"{self.name} is not available.")
        try:
            raw = self._complete(judgment_prompt(claim, passage))
        except (OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            return Judgment(
                Stance.NEUTRAL,
                0.0,
                f"The local model did not answer ({type(exc).__name__}).",
            )
        return parse_judgment(raw)


class NliJudge:
    """DeBERTa MNLI / MiniCheck-style entailment. The pipeline is injected in tests."""

    name = "nli"

    def __init__(
        self,
        model_id: str,
        *,
        factory: Callable[[str], Callable[..., object]] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self._factory = factory
        self._enabled = enabled
        self._pipe: Callable[..., object] | None = None
        self._failed = False

    def available(self) -> bool:
        if self._failed:
            return False
        if self._factory is not None:
            return True
        if self._enabled is not None:
            return self._enabled
        return importlib.util.find_spec("transformers") is not None

    def judge(self, claim: str, passage: str) -> Judgment:
        if not self.available():
            return Judgment(Stance.NEUTRAL, 0.0, "The NLI model is not installed.")
        try:
            rows = self._pipeline()({"text": passage[:1500], "text_pair": claim[:400]})
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self._failed = True
            return Judgment(Stance.NEUTRAL, 0.0, f"The NLI model failed ({type(exc).__name__}).")
        return map_nli_rows(rows)

    def _pipeline(self) -> Callable[..., object]:
        if self._pipe is None:
            if self._factory is None:
                self._pipe = _transformers_pipeline(self.model_id)
            else:
                self._pipe = self._factory(self.model_id)
        return self._pipe


class MlxJudge(GenerativeJudge):
    """MLX on Apple silicon. The default weights are a 4-bit Qwen3.5-9B."""

    def __init__(
        self,
        model_id: str,
        *,
        generate: Complete | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self._generate = generate
        if enabled is None:
            enabled = mlx_supported() and (generate is not None or mlx_installed())
        self._enabled = enabled
        super().__init__("mlx", generate if enabled else None)

    def available(self) -> bool:
        return self._enabled

    def judge(self, claim: str, passage: str) -> Judgment:
        if self._enabled and self._complete is None:
            try:
                self._complete = mlx_complete(self.model_id)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                self._enabled = False
                self._complete = None
                return Judgment(Stance.NEUTRAL, 0.0, f"MLX could not load ({type(exc).__name__}).")
        return super().judge(claim, passage)


class OllamaJudge(GenerativeJudge):
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        super().__init__("ollama", _ollama_complete(self.base_url, model, client))


class LlamaCppJudge(GenerativeJudge):
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        super().__init__("llamacpp", _llamacpp_complete(self.base_url, model, client))


class AutoJudge:
    """Prefer NLI for entailment, then MLX, then Ollama, then llama.cpp."""

    name = "auto"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chosen: Judge | None = None

    def available(self) -> bool:
        return self._pick().available()

    def judge(self, claim: str, passage: str) -> Judgment:
        chosen = self._pick()
        if not chosen.available():
            return NullJudge().judge(claim, passage)
        return chosen.judge(claim, passage)

    def _pick(self) -> Judge:
        if self._chosen is None:
            self._chosen = pick_auto(self._settings)
        return self._chosen


def build_judge(settings: Settings) -> Judge:
    backend = settings.model_backend
    if backend == "none":
        return NullJudge()
    if backend == "nli":
        return NliJudge(settings.nli_model)
    if backend == "mlx":
        return MlxJudge(settings.mlx_model)
    if backend == "ollama":
        return OllamaJudge(settings.ollama_base_url, settings.ollama_model)
    if backend == "llamacpp":
        return LlamaCppJudge(settings.llamacpp_base_url, settings.llamacpp_model)
    return AutoJudge(settings)


def pick_auto(settings: Settings) -> Judge:
    if settings.nli_enabled:
        nli = NliJudge(settings.nli_model)
        if nli.available():
            return nli
    mlx = MlxJudge(settings.mlx_model)
    if mlx.available():
        return mlx
    if probe(f"{settings.ollama_base_url.rstrip('/')}/api/tags"):
        return OllamaJudge(settings.ollama_base_url, settings.ollama_model)
    if probe(f"{settings.llamacpp_base_url.rstrip('/')}/models"):
        return LlamaCppJudge(settings.llamacpp_base_url, settings.llamacpp_model)
    return NullJudge()


def mlx_supported() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def mlx_installed() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def mlx_complete(model_id: str) -> Complete:
    mlx_lm: Any = importlib.import_module("mlx_lm")
    model, tokenizer = mlx_lm.load(model_id)

    def complete(prompt: str) -> str:
        rendered = prompt
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            rendered = apply(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return str(
            mlx_lm.generate(model, tokenizer, prompt=rendered, max_tokens=300, verbose=False)
        )

    return complete


def probe(url: str, *, timeout: float = 0.4) -> bool:
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _transformers_pipeline(model_id: str) -> Callable[..., object]:
    transformers: Any = importlib.import_module("transformers")
    return cast(
        "Callable[..., object]",
        transformers.pipeline(
            "text-classification",
            model=model_id,
            top_k=None,
            truncation=True,
        ),
    )


def _ollama_complete(base_url: str, model: str, client: httpx.Client | None) -> Complete:
    def complete(prompt: str) -> str:
        http = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
        try:
            response = http.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            body = response.json()
        finally:
            if client is None:
                http.close()
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned no text")
        return content

    return complete


def _llamacpp_complete(base_url: str, model: str, client: httpx.Client | None) -> Complete:
    def complete(prompt: str) -> str:
        http = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0))
        try:
            response = http.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            body = response.json()
        finally:
            if client is None:
                http.close()
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("llama.cpp returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("llama.cpp returned no text")
        return content

    return complete
