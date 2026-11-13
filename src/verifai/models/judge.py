"""Pluggable local judges: MLX, Ollama, llama.cpp, and a small NLI model.

Weights are optional. When none are installed the judge reports that it is
unavailable and returns a neutral stance, so a check can still list the
pages it retrieved.
"""

from __future__ import annotations

import importlib.util
import platform
import re
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

import httpx

from verifai.config import Settings
from verifai.corroboration import Stance
from verifai.models.parsing import (
    Judgment,
    agreement_prompt,
    correction_prompt,
    judgment_prompt,
    map_nli_rows,
    parse_correction,
    parse_judgment,
    parse_translation,
    parse_yes,
    translation_prompt,
)

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
    def __init__(
        self, name: str, complete: Complete | None, *, translator: Complete | None = None
    ) -> None:
        self.name = name
        self._complete = complete
        self._translator = translator

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

    def correct(self, claim: str, evidence: list[str]) -> str | None:
        """What the contradicting sources say instead, or None."""
        if self._complete is None or not evidence:
            return None
        try:
            return parse_correction(self._complete(correction_prompt(claim, evidence)), claim)
        except (OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError):
            return None

    def translate(self, text: str) -> str | None:
        """The claim in English, for searching sources. None if it cannot be trusted.

        With a separate translation model, the judge model translates too,
        and the two must make the same claim. In testing, one model alone
        once added a "not" to a Yoruba claim, which turned a false claim
        into a supported one. Disagreement means no translation at all.
        """
        complete = self._translator or self._complete
        if complete is None:
            return None
        try:
            first = parse_translation(complete(translation_prompt(text)))
            if first is None or self._translator is None or self._complete is None:
                return first
            second = parse_translation(self._complete(translation_prompt(text)))
            if second is None:
                return None
            if _plain(first) == _plain(second):
                return first
            return first if parse_yes(self._translator(agreement_prompt(first, second))) else None
        except (OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError):
            return None


def _plain(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def translate_with(judge: object, text: str) -> str | None:
    """Ask any judge that can translate. Others, such as NLI, cannot."""
    translate = getattr(judge, "translate", None)
    if not callable(translate):
        return None
    result = translate(text)
    return result if isinstance(result, str) else None


def correct_with(judge: object, claim: str, evidence: list[str]) -> str | None:
    """Ask any judge that can write a correction. NLI cannot."""
    correct = getattr(judge, "correct", None)
    if not callable(correct):
        return None
    result = correct(claim, evidence)
    return result if isinstance(result, str) else None


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
        # mlx-lm pulls in transformers without PyTorch, and the pipeline
        # cannot run without it.
        return all(importlib.util.find_spec(name) is not None for name in ("transformers", "torch"))

    def judge(self, claim: str, passage: str) -> Judgment:
        if not self.available():
            return Judgment(Stance.NEUTRAL, 0.0, "The NLI model is not installed.")
        try:
            pipe = self._pipeline()
        except Exception as exc:  # noqa: BLE001 - transformers raises many types while loading
            self._failed = True
            return Judgment(
                Stance.NEUTRAL, 0.0, f"The NLI model could not load ({type(exc).__name__})."
            )
        try:
            rows = pipe({"text": passage[:1500], "text_pair": claim[:400]})
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

    def translate(self, text: str) -> str | None:
        if self._enabled and self._complete is None:
            try:
                self._complete = mlx_complete(self.model_id)
            except (OSError, RuntimeError, ValueError, ImportError):
                return None
        return super().translate(text)


class OllamaJudge(GenerativeJudge):
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        super().__init__("ollama", _ollama_complete(self.base_url, model, client))


class HostedJudge(GenerativeJudge):
    """An open-weights model served behind an OpenAI-compatible chat API."""

    # A hosted API answers several requests at once; a local model does not.
    concurrent = True

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        translate_model: str = "",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        translator = (
            _llamacpp_complete(self.base_url, translate_model, client, api_key=api_key)
            if self.base_url and api_key and translate_model and translate_model != model
            else None
        )
        complete = (
            _llamacpp_complete(self.base_url, model, client, api_key=api_key)
            if self.base_url and api_key
            else None
        )
        if complete is not None and _says_no_think(model):
            # Qwen3 reasons before it answers unless told not to, and the
            # reasoning can use up the reply. "/no_think" is its own switch.
            inner = complete

            def complete(prompt: str) -> str:
                return inner(f"{prompt}\n/no_think")

        self._clock = clock
        self._spent_until: float | None = None
        if complete is not None:
            complete = self._watch_quota(complete)
        if translator is not None:
            translator = self._watch_quota(translator)
        super().__init__("hosted", complete, translator=translator)

    def available(self) -> bool:
        """Configured, and not out of its daily allowance."""
        if self._spent_until is not None:
            if self._clock() < self._spent_until:
                return False
            self._spent_until = None
        return super().available()

    def judge(self, claim: str, passage: str) -> Judgment:
        if not self.available():
            return Judgment(Stance.NEUTRAL, 0.0, "The model's daily allowance is used up.")
        return super().judge(claim, passage)

    def _watch_quota(self, inner: Complete) -> Complete:
        """A spent daily allowance pauses verdicts until the next UTC midnight.

        Without this, every claim would come back "unverified" as if no source
        answered. Paused, the page says so and still checks citations.
        """

        def complete(prompt: str) -> str:
            try:
                return inner(prompt)
            except httpx.HTTPStatusError as exc:
                if _out_of_allowance(exc.response):
                    now = self._clock()
                    self._spent_until = now + (86_400 - now % 86_400)
                raise

        return complete


class LlamaCppJudge(GenerativeJudge):
    def __init__(self, base_url: str, model: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        super().__init__("llamacpp", _llamacpp_complete(self.base_url, model, client))


class RemoteJudge:
    """The model of another VerifAI, reached over HTTPS with a shared token.

    The lender may be a laptop that sleeps. Whether it is up is checked at
    most every few seconds, and an unreachable lender makes this judge
    unavailable, so claims say "not judged" instead of failing.
    """

    name = "remote"

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client = client
        self._clock = clock
        self._checked_at: float | None = None
        self._up = False

    def available(self) -> bool:
        if not self.base_url or not self.token:
            return False
        now = self._clock()
        ttl = 30.0 if self._up else 10.0
        if self._checked_at is None or now - self._checked_at > ttl:
            self._up = self._probe()
            self._checked_at = now
        return self._up

    def judge(self, claim: str, passage: str) -> Judgment:
        if not self.available():
            return Judgment(Stance.NEUTRAL, 0.0, "The model host could not be reached.")
        try:
            response = self._http().post(
                f"{self.base_url}/api/v1/judge",
                json={"claim": claim[:600], "passage": passage[:3500]},
                headers=self._headers(),
                timeout=httpx.Timeout(45.0, connect=5.0),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            self._up = False
            self._checked_at = self._clock()
            return Judgment(
                Stance.NEUTRAL, 0.0, f"The model host did not answer ({type(exc).__name__})."
            )
        stance = {item.value: item for item in Stance}.get(str(body.get("stance")), Stance.NEUTRAL)
        confidence = body.get("confidence")
        return Judgment(
            stance,
            float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            str(body.get("explanation") or "")[:400],
        )

    def _probe(self) -> bool:
        try:
            response = self._http().get(
                f"{self.base_url}/api/v1/judge",
                headers=self._headers(),
                timeout=httpx.Timeout(4.0, connect=3.0),
            )
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        try:
            return bool(response.json().get("ready"))
        except ValueError:
            return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(follow_redirects=False)
        return self._client


class AutoJudge:
    """Prefer NLI for entailment, then MLX, then Ollama, then llama.cpp."""

    name = "auto"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chosen: Judge | None = None
        self._skip: set[str] = set()

    def available(self) -> bool:
        return self._pick().available()

    def judge(self, claim: str, passage: str) -> Judgment:
        chosen = self._pick()
        if not chosen.available():
            return NullJudge().judge(claim, passage)
        judgment = chosen.judge(claim, passage)
        if chosen.available() or chosen.name == "none":
            return judgment
        # The backend failed to load, for example weights that cannot be
        # downloaded. Fall through to the next backend instead of staying
        # unjudged for the rest of the process.
        self._skip.add(chosen.name)
        self._chosen = None
        fallback = self._pick()
        if fallback.available():
            return fallback.judge(claim, passage)
        return judgment

    def translate(self, text: str) -> str | None:
        return translate_with(self._pick(), text)

    def correct(self, claim: str, evidence: list[str]) -> str | None:
        return correct_with(self._pick(), claim, evidence)

    def _pick(self) -> Judge:
        if self._chosen is None:
            self._chosen = pick_auto(self._settings, skip=self._skip)
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
    if backend == "remote":
        return RemoteJudge(settings.remote_judge_url, settings.judge_token)
    if backend == "hosted":
        return HostedJudge(
            settings.hosted_base_url,
            settings.hosted_model,
            settings.hosted_api_key,
            translate_model=settings.hosted_translate_model,
        )
    return AutoJudge(settings)


def pick_auto(settings: Settings, *, skip: set[str] | frozenset[str] = frozenset()) -> Judge:
    if settings.nli_enabled and "nli" not in skip:
        nli = NliJudge(settings.nli_model)
        if nli.available():
            return nli
    if "mlx" not in skip:
        mlx = MlxJudge(settings.mlx_model)
        if mlx.available():
            return mlx
    if "ollama" not in skip and probe(f"{settings.ollama_base_url.rstrip('/')}/api/tags"):
        return OllamaJudge(settings.ollama_base_url, settings.ollama_model)
    if "llamacpp" not in skip and probe(f"{settings.llamacpp_base_url.rstrip('/')}/models"):
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
            # Qwen3 and later think before answering by default, and the
            # reasoning uses up the token budget before any JSON appears.
            rendered = apply(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
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
                    "think": False,
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


def _llamacpp_complete(
    base_url: str, model: str, client: httpx.Client | None, *, api_key: str = ""
) -> Complete:
    """OpenAI-style chat completions: llama.cpp locally, or a hosted provider."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    # A one-sentence verdict needs no reasoning pass. Models that reason by
    # default (Gemma 4, newer Qwen) take the switch from their chat template;
    # the first Qwen3 releases take "/no_think" in the prompt instead, and
    # return nothing when sent both.
    body_extra: dict[str, Any] = (
        {} if _says_no_think(model) else {"chat_template_kwargs": {"enable_thinking": False}}
    )
    if api_key:
        body_extra["max_tokens"] = 300

    def complete(prompt: str) -> str:
        http = client or httpx.Client(timeout=httpx.Timeout(45.0, connect=5.0))
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                **body_extra,
            }
            url = f"{base_url}/chat/completions"
            if headers:
                response = http.post(url, json=payload, headers=headers)
            else:
                response = http.post(url, json=payload)
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


def _says_no_think(model: str) -> bool:
    """The first Qwen3 models (qwen3-30b-a3b and kin), switched off in the prompt."""
    return "qwen3-" in model.casefold()


def _out_of_allowance(response: httpx.Response) -> bool:
    """Rate limited, or a provider's daily allowance is spent (Workers AI code 4006)."""
    if response.status_code == 429:
        return True
    try:
        text = response.text.casefold()
    except (httpx.ResponseNotRead, UnicodeDecodeError):
        return False
    return "daily free allocation" in text or '"code":4006' in text.replace(" ", "")
