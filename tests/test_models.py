from collections.abc import Callable

import httpx
import pytest
import respx

from verifai.config import Settings
from verifai.corroboration import Stance
from verifai.models.judge import (
    AutoJudge,
    LlamaCppJudge,
    MlxJudge,
    NliJudge,
    NullJudge,
    OllamaJudge,
    build_judge,
    mlx_complete,
    mlx_installed,
    mlx_supported,
    pick_auto,
    probe,
)
from verifai.models.parsing import judgment_prompt, map_nli_label, map_nli_rows, parse_judgment


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"model_backend": "none", "nli_enabled": False}
    values.update(overrides)
    return Settings.model_validate(values)


def test_parse_judgment_reads_json_and_rejects_junk() -> None:
    parsed = parse_judgment(
        'Sure. {"stance":"supports","confidence":0.8,"explanation":"The page says so."}'
    )
    assert parsed.stance is Stance.SUPPORTS
    assert parsed.confidence == 0.8
    assert "page" in parsed.explanation

    missing = parse_judgment("no json here")
    assert missing.stance is Stance.NEUTRAL
    assert "JSON" in missing.explanation

    broken = parse_judgment("{not json")
    assert broken.stance is Stance.NEUTRAL

    array = parse_judgment("[1, 2]")
    assert array.stance is Stance.NEUTRAL

    unknown = parse_judgment('{"stance":"maybe","confidence":"high"}')
    assert unknown.stance is Stance.NEUTRAL
    assert unknown.confidence == 0.0

    high = parse_judgment('{"stance":"refutes","confidence":4}')
    assert high.stance is Stance.REFUTES
    assert high.confidence == 1.0

    low = parse_judgment('{"stance":"false","confidence":-1}')
    assert low.stance is Stance.REFUTES
    assert low.confidence == 0.0


def test_prompt_includes_the_claim_and_clips_it() -> None:
    prompt = judgment_prompt("claim " * 200, "evidence " * 2000)
    assert "claim claim" in prompt
    assert len(prompt) < 5000


def test_nli_labels() -> None:
    entail = map_nli_label("entailment", 0.91)
    assert entail.stance is Stance.SUPPORTS
    weak = map_nli_label("ENTAILMENT", 0.4)
    assert weak.stance is Stance.NEUTRAL
    refute = map_nli_label("contradiction", 0.7)
    assert refute.stance is Stance.REFUTES
    mini = map_nli_label("1", 0.8)
    assert mini.stance is Stance.SUPPORTS
    no = map_nli_label("no", 0.66)
    assert no.stance is Stance.REFUTES
    blank = map_nli_label("   ", 0.9)
    assert blank.stance is Stance.NEUTRAL
    assert "unknown" in blank.explanation

    rows = map_nli_rows(
        [
            {"label": "neutral", "score": 0.2},
            {"label": "entailment", "score": 0.7},
            "skip",
        ]
    )
    assert rows.stance is Stance.SUPPORTS
    single = map_nli_rows({"label": "contradiction", "score": 0.8})
    assert single.stance is Stance.REFUTES
    empty = map_nli_rows([])
    assert empty.stance is Stance.NEUTRAL
    junk = map_nli_rows("nope")
    assert junk.stance is Stance.NEUTRAL
    bad_score = map_nli_rows([{"label": "entailment", "score": "high"}])
    assert bad_score.stance is Stance.NEUTRAL


def test_null_judge_stays_neutral() -> None:
    judge = NullJudge()
    assert judge.available() is False
    result = judge.judge("The tower is tall.", "A long passage.")
    assert result.stance is Stance.NEUTRAL
    assert "No local model" in result.explanation


def test_ollama_judges_json_and_reports_a_bad_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if "bad claim" in body:
            return httpx.Response(200, json={"message": {}})
        assert "Eiffel" in body
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"stance":"supports","confidence":0.7,"explanation":"Yes."}'
                }
            },
        )

    with respx.mock(assert_all_called=False) as router:
        router.post("http://ollama.test/api/chat").mock(side_effect=handler)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        judge = OllamaJudge("http://ollama.test", "qwen3.5:9b", client=client)
        assert judge.available() is True
        good = judge.judge("The Eiffel Tower is in Paris.", "It stands in Paris.")
        assert good.stance is Stance.SUPPORTS
        bad = judge.judge("bad claim", "bad evidence")
        assert bad.stance is Stance.NEUTRAL
        assert "ValueError" in bad.explanation


def test_llamacpp_reads_chat_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if b"empty" in request.read():
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"stance":"refutes","confidence":0.6,"explanation":"No."}'
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = LlamaCppJudge("http://llama.test/v1", "qwen", client=client)
    assert judge.judge("The claim", "evidence").stance is Stance.REFUTES
    assert judge.judge("empty", "empty").stance is Stance.NEUTRAL


def test_generative_http_failure_is_neutral() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, text="down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = OllamaJudge("http://ollama.test", "qwen", client=client)
    result = judge.judge("claim", "passage")
    assert result.stance is Stance.NEUTRAL
    assert "did not answer" in result.explanation


def test_nli_factory_and_failure() -> None:
    def factory(model_id: str) -> Callable[..., object]:
        assert model_id == "test-nli"

        def pipe(payload: dict[str, str]) -> list[dict[str, float | str]]:
            assert payload["text_pair"] == "claim"
            return [{"label": "entailment", "score": 0.8}]

        return pipe

    judge = NliJudge("test-nli", factory=factory)
    assert judge.available() is True
    assert judge.judge("claim", "passage").stance is Stance.SUPPORTS
    assert judge.judge("claim", "passage").stance is Stance.SUPPORTS

    def broken(model_id: str) -> Callable[..., object]:
        del model_id

        def pipe(payload: dict[str, str]) -> None:
            del payload
            raise RuntimeError("weights missing")

        return pipe

    failed = NliJudge("test-nli", factory=broken)
    result = failed.judge("claim", "passage")
    assert result.stance is Stance.NEUTRAL
    assert failed.available() is False
    assert "not installed" in failed.judge("claim", "passage").explanation


def test_nli_disabled_and_missing_package() -> None:
    off = NliJudge("some-model", enabled=False)
    assert off.available() is False
    assert off.judge("c", "p").stance is Stance.NEUTRAL
    missing = NliJudge("some-model")
    assert missing.available() is False


def test_mlx_is_unavailable_without_apple_silicon() -> None:
    assert mlx_supported() is False
    assert mlx_installed() is False
    judge = MlxJudge("mlx-community/Qwen3.5-9B-4bit")
    assert judge.available() is False
    assert judge.judge("c", "p").stance is Stance.NEUTRAL

    def generate(prompt: str) -> str:
        assert "Paris" in prompt
        return '{"stance":"supports","confidence":0.66,"explanation":"Local."}'

    forced = MlxJudge("custom", generate=generate, enabled=True)
    assert forced.available() is True
    assert forced.judge("Paris", "text").explanation == "Local."

    def boom(prompt: str) -> str:
        del prompt
        raise RuntimeError("gpu")

    broken = MlxJudge("custom", generate=boom, enabled=True)
    assert "did not answer" in broken.judge("c", "p").explanation


def test_mlx_load_failure_disables_the_backend() -> None:
    judge = MlxJudge("missing", enabled=True)
    result = judge.judge("claim", "passage")
    assert result.stance is Stance.NEUTRAL
    assert judge.available() is False
    assert "MLX" in result.explanation


def test_probe_and_auto_selection() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json={})
        )
        router.get("http://127.0.0.1:8080/v1/models").mock(return_value=httpx.Response(503))
        assert probe("http://127.0.0.1:11434/api/tags") is True
        assert probe("http://127.0.0.1:8080/v1/models") is False
        router.get("http://127.0.0.1:9/api/tags").mock(side_effect=httpx.ConnectError("nope"))
        assert probe("http://127.0.0.1:9/api/tags") is False

        settings = _settings(
            model_backend="auto",
            nli_enabled=False,
            ollama_base_url="http://127.0.0.1:9",
            llamacpp_base_url="http://127.0.0.1:9/v1",
        )
        router.get("http://127.0.0.1:9/v1/models").mock(side_effect=httpx.ConnectError("nope"))
        chosen = pick_auto(settings)
        assert isinstance(chosen, NullJudge)
        auto = AutoJudge(settings)
        assert auto.available() is False
        assert auto.judge("c", "p").stance is Stance.NEUTRAL
        assert auto.available() is False

        live = _settings(
            model_backend="auto",
            nli_enabled=False,
            ollama_base_url="http://127.0.0.1:11434",
        )
        assert isinstance(pick_auto(live), OllamaJudge)


def test_auto_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> object | None:
        if name == "transformers":
            return object()
        return None

    monkeypatch.setattr("verifai.models.judge.importlib.util.find_spec", find_spec)
    nli_settings = _settings(model_backend="auto", nli_enabled=True)
    assert isinstance(pick_auto(nli_settings), NliJudge)

    monkeypatch.setattr("verifai.models.judge.mlx_supported", lambda: True)
    monkeypatch.setattr("verifai.models.judge.mlx_installed", lambda: True)
    monkeypatch.setattr("verifai.models.judge.importlib.util.find_spec", lambda _name: None)
    assert isinstance(pick_auto(_settings(model_backend="auto", nli_enabled=False)), MlxJudge)
    monkeypatch.setattr("verifai.models.judge.mlx_supported", lambda: False)
    monkeypatch.setattr("verifai.models.judge.mlx_installed", lambda: False)

    with respx.mock(assert_all_called=False) as router:
        router.get("http://llama.only/v1/models").mock(return_value=httpx.Response(200, json={}))
        router.get("http://ollama.down/api/tags").mock(return_value=httpx.Response(503))
        chosen = pick_auto(
            _settings(
                model_backend="auto",
                nli_enabled=False,
                ollama_base_url="http://ollama.down",
                llamacpp_base_url="http://llama.only/v1",
            )
        )
        assert isinstance(chosen, LlamaCppJudge)
        auto = AutoJudge(
            _settings(
                model_backend="auto",
                nli_enabled=False,
                ollama_base_url="http://127.0.0.1:11434",
            )
        )
        router.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json={})
        )
        router.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "content": '{"stance":"supports","confidence":0.5,"explanation":"ok"}'
                    }
                },
            )
        )
        assert auto.available() is True
        assert auto.judge("claim", "passage").stance is Stance.SUPPORTS


def test_mlx_without_a_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from types import SimpleNamespace

    def generate(model: object, tokenizer: object, **kwargs: object) -> str:
        del model, tokenizer
        assert kwargs["prompt"] == "plain"
        return "ok"

    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name == "mlx_lm":
            return SimpleNamespace(generate=generate, load=lambda _model_id: (object(), object()))
        return real(name, package)

    monkeypatch.setattr("verifai.models.judge.importlib.import_module", fake)
    assert mlx_complete("model")("plain") == "ok"


def test_llamacpp_opens_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[object]:
            return ["not-a-dict"]

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
            assert url.endswith("/chat/completions")
            assert json["temperature"] == 0
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr("verifai.models.judge.httpx.Client", FakeClient)
    judge = LlamaCppJudge("http://llama.test/v1", "qwen")
    assert judge.judge("claim", "passage").stance is Stance.NEUTRAL


def test_auto_skips_nli_when_transformers_is_missing() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get("http://127.0.0.1:9/api/tags").mock(side_effect=httpx.ConnectError("nope"))
        router.get("http://127.0.0.1:9/v1/models").mock(side_effect=httpx.ConnectError("nope"))
        chosen = pick_auto(
            _settings(
                model_backend="auto",
                nli_enabled=True,
                ollama_base_url="http://127.0.0.1:9",
                llamacpp_base_url="http://127.0.0.1:9/v1",
            )
        )
    assert isinstance(chosen, NullJudge)


def test_auto_prefers_an_injected_nli_only_through_build() -> None:
    settings = _settings(model_backend="nli", nli_enabled=True)
    judge = build_judge(settings)
    assert isinstance(judge, NliJudge)
    assert isinstance(build_judge(_settings(model_backend="mlx")), MlxJudge)
    assert isinstance(build_judge(_settings(model_backend="ollama")), OllamaJudge)
    assert isinstance(build_judge(_settings(model_backend="llamacpp")), LlamaCppJudge)
    assert isinstance(build_judge(_settings(model_backend="none")), NullJudge)
    assert isinstance(build_judge(_settings(model_backend="auto", nli_enabled=False)), AutoJudge)


def test_llamacpp_rejects_a_choice_without_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = LlamaCppJudge("http://llama.test/v1", "qwen", client=client)
    assert judge.judge("c", "p").stance is Stance.NEUTRAL


def test_mlx_complete_uses_a_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from types import SimpleNamespace

    captured: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "rendered"

    def generate(model: object, tokenizer: object, **kwargs: object) -> str:
        del model, tokenizer
        captured["prompt"] = kwargs["prompt"]
        return '{"stance":"neutral","confidence":0.1,"explanation":"n"}'

    def load(model_id: str) -> tuple[object, Tokenizer]:
        captured["model_id"] = model_id
        return object(), Tokenizer()

    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name == "mlx_lm":
            return SimpleNamespace(generate=generate, load=load)
        return real(name, package)

    monkeypatch.setattr("verifai.models.judge.importlib.import_module", fake)
    complete = mlx_complete("mlx-community/Qwen3.5-9B-4bit")
    assert "neutral" in complete("hello")
    assert captured["model_id"] == "mlx-community/Qwen3.5-9B-4bit"
    assert captured["prompt"] == "rendered"


def test_transformers_pipeline_is_built_from_the_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from types import SimpleNamespace

    from verifai.models.judge import _transformers_pipeline

    sentinel = object()

    def pipeline(task: str, **kwargs: object) -> object:
        assert task == "text-classification"
        assert kwargs["model"] == "some-nli"
        return sentinel

    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> object:
        if name == "transformers":
            return SimpleNamespace(pipeline=pipeline)
        return real(name, package)

    monkeypatch.setattr("verifai.models.judge.importlib.import_module", fake)
    assert _transformers_pipeline("some-nli") is sentinel


def test_nli_without_factory_uses_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = NliJudge("some-nli", enabled=True)

    def pipeline(model_id: str) -> object:
        assert model_id == "some-nli"

        def pipe(payload: dict[str, str]) -> dict[str, float | str]:
            del payload
            return {"label": "neutral", "score": 0.9}

        return pipe

    monkeypatch.setattr("verifai.models.judge._transformers_pipeline", pipeline)
    assert judge.judge("claim", "passage").stance is Stance.NEUTRAL


def test_ollama_opens_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "message": {"content": '{"stance":"supports","confidence":0.5,"explanation":"ok"}'}
            }

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
            assert url.endswith("/api/chat")
            assert json["model"] == "qwen3.5:9b"
            return FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr("verifai.models.judge.httpx.Client", FakeClient)
    judge = OllamaJudge("http://ollama.test", "qwen3.5:9b")
    assert judge.judge("claim", "passage").stance is Stance.SUPPORTS
