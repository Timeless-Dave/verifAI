import httpx
import respx
from fastapi.testclient import TestClient
from tests.conftest import make_settings

from verifai.corroboration import Stance
from verifai.main import create_app
from verifai.models.judge import RemoteJudge, build_judge
from verifai.models.parsing import Judgment


class FixedJudge:
    name = "mlx"

    def available(self) -> bool:
        return True

    def judge(self, claim: str, passage: str) -> Judgment:
        assert claim and passage
        return Judgment(Stance.REFUTES, 0.8, "The page says otherwise.")


def test_the_judge_endpoint_needs_its_token() -> None:
    hidden = create_app(make_settings())
    with TestClient(hidden) as client:
        assert client.get("/api/v1/judge").status_code == 404
    app = create_app(make_settings(judge_token="s3cret"))
    with TestClient(app) as client:
        app.state.judge = FixedJudge()
        assert client.get("/api/v1/judge").status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert client.get("/api/v1/judge", headers=bad).status_code == 401
        good = {"Authorization": "Bearer s3cret"}
        assert client.get("/api/v1/judge", headers=good).json() == {"ready": True, "model": "mlx"}
        answer = client.post(
            "/api/v1/judge", json={"claim": "c", "passage": "p"}, headers=good
        ).json()
        assert answer == {
            "stance": "refutes",
            "confidence": 0.8,
            "explanation": "The page says otherwise.",
        }
        assert (
            client.post(
                "/api/v1/judge", json={"claim": "", "passage": "p"}, headers=good
            ).status_code
            == 422
        )


def test_remote_judge_follows_the_host_up_and_down() -> None:
    now = [0.0]
    judge = RemoteJudge("https://mac.example", "s3cret", clock=lambda: now[0])
    with respx.mock() as router:
        ready = router.get("https://mac.example/api/v1/judge").mock(
            return_value=httpx.Response(200, json={"ready": True, "model": "mlx"})
        )
        router.post("https://mac.example/api/v1/judge").mock(
            return_value=httpx.Response(
                200, json={"stance": "supports", "confidence": 0.9, "explanation": "Yes."}
            )
        )
        assert judge.available()
        assert judge.judge("c", "p") == Judgment(Stance.SUPPORTS, 0.9, "Yes.")
        assert ready.call_count == 1  # cached while it is up
        assert ready.calls[0].request.headers["authorization"] == "Bearer s3cret"
    with respx.mock() as router:
        router.post("https://mac.example/api/v1/judge").mock(
            side_effect=httpx.ConnectError("asleep")
        )
        router.get("https://mac.example/api/v1/judge").mock(
            side_effect=httpx.ConnectError("asleep")
        )
        result = judge.judge("c", "p")
        assert result.stance is Stance.NEUTRAL
        assert "did not answer" in result.explanation
        now[0] += 11
        assert judge.available() is False


def test_remote_backend_is_built_from_settings() -> None:
    judge = build_judge(
        make_settings(model_backend="remote", remote_judge_url="https://m", judge_token="t")
    )
    assert isinstance(judge, RemoteJudge)
    assert RemoteJudge("", "t").available() is False


def test_hosted_judge_sends_the_key_and_reads_json() -> None:
    from verifai.models.judge import HostedJudge

    base = "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"
    with respx.mock() as router:
        route = router.post(f"{base}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"stance":"refutes","confidence":0.9,"explanation":"No."}'
                            }
                        }
                    ]
                },
            )
        )
        judge = HostedJudge(base, "@cf/meta/llama-3.1-8b-instruct-fp8-fast", "key123")
        assert judge.available()
        assert judge.judge("c", "p").stance is Stance.REFUTES
        sent = route.calls[0].request
        assert sent.headers["authorization"] == "Bearer key123"
        assert b'"max_tokens"' in sent.content
    assert HostedJudge(base, "m", "").available() is False


def test_qwen3_is_asked_not_to_think() -> None:
    from verifai.models.judge import HostedJudge

    with respx.mock() as router:
        route = router.post("https://h.example/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": '{"stance":"neutral"}'}}]}
            )
        )
        HostedJudge("https://h.example/v1", "@cf/qwen/qwen3-30b-a3b-fp8", "k").judge("c", "p")
        assert b"/no_think" in route.calls[0].request.content
