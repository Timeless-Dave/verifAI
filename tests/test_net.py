import socket

from verifai.net import public_url_problem


def _reason(url: str) -> str:
    problem = public_url_problem(url)
    assert problem is not None
    return problem


def test_rejects_private_and_credentialed_urls() -> None:
    assert "not a public" in _reason("http://127.0.0.1/latest/meta-data").lower()
    assert "local name" in _reason("http://localhost/admin").lower()
    assert "password" in _reason("https://user:secret@example.com/paper").lower()
    assert "only http" in _reason("file:///etc/passwd").lower()


def test_dns_failure_is_left_to_the_fetch(monkeypatch):
    def explode(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    assert public_url_problem("https://verifai.invalid/paper", resolve=explode) is None


def test_public_literal_is_allowed():
    assert public_url_problem("https://1.1.1.1/paper") is None
