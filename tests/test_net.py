import socket

from verifai.net import public_url_problem


def test_rejects_private_and_credentialed_urls():
    assert "not a public" in public_url_problem("http://127.0.0.1/latest/meta-data").lower()
    assert "local name" in public_url_problem("http://localhost/admin").lower()
    assert "password" in public_url_problem("https://user:secret@example.com/paper").lower()
    assert "only http" in public_url_problem("file:///etc/passwd").lower()


def test_dns_failure_is_left_to_the_fetch(monkeypatch):
    def explode(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    assert public_url_problem("https://verifai.invalid/paper", resolve=explode) is None


def test_public_literal_is_allowed():
    assert public_url_problem("https://1.1.1.1/paper") is None
