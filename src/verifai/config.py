"""Runtime settings.

Environment variables use the ``VERIFAI_`` prefix. See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration. Secrets are optional; the citation checker runs without them."""

    model_config = SettingsConfigDict(
        env_prefix="VERIFAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = "0.1.0"
    mailto: str = "verifai-dev@example.com"
    host: str = "127.0.0.1"
    port: int = 8741
    http_timeout: float = 20.0
    http_retries: int = 1
    http_backoff_seconds: float = 0.4
    cache_ttl_seconds: float = 60 * 60
    negative_cache_ttl_seconds: float = 10 * 60
    max_text_chars: int = 50_000
    max_concurrency: int = 4
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None

    @property
    def user_agent(self) -> str:
        """Crossref's polite pool reads the mailto address from the User-Agent."""
        return f"VerifAI/{self.app_version} (mailto:{self.mailto})"

    @property
    def mailto_is_placeholder(self) -> bool:
        return self.mailto.endswith("@example.com")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
