"""Runtime settings.

Environment variables use the ``VERIFAI_`` prefix. See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    port: int = Field(default=8741, ge=1, le=65535)
    http_timeout: float = Field(default=20.0, gt=0, le=120)
    http_retries: int = Field(default=1, ge=0, le=8)
    http_backoff_seconds: float = Field(default=0.4, ge=0, le=30)
    http_min_interval_seconds: float = Field(default=0.2, ge=0, le=5)
    http_max_response_bytes: int = Field(default=2_000_000, ge=1_024, le=8_000_000)
    cache_ttl_seconds: float = Field(default=60 * 60, ge=0)
    negative_cache_ttl_seconds: float = Field(default=10 * 60, ge=0)
    max_text_chars: int = Field(default=50_000, ge=1, le=200_000)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None

    @property
    def max_body_bytes(self) -> int:
        """Upper bound for an HTTP body. JSON escaping can grow a paste several times."""
        return self.max_text_chars * 6 + 2_048

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
