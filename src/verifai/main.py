"""VerifAI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from verifai import __version__
from verifai.config import Settings, get_settings
from verifai.experts.citation import build_citation_expert
from verifai.experts.router import Router
from verifai.http import PoliteHttp
from verifai.web.routes import STATIC_DIR, build_web_router

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = PoliteHttp(settings)
        expert = build_citation_expert(http, settings)
        app.state.settings = settings
        app.state.http = http
        app.state.router = Router([expert])
        log.info(
            "VerifAI %s ready. Providers: %s",
            settings.app_version,
            ", ".join(expert.registry.names()),
        )
        try:
            yield
        finally:
            http.close()

    app = FastAPI(
        title="VerifAI",
        version=settings.app_version,
        summary="Self-hosted citation checks. Verdicts come from retrieved records.",
        lifespan=lifespan,
    )
    app.include_router(build_web_router())
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()


__all__ = ["__version__", "app", "create_app", "run"]
