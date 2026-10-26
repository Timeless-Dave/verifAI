"""VerifAI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from verifai import __version__
from verifai.config import Settings, get_settings
from verifai.experts.citation import build_citation_expert
from verifai.experts.router import Router
from verifai.http import PoliteHttp
from verifai.logging import configure_logging, request_id_var
from verifai.web.limits import BodySizeLimitMiddleware
from verifai.web.request_id import RequestIdMiddleware, request_id_from_scope
from verifai.web.routes import STATIC_DIR, build_web_router

log = logging.getLogger(__name__)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        log.info("rejected %s %s", request.method, request.url.path)
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = request_id_from_scope(request.scope)
        token = request_id_var.set(request_id) if request_id else None
        try:
            log.error(
                "unhandled %s on %s %s",
                type(exc).__name__,
                request.method,
                request.url.path,
                exc_info=exc,
            )
        finally:
            if token is not None:
                request_id_var.reset(token)
        response = JSONResponse(
            status_code=500,
            content={"detail": "The check failed before it could finish."},
        )
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestIdMiddleware)
    _install_error_handlers(app)
    configure_logging()
    return app


app = create_app()


def run() -> None:
    configure_logging()
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()


__all__ = ["__version__", "app", "create_app", "run"]
