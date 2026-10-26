"""HTML form and JSON API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from verifai.sample import EXAMPLE_TEXT
from verifai.schemas import CheckRequest, CheckResponse, ExpertResult, source_label

log = logging.getLogger(__name__)

_WEB_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEB_ROOT / "templates"))
templates.env.globals["source_label"] = source_label
templates.env.filters["pct"] = lambda value: f"{round(float(value) * 100)}%"

STATIC_DIR = _WEB_ROOT / "static"


def build_web_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return _render(
            request,
            "index.html",
            {
                "results": None,
                "error": None,
                "text": "",
                "example_json": _example_json(),
                "mailto_is_placeholder": request.app.state.settings.mailto_is_placeholder,
            },
        )

    @router.post("/check", response_class=HTMLResponse)
    def check_form(request: Request, text: Annotated[str, Form()] = "") -> HTMLResponse:
        partial = request.headers.get("hx-request") == "true"
        error, cleaned = _validate(text, request.app.state.settings.max_text_chars)
        context = {
            "results": None,
            "error": error,
            "text": cleaned,
            "example_json": _example_json(),
            "mailto_is_placeholder": request.app.state.settings.mailto_is_placeholder,
        }
        if error is None:
            try:
                context["results"] = _run(request, cleaned)
            except Exception:
                log.exception("citation check failed")
                context["error"] = (
                    "The check failed before it could finish. See the server log for the traceback."
                )
        if partial:
            return _render(request, "results.html", context)
        return _render(request, "index.html", context)

    @router.post("/api/v1/check", response_model=CheckResponse)
    def check_api(request: Request, payload: CheckRequest) -> CheckResponse:
        error, cleaned = _validate(payload.text, request.app.state.settings.max_text_chars)
        if error:
            raise HTTPException(status_code=422, detail=error)
        try:
            results = _run(request, cleaned)
        except Exception as exc:
            log.exception("citation check failed")
            raise HTTPException(
                status_code=500, detail="The check failed before it could finish."
            ) from exc
        return CheckResponse(results=results)

    @router.get("/health")
    def health(request: Request) -> dict[str, Any]:
        expert = request.app.state.router.experts[0]
        return {
            "status": "ok",
            "version": request.app.state.settings.app_version,
            "providers": expert.registry.names(),
        }

    return router


def _run(request: Request, text: str) -> list[ExpertResult]:
    experts = request.app.state.router.select(text)
    if not experts:
        return []
    return [expert.check(text) for expert in experts]


def _validate(text: str, limit: int) -> tuple[str | None, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return "Paste some text first.", ""
    if len(cleaned) > limit:
        return f"That paste is {len(cleaned):,} characters. The limit is {limit:,}.", cleaned
    return None, cleaned


def _example_json() -> str:
    return json.dumps(EXAMPLE_TEXT).replace("<", "\\u003c")


def _render(
    request: Request, name: str, context: dict[str, Any], status_code: int = 200
) -> HTMLResponse:
    try:
        return templates.TemplateResponse(request, name, context, status_code=status_code)
    except TypeError:
        # Starlette before the request-first signature.
        merged = {"request": request, **context}
        return templates.TemplateResponse(name, merged, status_code=status_code)  # type: ignore[arg-type]
