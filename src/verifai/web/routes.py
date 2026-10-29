"""HTML form and JSON API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from verifai.orchestrate import CheckBundle, run_check
from verifai.sample import EXAMPLE_TEXT
from verifai.schemas import CheckRequest, CheckResponse, clean_paste, source_label

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
                "combined": None,
                "verdict": None,
                "confidence": None,
                "explanation": "",
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
            "combined": None,
            "verdict": None,
            "confidence": None,
            "explanation": "",
        }
        if error is None:
            try:
                bundle = _run(request, cleaned)
                context["results"] = bundle.results
                context["combined"] = bundle.combined
                context["verdict"] = bundle.verdict
                context["confidence"] = bundle.confidence
                context["explanation"] = bundle.explanation
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
            bundle = _run(request, cleaned)
        except Exception as exc:
            log.exception("citation check failed")
            raise HTTPException(
                status_code=500, detail="The check failed before it could finish."
            ) from exc
        return CheckResponse(
            results=bundle.results,
            combined=bundle.combined,
            verdict=bundle.verdict,
            confidence=bundle.confidence,
            explanation=bundle.explanation,
        )

    @router.get("/health")
    def health(request: Request) -> dict[str, Any]:
        judge = request.app.state.judge
        return {
            "status": "ok",
            "version": request.app.state.settings.app_version,
            "providers": _provider_names(request),
            "experts": [expert.name for expert in request.app.state.router.experts],
            "model": {"name": judge.name, "loaded": judge.available()},
        }

    return router


def _run(request: Request, text: str) -> CheckBundle:
    return run_check(
        request.app.state.router,
        text,
        limit=request.app.state.settings.max_claims,
    )


def _provider_names(request: Request) -> list[str]:
    names: list[str] = []
    for expert in request.app.state.router.experts:
        registry = getattr(expert, "registry", None)
        if registry is None:
            continue
        for name in registry.names():
            if name not in names:
                names.append(name)
    return names


def _validate(text: str, limit: int) -> tuple[str | None, str]:
    try:
        cleaned = clean_paste(text or "")
    except ValueError as exc:
        return str(exc), text or ""
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
