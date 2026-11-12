"""HTML form and JSON API."""

from __future__ import annotations

import hmac
import json
import logging
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from verifai.orchestrate import CheckBundle, run_check
from verifai.sample import EXAMPLE_TEXT
from verifai.schemas import CheckRequest, CheckResponse, clean_paste, source_label
from verifai.web.throttle import Throttled
from verifai.web.view import build_view

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

    @router.get("/privacy", response_class=HTMLResponse)
    def privacy(request: Request) -> HTMLResponse:
        return _render(request, "privacy.html", {})

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
                context["skipped"] = list(bundle.claims_skipped)
                context["view"] = build_view(
                    bundle, judged=request.app.state.judge.available(), text=cleaned
                )
            except Throttled as refused:
                context["error"] = refused.message
            except Exception:
                log.exception("citation check failed")
                context["error"] = (
                    "The check failed before it could finish. See the server log for the traceback."
                )
        if partial:
            return _render(request, "results.html", context)
        return _render(request, "index.html", context)

    @router.post(
        "/api/v1/check",
        response_model=CheckResponse,
        responses={429: {"description": "Rate limited"}, 503: {"description": "Busy"}},
    )
    def check_api(request: Request, payload: CheckRequest) -> CheckResponse | JSONResponse:
        error, cleaned = _validate(payload.text, request.app.state.settings.max_text_chars)
        if error:
            raise HTTPException(status_code=422, detail=error)
        try:
            bundle = _run(request, cleaned)
        except Throttled as refused:
            return JSONResponse(
                status_code=refused.status,
                content={"detail": refused.message},
                headers={"Retry-After": str(refused.retry_after)},
            )
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
            claims_found=bundle.claims_found,
            claims_checked=bundle.claims_checked,
            claims_skipped=list(bundle.claims_skipped),
        )

    @router.post("/check/more", response_class=HTMLResponse)
    def check_more(
        request: Request,
        text: Annotated[str, Form()] = "",
        offset: Annotated[int, Form(ge=0, le=100_000)] = 0,
        total: Annotated[int, Form(ge=0, le=100_000)] = 0,
    ) -> HTMLResponse:
        """The next batch of claims from a paste, appended where the last one ended."""
        settings = request.app.state.settings
        _, cleaned = _validate(text, settings.max_text_chars)
        claims = [line.strip() for line in cleaned.splitlines() if line.strip()]
        context: dict[str, Any] = {
            "offset": offset,
            "total": max(total, offset + len(claims)),
            "batch": settings.max_claims,
            "skipped": claims,
            "cards": [],
            "paused": "",
        }
        try:
            bundle = _run(request, "\n".join(claims))
        except Throttled as refused:
            context["paused"] = refused.message
            return _render(request, "more.html", context)
        except Exception:
            log.exception("follow-up check failed")
            context["paused"] = "This batch could not finish."
            return _render(request, "more.html", context)
        view = build_view(bundle, judged=request.app.state.judge.available())
        context["cards"] = view.claims
        context["skipped"] = list(bundle.claims_skipped)
        return _render(request, "more.html", context)

    @router.get("/api/v1/judge", include_in_schema=False)
    def judge_ready(
        request: Request, authorization: Annotated[str, Header()] = ""
    ) -> dict[str, Any]:
        """Lets another VerifAI ask whether this one can lend its model."""
        _require_judge_token(request, authorization)
        judge = request.app.state.judge
        return {"ready": judge.available(), "model": judge.name}

    @router.post("/api/v1/judge", include_in_schema=False)
    def judge_passage(
        request: Request,
        payload: JudgeRequest,
        authorization: Annotated[str, Header()] = "",
    ) -> dict[str, Any]:
        """Judge one passage against one claim with this machine's model."""
        _require_judge_token(request, authorization)
        judge = request.app.state.judge
        if not judge.available():
            raise HTTPException(status_code=503, detail="No model is loaded here.")
        # One model, one answer at a time. MLX is not safe to call in parallel.
        with _JUDGE_LOCK:
            result = judge.judge(payload.claim, payload.passage)
        return {
            "stance": result.stance.value,
            "confidence": result.confidence,
            "explanation": result.explanation,
        }

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
    state = request.app.state
    client = _client_address(request)
    state.rate_limiter.admit(client)
    with state.check_slots.hold():
        return run_check(state.router, text, limit=state.settings.max_claims)


_JUDGE_LOCK = threading.Lock()


class JudgeRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=600)
    passage: str = Field(min_length=1, max_length=4000)


def _require_judge_token(request: Request, authorization: str) -> None:
    expected = request.app.state.settings.judge_token
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="A valid judge token is required.")


def _client_address(request: Request) -> str:
    header = request.app.state.settings.client_ip_header
    if header:
        value = request.headers.get(header, "").split(",")[0].strip()
        if value:
            return value
    return request.client.host if request.client else "unknown"


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
