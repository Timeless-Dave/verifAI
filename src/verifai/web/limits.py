"""Request body limits. A pasted citation is small; a huge body is rejected early.

The check form may also carry one attached PDF, so a multipart post to it is
allowed that much more.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int, upload_bytes: int = 0) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes
        self.upload_bytes = upload_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        limit = self.max_bytes
        multipart = request.headers.get("content-type", "").startswith("multipart/form-data")
        if multipart and request.url.path == "/check":
            limit += self.upload_bytes
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return _too_large(limit)
        body = await request.body()
        if len(body) > limit:
            return _too_large(limit)
        return await call_next(request)


def _too_large(limit: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body is larger than {limit:,} bytes."},
    )
