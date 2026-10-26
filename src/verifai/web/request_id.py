"""Attach a short id to each HTTP request so log lines can be tied together."""

from __future__ import annotations

import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from verifai.logging import request_id_var

# Starlette sends an unhandled error from outside this middleware, after the
# context var has been reset. The id is also stored on the scope so that
# handler can still put it on the response and on the log line.
REQUEST_ID_SCOPE_KEY = "verifai.request_id"


def request_id_from_scope(scope: Scope) -> str:
    value = scope.get(REQUEST_ID_SCOPE_KEY, "")
    return value if isinstance(value, str) else ""


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:12]
        scope[REQUEST_ID_SCOPE_KEY] = request_id
        token = request_id_var.set(request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)
