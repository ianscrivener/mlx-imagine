"""Pure ASGI middleware. A Starlette `BaseHTTPMiddleware` passes each response through a memory stream, which is the
wrong shape for the Server-Sent Events of a job (docs/mfluxible-review.md 2.3), so the auth check and the body limit
wrap the app directly. They never touch the response of an allowed request."""

import hmac
import json
from collections.abc import Awaitable, Callable
from typing import Any

from .errors import ApiError

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
PUBLIC_PATHS = frozenset({"/v1/health"})
# The Swagger page and its schema load in a browser, which cannot send the token header. The Authorize button of
# the page sends it on each call.
DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})
MB = 1024 * 1024
JSON_BODY_LIMIT = MB
UPLOAD_PATHS = frozenset({"/v1/uploads", "/v1/images/edits"})
UPLOAD_OVERHEAD = MB  # the multipart boundary and the headers around the file


async def send_error(scope: Scope, receive: Receive, send: Send, error: ApiError) -> None:
    body = json.dumps({"error": error.to_dict()}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": error.status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


class AuthMiddleware:
    """Refuse a request with no token when a token is on, and every /v1 route when the REST API is off."""

    def __init__(self, app: Any, get_token: Callable[[], str | None], api_enabled: bool, swagger: bool) -> None:
        self.app = app
        self.get_token = get_token
        self.api_enabled = api_enabled
        self.swagger = swagger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if not self.api_enabled and path.startswith("/v1/") and path not in PUBLIC_PATHS:
            error = ApiError("api_disabled", 404, "The REST API is off: services.api_enable is false.")
            await send_error(scope, receive, send, error)
            return
        token = self.get_token()
        if token and path not in PUBLIC_PATHS and not (self.swagger and path in DOCS_PATHS):
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, f"Bearer {token}".encode()):
                error = ApiError("unauthorized", 401, "The token is missing or not correct.")
                await send_error(scope, receive, send, error)
                return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Refuse a request body that is too large, before the app reads it. JSON bodies have a small limit. An upload
    may be as large as `max_upload_mb`."""

    def __init__(self, app: Any, get_upload_mb: Callable[[], int]) -> None:
        self.app = app
        self.get_upload_mb = get_upload_mb

    def limit(self, path: str) -> int:
        return (self.get_upload_mb() * MB + UPLOAD_OVERHEAD) if path in UPLOAD_PATHS else JSON_BODY_LIMIT

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit(scope["path"])
        error = ApiError("body_too_large", 413, f"The request body is larger than {limit // MB} MB.")
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await send_error(scope, receive, send, error)
            return
        received = 0
        refused = False

        async def counting_receive() -> dict[str, Any]:
            nonlocal received, refused
            if refused:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # FastAPI turns an exception from receive() into HTTP 400, so the 413 goes out here, and the
                    # app sees a disconnect that stops its read.
                    refused = True
                    await send_error(scope, receive, send, error)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: dict[str, Any]) -> None:
            if not refused:
                await send(message)

        await self.app(scope, counting_receive, guarded_send)
