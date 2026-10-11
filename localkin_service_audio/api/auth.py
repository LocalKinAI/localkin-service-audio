"""
An optional access key for the HTTP servers (`kin audio serve`, `kin mcp --http`).

Clients send it as ``Authorization: Bearer <key>``, which is exactly what an
OpenAI SDK does with its ``api_key``, or as ``X-API-Key: <key>``.
"""
import secrets
from typing import Iterable, Optional

from starlette.responses import JSONResponse

API_KEY_ENV = "LOCALKIN_API_KEY"

# Health checks and the API docs stay reachable without the key.
OPEN_PATHS = frozenset({"/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def presented_key(headers: Iterable) -> str:
    """The key in a request's raw ASGI headers, or ''."""
    found = {k.lower(): v for k, v in headers}
    auth = found.get(b"authorization", b"").decode("latin-1")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return found.get(b"x-api-key", b"").decode("latin-1").strip()


class BearerAuth:
    """ASGI middleware that turns away HTTP and WebSocket requests without the key."""

    def __init__(self, app, api_key: str, open_paths: Optional[Iterable[str]] = None):
        self.app = app
        self.key = api_key.encode()
        self.open_paths = frozenset(OPEN_PATHS if open_paths is None else open_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope["path"] not in self.open_paths:
            if not secrets.compare_digest(presented_key(scope["headers"]).encode(), self.key):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 4401})
                    return
                response = JSONResponse({"detail": "Missing or wrong API key"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
