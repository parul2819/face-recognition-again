"""
Server-side session auth for the admin panel. POST /admin/login checks
credentials once and hands back a random session token as an HttpOnly
cookie; every protected route depends on require_admin_session, which just
checks that cookie against the in-memory _sessions store. POST /admin/logout
deletes exactly that one session, so the very next request with the old
cookie -- including a reload of /admin.html -- is unauthenticated and needs
a fresh login, without affecting any other logged-in browser.
"""

import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from api.config import ADMIN_PASSWORD, ADMIN_USERNAME

SESSION_COOKIE_NAME = "admin_session"
SESSION_TTL_SECONDS = 12 * 60 * 60  # 12 hours

# token -> expiry (epoch seconds). In-memory is fine here: single-process
# admin panel, and a server restart forcing everyone to log in again is
# acceptable.
_sessions: dict[str, float] = {}

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


def _purge_expired() -> None:
    now = time.time()
    for token in [t for t, expires_at in _sessions.items() if expires_at <= now]:
        _sessions.pop(token, None)


def create_session() -> str:
    _purge_expired()
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL_SECONDS
    return token


def invalidate_session(token: str | None) -> None:
    if token:
        _sessions.pop(token, None)


def require_admin_session(request: Request) -> str:
    """Dependency for every protected admin/search/identify route."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    _purge_expired()
    if not token or token not in _sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token


@router.post("/admin/login")
async def login(payload: LoginRequest, response: Response):
    # secrets.compare_digest instead of == to avoid leaking timing
    # information about how many characters matched.
    username_ok = secrets.compare_digest(payload.username, ADMIN_USERNAME)
    password_ok = secrets.compare_digest(payload.password, ADMIN_PASSWORD)
    if not (username_ok and password_ok):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_session()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return {"status": "logged in"}


@router.post("/admin/logout")
async def logout(request: Request, response: Response):
    invalidate_session(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"status": "logged out"}


@router.get("/admin/session")
async def check_session(_: str = Depends(require_admin_session)):
    """Cheap check the admin page uses on load to decide whether to show
    the login form or the real admin panel."""
    return {"authenticated": True}
