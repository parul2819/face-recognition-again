"""
HTTP Basic Auth for the admin panel. Protects every /admin/* API route
(applied at the router level in admin_controller.py) and the admin.html
page itself (applied directly to its route in main.py) -- both under the
same Basic auth realm, so the browser's native login prompt appears once
and is then reused automatically for every subsequent request to either.
"""

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from api.config import ADMIN_PASSWORD, ADMIN_USERNAME

# The realm is random per process start (so a server restart alone forces
# a fresh login prompt) and can be rotated again at runtime -- see
# rotate_realm() below for why.
security = HTTPBasic(realm=secrets.token_hex(8))


def _auth_challenge_headers() -> dict[str, str]:
    return {"WWW-Authenticate": f'Basic realm="{security.realm}"'}


def rotate_realm() -> None:
    """
    Basic Auth has no real server-side session to end -- the browser just
    caches whichever credentials last worked for a given (origin, realm)
    pair and keeps resending them, and there's no reliable, cross-browser
    way to make it forget that cache entry directly.

    Changing the realm string sidesteps that: browsers key their Basic Auth
    cache by (origin, realm), so once the realm changes, every future
    request gets challenged under a realm the browser has no cached
    credentials for, forcing it to prompt again -- even though the
    underlying admin/password pair hasn't changed. Called on logout (see
    POST /admin/logout). There's one shared admin login for the whole app,
    not per-browser sessions, so this is intentionally global: it logs out
    every browser holding the old prompt, not just the one that clicked
    "Log out".
    """
    security.realm = secrets.token_hex(8)


def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    # secrets.compare_digest instead of == to avoid leaking timing
    # information about how many characters matched.
    username_ok = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    password_ok = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers=_auth_challenge_headers(),
        )
    return credentials.username
