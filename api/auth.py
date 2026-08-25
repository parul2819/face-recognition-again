"""
HTTP Basic Auth for the admin panel. Protects every /admin/* API route
(applied at the router level in admin_controller.py) and the admin.html
page itself (applied directly to its route in main.py) -- both under the
same "Basic" auth realm, so the browser's native login prompt appears
once and is then reused automatically for every subsequent request to
either.
"""

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from api.config import ADMIN_PASSWORD, ADMIN_USERNAME

security = HTTPBasic()


def verify_admin_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    # secrets.compare_digest instead of == to avoid leaking timing
    # information about how many characters matched.
    username_ok = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    password_ok = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
