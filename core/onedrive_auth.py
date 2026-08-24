"""
Delegated (device-code) auth for the OneDrive/Graph API app registration.

The app registration has a Client ID + Tenant ID but no client secret, which
means it's a public client -- delegated auth via the OAuth device-code flow
is the only option (no app-only/client-credentials flow without a secret).

Device-code sign-in is interactive and blocking (a human visits a URL and
types a code), so it must never run inside a live API request. Instead:

- scripts/onedrive_auth_setup.py runs the device-code flow once (and again
  whenever the cached refresh token fully expires, ~90 days), saving the
  resulting token cache to disk.
- The running server only ever does *silent* token acquisition from that
  cache. If there's no usable cached account, get_access_token() raises a
  clear error telling the admin to (re-)run the setup script, rather than
  hanging the request.
"""

import msal

from api.config import MS_CLIENT_ID, MS_TENANT_ID, ONEDRIVE_TOKEN_CACHE_PATH

AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}"

# Files.Read.All covers both "files shared with me" and, if the tenant has
# granted it, other users' drives -- a superset of what a shared-folder-link
# flow needs. If only a narrower delegated scope was actually consented in
# Azure Portal, the device-code sign-in in the setup script will surface a
# clear AADSTS error naming the missing consent.
SCOPES = ["Files.Read.All"]


class OneDriveAuthError(RuntimeError):
    pass


def _build_app(cache: msal.SerializableTokenCache | None = None) -> msal.PublicClientApplication:
    return msal.PublicClientApplication(
        client_id=MS_CLIENT_ID,
        authority=AUTHORITY,
        token_cache=cache,
    )


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if ONEDRIVE_TOKEN_CACHE_PATH.exists():
        cache.deserialize(ONEDRIVE_TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        ONEDRIVE_TOKEN_CACHE_PATH.write_text(cache.serialize(), encoding="utf-8")


def run_device_code_login() -> None:
    """
    Interactive, blocking sign-in. Called only from
    scripts/onedrive_auth_setup.py -- never from the API server.
    """
    if not MS_CLIENT_ID or not MS_TENANT_ID:
        raise OneDriveAuthError("MS_CLIENT_ID / MS_TENANT_ID are not set in .env")

    cache = _load_cache()
    app = _build_app(cache)

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise OneDriveAuthError(f"Failed to start device-code flow: {flow}")

    print(flow["message"])  # noqa: T201 -- interactive setup script, not the server
    result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise OneDriveAuthError(
            f"Device-code sign-in failed: {result.get('error')}: {result.get('error_description')}"
        )

    _save_cache(cache)


def get_access_token() -> str:
    """
    Silent token acquisition only -- called from request-handling code.
    Raises OneDriveAuthError (never blocks/prompts) if there's no signed-in
    account cached yet, or the cached refresh token itself has expired.
    """
    if not MS_CLIENT_ID or not MS_TENANT_ID:
        raise OneDriveAuthError("MS_CLIENT_ID / MS_TENANT_ID are not set in .env")

    cache = _load_cache()
    app = _build_app(cache)

    accounts = app.get_accounts()
    if not accounts:
        raise OneDriveAuthError(
            "No signed-in OneDrive account found. Run "
            "`python scripts/onedrive_auth_setup.py` once to sign in."
        )

    result = app.acquire_token_silent(SCOPES, account=accounts[0])
    _save_cache(cache)

    if not result or "access_token" not in result:
        raise OneDriveAuthError(
            "OneDrive sign-in has expired. Run "
            "`python scripts/onedrive_auth_setup.py` again to re-authenticate."
        )

    return result["access_token"]
