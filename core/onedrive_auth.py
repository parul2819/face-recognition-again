"""
App-only (client secret) auth for the OneDrive/Graph API app registration.

The server gets its own token using MS_CLIENT_ID + MS_TENANT_ID +
MS_CLIENT_SECRET. No user sign-in and no token cache file are needed.
MSAL keeps the token in memory and gets a new one when it expires.

Azure needs: a client secret, the Application permission Files.Read.All
(or Sites.Read.All), and admin consent.
"""

import msal

from api.config import MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID

AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}"

# App-only tokens always use the .default scope -- it means "all the
# Application permissions that were granted admin consent in Azure".
SCOPES = ["https://graph.microsoft.com/.default"]


class OneDriveAuthError(RuntimeError):
    pass


_app: msal.ConfidentialClientApplication | None = None


def _get_app() -> msal.ConfidentialClientApplication:
    global _app
    if not MS_CLIENT_ID or not MS_TENANT_ID or not MS_CLIENT_SECRET:
        raise OneDriveAuthError(
            "MS_CLIENT_ID / MS_TENANT_ID / MS_CLIENT_SECRET are not set in .env"
        )
    if _app is None:
        _app = msal.ConfidentialClientApplication(
            client_id=MS_CLIENT_ID,
            authority=AUTHORITY,
            client_credential=MS_CLIENT_SECRET,
        )
    return _app


def get_access_token() -> str:
    """Returns a Graph token. MSAL reuses the cached one until it expires."""
    result = _get_app().acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in result:
        raise OneDriveAuthError(
            f"OneDrive auth failed: {result.get('error')}: "
            f"{result.get('error_description')}"
        )
    return result["access_token"]


def run_device_code_login() -> None:
    get_access_token()
    print("OneDrive client-secret auth is working.")  # noqa: T201