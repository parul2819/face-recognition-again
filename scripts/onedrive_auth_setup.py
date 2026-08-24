"""
One-time (and ~every 90 days, when the refresh token fully expires)
interactive sign-in for OneDrive folder ingestion.

Run this manually:

    poetry run python scripts/onedrive_auth_setup.py

It prints a URL + a short code. Sign in with an account that has access to
whichever OneDrive folders you'll be pulling from (the shared-folder links
must be shared with this account). Once signed in, the resulting token is
cached to disk (ONEDRIVE_TOKEN_CACHE_PATH) and the running API server will
silently refresh it from there -- no further interactive login needed until
that cache's refresh token itself expires.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.onedrive_auth import OneDriveAuthError, run_device_code_login


def main() -> None:
    try:
        run_device_code_login()
    except OneDriveAuthError as e:
        print(f"Sign-in failed: {e}")
        raise SystemExit(1)

    print("Signed in and cached. The API server can now pull OneDrive folders.")


if __name__ == "__main__":
    main()
