"""
Checks that OneDrive client-secret auth is working.

    poetry run python scripts/onedrive_auth_setup.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.onedrive_auth import OneDriveAuthError, run_device_code_login


def main() -> None:
    try:
        run_device_code_login()
    except OneDriveAuthError as e:
        print(f"OneDrive auth failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
