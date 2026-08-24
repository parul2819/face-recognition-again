"""
Lists images in a OneDrive folder given its shared link, using Microsoft
Graph's shares API. This is the real implementation of the pluggable
folder-listing function called for in docs/folder-batch-ingestion.md --
that doc's stub (a Wikimedia Commons category listing) is no longer needed
now that real OneDrive credentials are available.
"""

import base64

import requests

from core.onedrive_auth import get_access_token

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _encode_share_url(shared_url: str) -> str:
    """
    Graph requires a shared link to be base64-encoded into its 'u!' sharing
    token format before it can be resolved to a driveItem:
    https://learn.microsoft.com/en-us/graph/api/shares-get
    """
    b64 = base64.urlsafe_b64encode(shared_url.encode("utf-8")).decode("utf-8")
    return "u!" + b64.rstrip("=")


def _graph_get(url: str) -> dict:
    token = get_access_token()
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if response.status_code != 200:
        # Surface Graph's actual error body (expired token, missing consent,
        # link not shared with the signed-in account, etc.) rather than a
        # generic "request failed" -- this is the most useful signal when
        # something's misconfigured.
        raise RuntimeError(f"Graph API error {response.status_code}: {response.text}")
    return response.json()


def list_folder_images(folder_url: str) -> list[dict]:
    """
    Given a OneDrive shared-folder link, returns every image file directly
    inside it as [{"name": ..., "download_url": ...}, ...]. Non-image files
    and subfolders are skipped (no recursion into subfolders).
    """
    encoded = _encode_share_url(folder_url)
    url = f"{GRAPH_BASE_URL}/shares/{encoded}/driveItem/children"

    images: list[dict] = []
    while url:
        page = _graph_get(url)
        for item in page.get("value", []):
            if "folder" in item:
                continue  # skip subfolders -- no recursion
            name = item.get("name", "")
            if not any(name.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                continue
            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                continue
            images.append({"name": name, "download_url": download_url})
        url = page.get("@odata.nextLink")

    return images
