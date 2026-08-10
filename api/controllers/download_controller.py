import io
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.remote_image import download_image

from api.config import PROJECT_ROOT
from api.utils import draw_labeled_box

router = APIRouter()


class DownloadBBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DownloadItem(BaseModel):
    # Local-file rows have blob_path set and source_url None; remote
    # (e.g. Wikimedia/OneDrive) rows are the other way around.
    blob_path: Optional[str] = None
    source_url: Optional[str] = None
    bbox: Optional[DownloadBBox] = None


class DownloadZipRequest(BaseModel):
    items: list[DownloadItem]
    annotated: bool = False
    label: Optional[str] = None  # e.g. the searched person's name, drawn on every box


def _filename_for(item: DownloadItem) -> str:
    if item.blob_path:
        return Path(item.blob_path).name
    # Remote URL: use the last path segment, stripping any query string
    # (e.g. Wikimedia's "?utm_source=..." tracking params).
    return Path(urlparse(item.source_url).path).name or "image.jpg"


@router.post("/download_zip")
async def download_zip(payload: DownloadZipRequest = Body(...)):
    """
    Zips up a list of photos and returns the zip for download. Each item is
    either a local file (blob_path) or a remote one (source_url, downloaded
    on demand -- never saved to disk). If annotated=true, draws a box (+
    label, if given) on each photo using the bbox supplied per item --
    used for "Download All" on the search results page, respecting
    whichever view mode is active.
    """
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items to download")

    zip_buffer = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in payload.items:
            raw_bytes: bytes | None = None

            if item.blob_path:
                full_path = (PROJECT_ROOT / item.blob_path).resolve()
                if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
                    continue  # skip missing files rather than failing the whole zip
                if not payload.annotated or item.bbox is None:
                    raw_bytes = full_path.read_bytes()
                else:
                    img = cv2.imread(str(full_path))
            elif item.source_url:
                try:
                    content = download_image(item.source_url)
                except Exception:
                    continue  # skip on any download failure, don't fail the whole zip
                if not payload.annotated or item.bbox is None:
                    raw_bytes = content
                else:
                    np_arr = np.frombuffer(content, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                continue  # neither a local path nor a URL -- nothing to fetch

            if raw_bytes is None:
                # Annotated path: `img` was decoded above, draw the box.
                if img is None:
                    continue
                draw_labeled_box(
                    img, item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height,
                    label=payload.label, matched=True,
                )
                success, buffer = cv2.imencode(".jpg", img)
                if not success:
                    continue
                file_bytes = buffer.tobytes()
            else:
                file_bytes = raw_bytes

            # Avoid filename collisions inside the zip
            name = _filename_for(item)
            final_name = name
            counter = 1
            while final_name in used_names:
                stem = Path(name).stem
                suffix = Path(name).suffix
                final_name = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(final_name)

            zf.writestr(final_name, file_bytes)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=search_results.zip"},
    )
