import io
import zipfile
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.config import PROJECT_ROOT
from api.utils import draw_labeled_box

router = APIRouter()


class DownloadBBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DownloadItem(BaseModel):
    blob_path: str
    bbox: Optional[DownloadBBox] = None


class DownloadZipRequest(BaseModel):
    items: list[DownloadItem]
    annotated: bool = False
    label: Optional[str] = None  # e.g. the searched person's name, drawn on every box


@router.post("/download_zip")
async def download_zip(payload: DownloadZipRequest = Body(...)):
    """
    Zips up a list of photos (by blob_path) and returns the zip for
    download. If annotated=true, draws a box (+ label, if given) on each
    photo using the bbox supplied per item -- used for "Download All" on
    the search results page, respecting whichever view mode is active.
    """
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items to download")

    zip_buffer = io.BytesIO()
    used_names = set()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in payload.items:
            full_path = (PROJECT_ROOT / item.blob_path).resolve()
            if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
                continue  # skip missing files rather than failing the whole zip

            if payload.annotated and item.bbox is not None:
                img = cv2.imread(str(full_path))
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
                file_bytes = full_path.read_bytes()

            # Avoid filename collisions inside the zip
            name = Path(item.blob_path).name
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
