import io

import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.config import PROJECT_ROOT

router = APIRouter()


@router.get("/annotated")
async def get_annotated_image(
    blob_path: str = Query(...),
    x: int = Query(...),
    y: int = Query(...),
    w: int = Query(...),
    h: int = Query(...),
):
    full_path = (PROJECT_ROOT / blob_path).resolve()
    if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = cv2.imread(str(full_path))
    if img is None:
        raise HTTPException(status_code=404, detail="Could not read image")

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
