import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.face_utils import get_face_embeddings_from_array

from api.auth import require_admin_session
from api.config import SEARCH_THRESHOLD
from api.db import get_pool
from api.utils import embedding_to_pgvector

router = APIRouter()


@router.post("/identify", dependencies=[Depends(require_admin_session)])
async def identify_photo(file: UploadFile = File(...)):
    """
    Upload any photo (single or group). Detects every face, matches each
    against the reference persons table, and returns the same photo with a
    bounding box + name label drawn on every face. Faces that don't match
    anyone in the reference list are labeled "Unknown".
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    faces = get_face_embeddings_from_array(img)
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image")

    pool = get_pool()
    async with pool.acquire() as conn:
        for face in faces:
            embedding_str = embedding_to_pgvector(face["embedding"])
            match = await conn.fetchrow(
                """
                SELECT name, 1 - (reference_embedding <=> $1::vector) AS similarity
                FROM persons
                ORDER BY reference_embedding <=> $1::vector
                LIMIT 1
                """,
                embedding_str,
            )

            label = "Unknown"
            if match is not None and float(match["similarity"]) >= SEARCH_THRESHOLD:
                label = match["name"]

            b = face["bbox"]
            x1, y1 = b["x"], b["y"]
            x2, y2 = b["x"] + b["width"], b["y"] + b["height"]
            color = (0, 200, 0) if label != "Unknown" else (0, 0, 220)

            # Scale box thickness, font size, and label padding relative to
            # the face's own size, so a photo full of small/close-together
            # faces doesn't get giant overlapping labels.
            face_size = max(b["width"], b["height"])
            scale = max(0.35, min(1.0, face_size / 180))
            box_thickness = max(1, round(scale * 3))
            font_scale = round(0.45 * scale + 0.15, 2)
            font_thickness = max(1, round(scale * 2))
            pad = max(2, round(4 * scale))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            label_y = max(y1, text_h + pad * 2)
            cv2.rectangle(
                img,
                (x1, label_y - text_h - pad * 2),
                (x1 + text_w + pad * 2, label_y),
                color,
                -1,
            )
            cv2.putText(
                img,
                label,
                (x1 + pad, label_y - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
