import re
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.face_utils import get_face_embeddings_from_array

from api.config import IMAGES_DIR, PROJECT_ROOT, REFERENCE_IMAGES_DIR, SEARCH_THRESHOLD
from api.db import get_pool
from api.utils import embedding_to_pgvector

router = APIRouter()

# Threshold used specifically to catch "this photo already belongs to
# someone else in the reference list" during person add/update.
DUPLICATE_PERSON_THRESHOLD = SEARCH_THRESHOLD


def sanitize_for_filename(text: str) -> str:
    """Turn 'Rohit Sharma' into 'Rohit_Sharma', strip anything unsafe for a filename."""
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_-]", "", text)


@router.get("/admin/stats")
async def get_stats():
    """Quick counts for the admin dashboard."""
    pool = get_pool()
    async with pool.acquire() as conn:
        persons_count = await conn.fetchval("SELECT COUNT(*) FROM persons")
        faces_count = await conn.fetchval("SELECT COUNT(*) FROM images")
        photos_count = await conn.fetchval("SELECT COUNT(DISTINCT image_id) FROM images")
    return {
        "persons_count": persons_count,
        "photos_count": photos_count,
        "faces_count": faces_count,
    }


@router.post("/admin/persons")
async def add_or_update_person(
    employee_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    allow_duplicate: bool = Form(False),
):
    """
    Add a new reference person, or update an existing one's photo (same
    employee_id = update, new employee_id = add). Saves the photo under
    pics/reference pics/ and stores the embedding + path in persons table.

    Before adding a NEW employee_id, checks whether this face already
    belongs to a DIFFERENT existing employee_id, to catch accidental
    duplicate uploads (same person added twice under different IDs).
    Pass allow_duplicate=true to force it through anyway.
    """
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    faces = get_face_embeddings_from_array(img)
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded photo")
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail="Multiple faces detected; please upload a solo photo")

    embedding_str = embedding_to_pgvector(faces[0]["embedding"])

    pool = get_pool()
    async with pool.acquire() as conn:
        if not allow_duplicate:
            # Look for the closest existing person, excluding this same
            # employee_id (that case is a legitimate photo update, not a dupe).
            closest = await conn.fetchrow(
                """
                SELECT employee_id, name,
                       1 - (reference_embedding <=> $1::vector) AS similarity
                FROM persons
                WHERE employee_id != $2
                ORDER BY reference_embedding <=> $1::vector
                LIMIT 1
                """,
                embedding_str,
                employee_id,
            )
            if closest is not None and float(closest["similarity"]) >= DUPLICATE_PERSON_THRESHOLD:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"This photo looks like it already belongs to "
                        f"{closest['name']} ({closest['employee_id']}), "
                        f"similarity={float(closest['similarity']):.2f}. "
                        f"If this is a different person, resubmit with allow_duplicate=true."
                    ),
                )

        REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        safe_employee_id = sanitize_for_filename(employee_id)
        safe_name = sanitize_for_filename(name)
        filename = f"{safe_employee_id}_{safe_name}.jpg"
        save_path = REFERENCE_IMAGES_DIR / filename
        cv2.imwrite(str(save_path), img)
        reference_image_path = str(save_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        result = await conn.fetchrow(
            """
            INSERT INTO persons (employee_id, name, reference_embedding, reference_image_path)
            VALUES ($1, $2, $3::vector, $4)
            ON CONFLICT (employee_id)
            DO UPDATE SET name = EXCLUDED.name,
                          reference_embedding = EXCLUDED.reference_embedding,
                          reference_image_path = EXCLUDED.reference_image_path
            RETURNING (xmax = 0) AS inserted
            """,
            employee_id,
            name,
            embedding_str,
            reference_image_path,
        )

    return {
        "employee_id": employee_id,
        "name": name,
        "action": "added" if result["inserted"] else "updated",
    }


@router.delete("/admin/persons/{employee_id}")
async def delete_person(employee_id: str):
    """Remove a single reference person."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM persons WHERE employee_id = $1", employee_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail=f"No person found with employee_id '{employee_id}'")
    return {"employee_id": employee_id, "deleted": True}


@router.post("/admin/test_images")
async def add_test_images(files: list[UploadFile] = File(...)):
    """
    Add one or more new test/bulk photos incrementally (does NOT clear
    existing data). Detects faces, matches each against the persons table,
    and inserts into the images table.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    pool = get_pool()
    total_photos = 0
    total_faces = 0
    total_matched = 0

    async with pool.acquire() as conn:
        for upload in files:
            contents = await upload.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            original_name = upload.filename or "photo.jpg"
            suffix = Path(original_name).suffix or ".jpg"
            unique_name = f"{uuid.uuid4().hex[:8]}_{Path(original_name).stem}{suffix}"
            save_path = IMAGES_DIR / unique_name
            cv2.imwrite(str(save_path), img)
            blob_path = str(save_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

            faces = get_face_embeddings_from_array(img)
            if len(faces) == 0:
                continue

            image_id = str(uuid.uuid4())
            for face in faces:
                embedding_str = embedding_to_pgvector(face["embedding"])
                match = await conn.fetchrow(
                    """
                    SELECT person_id,
                           1 - (reference_embedding <=> $1::vector) AS similarity
                    FROM persons
                    ORDER BY reference_embedding <=> $1::vector
                    LIMIT 1
                    """,
                    embedding_str,
                )

                matched_person_id = None
                similarity = None
                if match is not None:
                    similarity = float(match["similarity"])
                    if similarity >= SEARCH_THRESHOLD:
                        matched_person_id = match["person_id"]
                        total_matched += 1

                b = face["bbox"]
                await conn.execute(
                    """
                    INSERT INTO images
                        (image_id, blob_path, embedding, matched_person_id, confidence,
                         bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9)
                    """,
                    image_id,
                    blob_path,
                    embedding_str,
                    matched_person_id,
                    similarity,
                    b["x"], b["y"], b["width"], b["height"],
                )
                total_faces += 1

            total_photos += 1

    return {
        "photos_added": total_photos,
        "faces_detected": total_faces,
        "faces_matched": total_matched,
    }


@router.delete("/admin/reset")
async def reset_all_data(confirm: bool = False):
    """
    DANGER: wipes both persons and images tables. Requires ?confirm=true
    to actually run, as a basic safety guard against accidental clicks.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes all data. Call again with ?confirm=true to proceed.",
        )

    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE images, persons")

    return {"reset": True}
