import io
import re
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.face_utils import get_face_embeddings_from_array

from api.config import IMAGES_DIR, PROJECT_ROOT, REFERENCE_IMAGES_DIR, SEARCH_THRESHOLD
from api.db import get_pool
from api.utils import blob_path_to_url, draw_labeled_box, embedding_to_pgvector

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
    """
    Remove a single reference person. Any test-image faces previously
    matched to them are unmatched (become "Unknown") rather than deleted,
    since matched_person_id has no ON DELETE action of its own.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        person = await conn.fetchrow("SELECT person_id FROM persons WHERE employee_id = $1", employee_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"No person found with employee_id '{employee_id}'")
        async with conn.transaction():
            await conn.execute(
                "UPDATE images SET matched_person_id = NULL WHERE matched_person_id = $1", person["person_id"]
            )
            await conn.execute("DELETE FROM persons WHERE employee_id = $1", employee_id)
    return {"employee_id": employee_id, "deleted": True}


@router.delete("/admin/persons")
async def delete_all_persons(confirm: bool = False):
    """
    Removes every reference person. Any test-image faces that were matched
    to a deleted person are unmatched ("Unknown") rather than deleted
    themselves. Requires ?confirm=true, as a basic safety guard.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes all reference persons. Call again with ?confirm=true to proceed.",
        )

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE images SET matched_person_id = NULL WHERE matched_person_id IS NOT NULL")
            deleted_count = await conn.fetchval(
                "WITH deleted AS (DELETE FROM persons RETURNING 1) SELECT COUNT(*) FROM deleted"
            )
    return {"deleted_count": deleted_count}


@router.get("/admin/persons/{employee_id}/annotated")
async def get_person_annotated(employee_id: str):
    """
    Returns this person's reference photo with a box + their name drawn on
    it, for the admin "Delete Person" list so the photo is unambiguous.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        person = await conn.fetchrow(
            "SELECT name, reference_image_path FROM persons WHERE employee_id = $1",
            employee_id,
        )
    if person is None or not person["reference_image_path"]:
        raise HTTPException(status_code=404, detail=f"No reference photo for '{employee_id}'")

    full_path = (PROJECT_ROOT / person["reference_image_path"]).resolve()
    if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
        raise HTTPException(status_code=404, detail="Reference image file not found on disk")

    img = cv2.imread(str(full_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read reference image file")

    faces = get_face_embeddings_from_array(img)
    if faces:
        b = faces[0]["bbox"]
        draw_labeled_box(img, b["x"], b["y"], b["width"], b["height"], person["name"], matched=True)

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@router.post("/admin/test_images/check_filenames")
async def check_filenames_for_duplicates(filenames: list[str] = Body(..., embed=True)):
    """
    Given a list of filenames about to be uploaded, returns which ones are
    already present in the test image library (by original filename) or
    repeated within the submitted list itself. Called BEFORE the actual
    upload, so the UI can warn the user and let them choose to proceed
    with the remaining files or cancel.
    """
    seen = set()
    intra_batch_dupes = set()
    for name in filenames:
        if name in seen:
            intra_batch_dupes.add(name)
        seen.add(name)

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT original_filename FROM images WHERE original_filename = ANY($1::text[])",
            filenames,
        )
    existing_dupes = {row["original_filename"] for row in rows}

    all_duplicates = sorted(existing_dupes | intra_batch_dupes)
    return {
        "duplicate_filenames": all_duplicates,
        "clean_filenames": [f for f in filenames if f not in all_duplicates],
    }


@router.post("/admin/test_images")
async def add_test_images(files: list[UploadFile] = File(...)):
    """
    Add one or more new test/bulk photos incrementally (does NOT clear
    existing data). Detects faces, matches each against the persons table,
    and inserts into the images table. Assumes duplicate filenames have
    already been filtered out client-side via check_filenames_for_duplicates.
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
                        (image_id, blob_path, original_filename, embedding, matched_person_id,
                         confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                    VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10)
                    """,
                    image_id,
                    blob_path,
                    original_name,
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


@router.get("/admin/test_images")
async def list_test_images():
    """
    Returns every distinct test photo currently in the images table
    (one entry per photo, not per face), for the admin delete panel.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT image_id, blob_path, COUNT(*) AS face_count
            FROM images
            GROUP BY image_id, blob_path
            ORDER BY blob_path
            """
        )
    return {
        "images": [
            {
                "image_id": str(row["image_id"]),
                "blob_path": row["blob_path"],
                "image_url": blob_path_to_url(row["blob_path"]),
                "face_count": row["face_count"],
            }
            for row in rows
        ]
    }


@router.get("/admin/test_images/{image_id}/annotated")
async def get_test_image_annotated(image_id: str):
    """
    Returns a test photo with a box + name drawn on EVERY face already
    stored for it (using the matched_person_id/bbox saved at ingestion --
    no re-detection needed), for the admin "Delete Test Images" grid.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT i.blob_path, i.bbox_x, i.bbox_y, i.bbox_w, i.bbox_h, p.name
            FROM images i
            LEFT JOIN persons p ON p.person_id = i.matched_person_id
            WHERE i.image_id = $1
            """,
            image_id,
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No test image found with id '{image_id}'")

    blob_path = rows[0]["blob_path"]
    full_path = (PROJECT_ROOT / blob_path).resolve()
    if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    img = cv2.imread(str(full_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read image file")

    for row in rows:
        label = row["name"] if row["name"] else "Unknown"
        draw_labeled_box(
            img, row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"],
            label, matched=row["name"] is not None,
        )

    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode annotated image")
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@router.delete("/admin/test_images/{image_id}")
async def delete_test_image(image_id: str):
    """Removes one test photo (all its face rows) and its file on disk."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT blob_path FROM images WHERE image_id = $1 LIMIT 1", image_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"No test image found with id '{image_id}'")

        await conn.execute("DELETE FROM images WHERE image_id = $1", image_id)

    full_path = (PROJECT_ROOT / row["blob_path"]).resolve()
    if str(full_path).startswith(str(PROJECT_ROOT)) and full_path.exists():
        try:
            full_path.unlink()
        except OSError:
            pass

    return {"image_id": image_id, "deleted": True}


@router.delete("/admin/test_images")
async def delete_all_test_images(confirm: bool = False):
    """
    Removes every test photo (all face rows) and their files on disk.
    Requires ?confirm=true, as a basic safety guard.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes all test images. Call again with ?confirm=true to proceed.",
        )

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT blob_path FROM images")
        await conn.execute("DELETE FROM images")

    for row in rows:
        full_path = (PROJECT_ROOT / row["blob_path"]).resolve()
        if str(full_path).startswith(str(PROJECT_ROOT)) and full_path.exists():
            try:
                full_path.unlink()
            except OSError:
                pass

    return {"deleted_count": len(rows)}


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
