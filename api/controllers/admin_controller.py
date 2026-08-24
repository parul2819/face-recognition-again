import asyncio
import io
import os
import re
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.face_utils import get_face_embeddings_from_array, get_face_embeddings_from_array_bulk
from core.onedrive import list_folder_images
from core.remote_image import download_image

from api import ingestion_jobs
from api.auth import verify_admin_credentials
from api.config import (
    IMAGES_DIR,
    ONEDRIVE_INGEST_CONCURRENCY,
    PROJECT_ROOT,
    REFERENCE_IMAGES_DIR,
    SEARCH_THRESHOLD,
)
from api.db import get_pool
from api.jobs import UploadJob, create_job, get_job
from api.utils import (
    blob_path_to_url,
    draw_labeled_box,
    embedding_to_pgvector,
    parse_employee_id_and_name,
)

# Every route on this router requires HTTP Basic Auth (see api/auth.py) --
# applied once here rather than per-route so nothing new added to this
# file can accidentally be left unprotected.
router = APIRouter(dependencies=[Depends(verify_admin_credentials)])


class PhotoDecodeError(ValueError):
    """A photo's bytes could not be decoded as an image."""


class FaceDetectionError(ValueError):
    """A decoded photo had zero or more-than-one detected faces."""


class DuplicatePersonError(ValueError):
    """A new-employee_id photo matches an existing, different person too closely."""

# How many photos a bulk test-image upload processes at once. Each
# in-flight photo's face detection runs single-threaded (see
# core.face_utils._build_single_threaded_face_app), so this spreads work
# across CPU cores instead of one photo hogging all of them at a time.
BULK_UPLOAD_CONCURRENCY = min(4, os.cpu_count() or 4)

# Threshold used specifically to catch "this photo already belongs to
# someone else in the reference list" during person add/update.
DUPLICATE_PERSON_THRESHOLD = SEARCH_THRESHOLD


def sanitize_for_filename(text: str) -> str:
    """Turn 'Rohit Sharma' into 'Rohit_Sharma', strip anything unsafe for a filename."""
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_-]", "", text)


async def _log_completed_action(target: str, label: str, count: int) -> None:
    """
    Records any admin action (add, update, or delete -- not just OneDrive
    folder pulls) as an already-completed, single-entry ingestion_jobs row,
    so the "View Upload History" list is a full activity log, not just
    upload progress bars. `label` is shown as-is in place of a folder URL.
    """
    pool = get_pool()
    job_id = await ingestion_jobs.create_job(pool, target, label)
    await ingestion_jobs.mark_processing(pool, job_id, count)
    progress = ingestion_jobs.BatchedProgressWriter(pool, job_id)
    for _ in range(count):
        await progress.record_success()
    await progress.flush()
    await ingestion_jobs.mark_completed(pool, job_id)


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


MAX_REFERENCE_PHOTOS = 2


def _decode_and_validate_single_face(image_bytes: bytes, filename: str, *, bulk: bool = False):
    """
    Shared by the direct-upload /admin/persons route and the OneDrive
    reference-persons folder job. `bulk=True` uses the single-threaded
    model instance (safe to call from several concurrent workers at once --
    see core/face_utils.py); the direct-upload route keeps the default
    greedy model since it only ever processes 1-2 photos at a time.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise PhotoDecodeError(f"Could not decode photo '{filename}'")

    detect = get_face_embeddings_from_array_bulk if bulk else get_face_embeddings_from_array
    faces = detect(img)
    if len(faces) == 0:
        raise FaceDetectionError(f"No face detected in photo '{filename}'")
    if len(faces) > 1:
        raise FaceDetectionError(
            f"Multiple faces detected in photo '{filename}'; please use solo photos"
        )
    return img, faces[0]


async def _upsert_person_from_decoded_photos(
    employee_id: str, name: str, decoded: list[tuple], allow_duplicate: bool
) -> dict:
    """
    Shared by the direct-upload /admin/persons route and the OneDrive
    reference-persons folder job. `decoded` is a list of (img, face) pairs,
    each already validated to contain exactly one face (see
    _decode_and_validate_single_face). Averages their embeddings, keeps the
    photo with the largest detected face on disk, and upserts the person.

    Raises DuplicatePersonError instead of HTTPException so both callers can
    decide how to surface it (a 409 for the direct route, a per-file
    ingestion_job_failures row for the folder job).
    """
    avg_embedding = np.mean([face["embedding"] for _, face in decoded], axis=0)
    embedding_str = embedding_to_pgvector(avg_embedding)

    # Largest detected face = clearest/most zoomed-in shot -- best one to
    # keep as the single on-disk reference photo.
    best_img, _ = max(decoded, key=lambda pair: pair[1]["bbox"]["width"] * pair[1]["bbox"]["height"])

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
                raise DuplicatePersonError(
                    f"This photo looks like it already belongs to "
                    f"{closest['name']} ({closest['employee_id']}), "
                    f"similarity={float(closest['similarity']):.2f}. "
                    f"If this is a different person, resubmit with allow_duplicate=true."
                )

        REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        safe_employee_id = sanitize_for_filename(employee_id)
        safe_name = sanitize_for_filename(name)
        filename = f"{safe_employee_id}_{safe_name}.jpg"
        save_path = REFERENCE_IMAGES_DIR / filename
        cv2.imwrite(str(save_path), best_img)
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
        "photos_used": len(decoded),
        "action": "added" if result["inserted"] else "updated",
    }


@router.post("/admin/persons")
async def add_or_update_person(
    employee_id: str = Form(...),
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    allow_duplicate: bool = Form(False),
):
    """
    Add a new reference person, or update an existing one's photo (same
    employee_id = update, new employee_id = add).

    Accepts 1-2 photos (e.g. a couple of camera-captured angles of the same
    person). Each photo must show exactly one face; their embeddings are
    averaged into a single, more robust reference_embedding -- matching
    against several angles this way holds up better than a single photo.
    The photo with the largest detected face is kept on disk as
    reference_image_path (used for the annotated preview and for
    search_by_employee's fresh re-detection).

    Before adding a NEW employee_id, checks whether this face already
    belongs to a DIFFERENT existing employee_id, to catch accidental
    duplicate uploads (same person added twice under different IDs).
    Pass allow_duplicate=true to force it through anyway.
    """
    if len(files) > MAX_REFERENCE_PHOTOS:
        raise HTTPException(
            status_code=400, detail=f"Upload at most {MAX_REFERENCE_PHOTOS} reference photos"
        )

    decoded: list[tuple] = []
    for upload in files:
        contents = await upload.read()
        try:
            img, face = _decode_and_validate_single_face(contents, upload.filename or "photo.jpg")
        except PhotoDecodeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FaceDetectionError as e:
            raise HTTPException(status_code=422, detail=str(e))
        decoded.append((img, face))

    try:
        result = await _upsert_person_from_decoded_photos(employee_id, name, decoded, allow_duplicate)
    except DuplicatePersonError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Record this as a completed action too, so it shows up alongside
    # deletes and OneDrive folder jobs in the admin "View Upload History" list.
    await _log_completed_action("reference_persons", f"{result['action'].capitalize()}: {employee_id} ({name})", 1)

    return result


@router.delete("/admin/persons/{employee_id}")
async def delete_person(employee_id: str):
    """
    Remove a single reference person. Any test-image faces previously
    matched to them are unmatched (become "Unknown") rather than deleted,
    since matched_person_id has no ON DELETE action of its own.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        person = await conn.fetchrow("SELECT person_id, name FROM persons WHERE employee_id = $1", employee_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"No person found with employee_id '{employee_id}'")
        async with conn.transaction():
            await conn.execute(
                "UPDATE images SET matched_person_id = NULL WHERE matched_person_id = $1", person["person_id"]
            )
            await conn.execute("DELETE FROM persons WHERE employee_id = $1", employee_id)

    await _log_completed_action("reference_persons", f"Deleted: {employee_id} ({person['name']})", 1)
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

    await _log_completed_action("reference_persons", f"Deleted all ({deleted_count} person(s))", deleted_count)
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


class OneDriveFolderRequest(BaseModel):
    folder_url: str


class OneDriveTestImagesRequest(OneDriveFolderRequest):
    event_name: str | None = None


async def _run_persons_onedrive_job(job_id: str, folder_url: str) -> None:
    """
    Background job: lists every image in the OneDrive folder, and for each
    one -- one photo per person, no multi-angle averaging like the
    direct-upload route -- parses "<employee_id>_<name>.ext" from its
    filename and upserts it via the same logic add_or_update_person uses.
    Per-file problems (bad filename, no/multiple faces, duplicate-person
    conflict, download failure) are logged to ingestion_job_failures and
    don't stop the rest of the batch.
    """
    pool = get_pool()
    try:
        images = await asyncio.to_thread(list_folder_images, folder_url)
    except Exception as e:
        await ingestion_jobs.mark_failed_outright(pool, job_id, str(e))
        return

    await ingestion_jobs.mark_processing(pool, job_id, len(images))
    progress = ingestion_jobs.BatchedProgressWriter(pool, job_id)
    semaphore = asyncio.Semaphore(ONEDRIVE_INGEST_CONCURRENCY)

    async def _process_one(item: dict) -> None:
        async with semaphore:
            try:
                employee_id, person_name = parse_employee_id_and_name(Path(item["name"]).stem)
                contents = await asyncio.to_thread(download_image, item["download_url"])
                img, face = await asyncio.to_thread(
                    _decode_and_validate_single_face, contents, item["name"], bulk=True
                )
                await _upsert_person_from_decoded_photos(
                    employee_id, person_name, [(img, face)], allow_duplicate=False
                )
            except Exception as e:
                await ingestion_jobs.log_failure(pool, job_id, item["name"], str(e))
                await progress.record_failure()
                return
            await progress.record_success()

    await asyncio.gather(*(_process_one(item) for item in images))
    await progress.flush()
    await ingestion_jobs.mark_completed(pool, job_id)


@router.post("/admin/persons/onedrive")
async def add_persons_from_onedrive(payload: OneDriveFolderRequest = Body(...)):
    """
    Bulk-imports reference persons from a OneDrive folder link. Every file
    in the folder must follow the "<employee_id>_<name>.ext" convention
    (same as ingestion/ingest_reference_images.py for local folders) and
    show exactly one face. Runs as a background job (folders can be
    hundreds of images); poll GET /admin/ingestion_jobs/{job_id} for
    progress -- no auto-polling, the admin UI updates on a manual click.
    """
    pool = get_pool()
    job_id = await ingestion_jobs.create_job(pool, "reference_persons", payload.folder_url)
    asyncio.create_task(_run_persons_onedrive_job(job_id, payload.folder_url))
    return {"job_id": job_id}


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


def _decode_save_and_detect(contents: bytes, save_path: Path) -> tuple[bool, list[dict]]:
    """
    Sync, CPU-bound work for one photo (decode, write to disk, run face
    detection) -- run off the event loop via asyncio.to_thread so a bulk
    upload doesn't freeze the whole server while it runs. Uses the
    single-threaded model instance since several of these can be in flight
    at once (see BULK_UPLOAD_CONCURRENCY). Returns (decoded_ok, faces);
    faces is empty if decode failed or no face was found.
    """
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, []
    cv2.imwrite(str(save_path), img)
    return True, get_face_embeddings_from_array_bulk(img)


async def _process_one_upload_photo(
    job: UploadJob, original_name: str, contents: bytes, event_name: str | None
) -> tuple[bool, str | None]:
    """
    Decodes/saves/detects one photo and inserts its face rows, updating `job`
    as it goes. Saved under its own original filename (not a renamed copy),
    so re-uploading a photo that's already sitting in IMAGES_DIR just
    (re)writes the same path instead of creating a second file alongside it.
    check_filenames_for_duplicates (called before upload) is what stops the
    same photo from being added twice as separate database entries.

    Returns (success, failure_reason) so the caller can log/count failures
    for the ingestion_jobs history record -- a decode failure or a photo
    with no detected face counts as a failure here (unlike the OneDrive
    path, both are silent here only in the sense that they don't raise).
    """
    save_path = IMAGES_DIR / Path(original_name).name

    decoded_ok, faces = await asyncio.to_thread(_decode_save_and_detect, contents, save_path)
    job.processed_files += 1

    if not decoded_ok:
        return False, f"Could not decode photo '{original_name}'"
    if len(faces) == 0:
        return False, f"No face detected in '{original_name}'"

    blob_path = str(save_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    image_id = str(uuid.uuid4())

    pool = get_pool()
    async with pool.acquire() as conn:
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
                    job.faces_matched += 1

            b = face["bbox"]
            await conn.execute(
                """
                INSERT INTO images
                    (image_id, blob_path, original_filename, embedding, matched_person_id,
                     confidence, bbox_x, bbox_y, bbox_w, bbox_h, event_name)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11)
                """,
                image_id,
                blob_path,
                original_name,
                embedding_str,
                matched_person_id,
                similarity,
                b["x"], b["y"], b["width"], b["height"],
                event_name,
            )
            job.faces_detected += 1

    job.photos_added += 1
    return True, None


async def _run_bulk_upload_job(
    job: UploadJob, files: list[tuple[str, bytes]], event_name: str | None, ingestion_job_id: str
) -> None:
    """
    Background task: processes every photo in the batch (up to
    BULK_UPLOAD_CONCURRENCY at once), updating `job` (the in-memory tracker
    the panel polls live) as it goes. Also mirrors progress into the
    DB-backed `ingestion_jobs` row (ingestion_job_id) so this manual upload
    shows up in the admin "View Upload History" list alongside OneDrive
    folder jobs, not just while this page is open.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    pool = get_pool()
    await ingestion_jobs.mark_processing(pool, ingestion_job_id, len(files))
    progress = ingestion_jobs.BatchedProgressWriter(pool, ingestion_job_id)
    semaphore = asyncio.Semaphore(BULK_UPLOAD_CONCURRENCY)

    async def _bounded(original_name: str, contents: bytes) -> None:
        async with semaphore:
            success, reason = await _process_one_upload_photo(job, original_name, contents, event_name)
            if success:
                await progress.record_success()
            else:
                await ingestion_jobs.log_failure(pool, ingestion_job_id, original_name, reason or "Unknown error")
                await progress.record_failure()

    try:
        await asyncio.gather(*(_bounded(name, contents) for name, contents in files))
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)
    finally:
        job.finished_at = time.time()
        await progress.flush()
        if job.status == "done":
            await ingestion_jobs.mark_completed(pool, ingestion_job_id)
        else:
            await ingestion_jobs.mark_failed_outright(pool, ingestion_job_id, job.error or "Upload failed")


@router.post("/admin/test_images")
async def add_test_images(
    files: list[UploadFile] = File(...),
    event_name: str | None = Form(default=None),
):
    """
    Starts adding one or more new test/bulk photos incrementally (does NOT
    clear existing data). Detects faces, matches each against the persons
    table, and inserts into the images table. Assumes duplicate filenames
    have already been filtered out client-side via
    check_filenames_for_duplicates.

    event_name (optional) tags every photo in this batch, so the search
    page can later filter results down to a single event.

    Processing happens in a background task so this returns immediately --
    poll GET /admin/test_images/jobs/{job_id} for progress and the final
    summary (photos_added/faces_detected/faces_matched).
    """
    file_bytes = [(upload.filename or "photo.jpg", await upload.read()) for upload in files]
    clean_event_name = event_name.strip() if event_name and event_name.strip() else None

    job = create_job(total_files=len(file_bytes))

    # Also record a DB-backed ingestion_jobs row so this manual upload shows
    # up in the admin "View Upload History" list, not just in the live
    # in-page progress text (which is backed by the separate in-memory `job`
    # above and disappears once this page is closed/reloaded).
    pool = get_pool()
    folder_label = f"Manual upload ({len(file_bytes)} file(s))"
    if clean_event_name:
        folder_label += f" -- {clean_event_name}"
    ingestion_job_id = await ingestion_jobs.create_job(pool, "test_images", folder_label)

    asyncio.create_task(_run_bulk_upload_job(job, file_bytes, clean_event_name, ingestion_job_id))

    return {"job_id": job.job_id, "total_files": job.total_files}


@router.get("/admin/test_images/jobs/{job_id}")
async def get_test_images_job(job_id: str):
    """Poll the progress and, once finished, the result of a bulk test-image upload job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No upload job found with id '{job_id}'")
    return job.to_dict()


async def _process_one_onedrive_test_image(
    job_id: str,
    progress: "ingestion_jobs.BatchedProgressWriter",
    name: str,
    download_url: str,
    event_name: str | None,
) -> None:
    """
    Downloads, saves, detects, matches, and inserts one OneDrive-sourced
    test photo. Unlike _process_one_upload_photo (the direct-upload path),
    a decode failure or a photo with no detected face IS treated as a
    failure here -- logged to ingestion_job_failures -- since a folder pull
    has no admin watching each photo in real time to notice a silent skip.
    """
    pool = get_pool()
    save_path = IMAGES_DIR / name

    try:
        contents = await asyncio.to_thread(download_image, download_url)
        decoded_ok, faces = await asyncio.to_thread(_decode_save_and_detect, contents, save_path)
        if not decoded_ok:
            raise ValueError(f"Could not decode image '{name}'")
        if len(faces) == 0:
            raise ValueError(f"No face detected in '{name}'")
    except Exception as e:
        await ingestion_jobs.log_failure(pool, job_id, name, str(e))
        await progress.record_failure()
        return

    blob_path = str(save_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    image_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
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

            b = face["bbox"]
            await conn.execute(
                """
                INSERT INTO images
                    (image_id, blob_path, original_filename, embedding, matched_person_id,
                     confidence, bbox_x, bbox_y, bbox_w, bbox_h, event_name)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10, $11)
                """,
                image_id,
                blob_path,
                name,
                embedding_str,
                matched_person_id,
                similarity,
                b["x"], b["y"], b["width"], b["height"],
                event_name,
            )

    await progress.record_success()


async def _run_test_images_onedrive_job(job_id: str, folder_url: str, event_name: str | None) -> None:
    """Background job: lists the OneDrive folder, skips filenames already in
    the library (same intent as check_filenames_for_duplicates, just
    automatic since there's no admin choosing which dupes to keep here),
    then processes the rest with the same bounded-concurrency pattern as
    the direct-upload job."""
    pool = get_pool()
    try:
        images = await asyncio.to_thread(list_folder_images, folder_url)
    except Exception as e:
        await ingestion_jobs.mark_failed_outright(pool, job_id, str(e))
        return

    async with pool.acquire() as conn:
        existing = await conn.fetch(
            "SELECT DISTINCT original_filename FROM images WHERE original_filename = ANY($1::text[])",
            [item["name"] for item in images],
        )
    already_ingested = {row["original_filename"] for row in existing}
    images = [item for item in images if item["name"] not in already_ingested]

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    await ingestion_jobs.mark_processing(pool, job_id, len(images))
    progress = ingestion_jobs.BatchedProgressWriter(pool, job_id)
    semaphore = asyncio.Semaphore(ONEDRIVE_INGEST_CONCURRENCY)

    async def _bounded(item: dict) -> None:
        async with semaphore:
            await _process_one_onedrive_test_image(
                job_id, progress, item["name"], item["download_url"], event_name
            )

    await asyncio.gather(*(_bounded(item) for item in images))
    await progress.flush()
    await ingestion_jobs.mark_completed(pool, job_id)


@router.post("/admin/test_images/onedrive")
async def add_test_images_from_onedrive(payload: OneDriveTestImagesRequest = Body(...)):
    """
    Bulk-imports test images from a OneDrive folder link -- every image
    directly inside it (no recursion into subfolders), matched against the
    persons table exactly like a direct bulk upload. Runs as a background
    job (folders can be hundreds of images) with DB-persisted progress;
    poll GET /admin/ingestion_jobs/{job_id} for progress -- no
    auto-polling, the admin UI updates on a manual click.
    """
    pool = get_pool()
    clean_event_name = payload.event_name.strip() if payload.event_name and payload.event_name.strip() else None
    job_id = await ingestion_jobs.create_job(pool, "test_images", payload.folder_url)
    asyncio.create_task(_run_test_images_onedrive_job(job_id, payload.folder_url, clean_event_name))
    return {"job_id": job_id}


@router.get("/admin/ingestion_jobs")
async def list_ingestion_jobs(target: str = Query(...)):
    """
    History of past OneDrive folder-ingestion jobs for one target
    ("reference_persons" or "test_images"), newest first -- for the admin
    UI's "View Upload History" list, since the in-page job box only ever
    shows the most recently started job and loses that on reload.
    """
    if target not in ("reference_persons", "test_images"):
        raise HTTPException(status_code=400, detail="target must be 'reference_persons' or 'test_images'")
    pool = get_pool()
    jobs = await ingestion_jobs.list_jobs(pool, target)
    return {"jobs": jobs}


@router.get("/admin/ingestion_jobs/{job_id}")
async def get_ingestion_job(job_id: str):
    """
    Manual "Show me the update" poll for a OneDrive folder-ingestion job
    (test images or reference persons) -- no automatic polling loop.
    """
    pool = get_pool()
    job = await ingestion_jobs.get_job(pool, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No ingestion job found with id '{job_id}'")
    return job


@router.get("/admin/ingestion_jobs/{job_id}/failures")
async def get_ingestion_job_failures(job_id: str):
    """Per-file failures (filename + error) for a OneDrive folder-ingestion job."""
    pool = get_pool()
    job = await ingestion_jobs.get_job(pool, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No ingestion job found with id '{job_id}'")
    failures = await ingestion_jobs.get_job_failures(pool, job_id)
    return {"job_id": job_id, "failures": failures}


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
            SELECT image_id, blob_path, source_url, COUNT(*) AS face_count
            FROM images
            GROUP BY image_id, blob_path, source_url
            ORDER BY blob_path
            """
        )
    return {
        "images": [
            {
                "image_id": str(row["image_id"]),
                "blob_path": row["blob_path"],
                "source_url": row["source_url"],
                "image_url": row["source_url"] or blob_path_to_url(row["blob_path"]),
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
    if not blob_path:
        raise HTTPException(
            status_code=404,
            detail="Annotated view isn't available for remote-sourced images (no local file)",
        )
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
            "SELECT blob_path, original_filename FROM images WHERE image_id = $1 LIMIT 1", image_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"No test image found with id '{image_id}'")

        await conn.execute("DELETE FROM images WHERE image_id = $1", image_id)

    if row["blob_path"]:
        full_path = (PROJECT_ROOT / row["blob_path"]).resolve()
        if str(full_path).startswith(str(PROJECT_ROOT)) and full_path.exists():
            try:
                full_path.unlink()
            except OSError:
                pass

    label = row["original_filename"] or (Path(row["blob_path"]).name if row["blob_path"] else image_id)
    await _log_completed_action("test_images", f"Deleted: {label}", 1)
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
        if not row["blob_path"]:
            continue
        full_path = (PROJECT_ROOT / row["blob_path"]).resolve()
        if str(full_path).startswith(str(PROJECT_ROOT)) and full_path.exists():
            try:
                full_path.unlink()
            except OSError:
                pass

    await _log_completed_action("test_images", f"Deleted all ({len(rows)} photo(s))", len(rows))
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
