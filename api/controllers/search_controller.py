import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

# Add project root to sys.path so "from core.face_utils import ..." works
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.face_utils import get_face_embeddings_from_array

from api.auth import require_admin_session
from api.config import DEFAULT_PAGE_SIZE, PROJECT_ROOT, SEARCH_THRESHOLD
from api.db import get_pool
from api.utils import blob_path_to_url, embedding_to_pgvector

router = APIRouter()


async def run_embedding_search(
    conn,
    embedding_value,
    page: int,
    page_size: int,
    threshold: float,
    date_from: date | None = None,
    date_to: date | None = None,
    event_name: str | None = None,
):
    """
    Shared search logic: given an embedding, run a live cosine-similarity
    search against images.embedding, dedupe to the best face per photo, and
    paginate. Used by both /search (uploaded image) and /search_by_employee
    (reference image re-detected fresh from disk).

    threshold overrides the SEARCH_THRESHOLD .env default so the UI can
    adjust it per search. date_from/date_to filter on uploaded_at (both
    inclusive, compared by calendar day); event_name is an exact match.
    """
    offset = (page - 1) * page_size

    # $1 is always the query embedding; filters get appended before the
    # threshold/limit/offset params so their placeholder numbers stay in sync.
    params: list = [embedding_value]
    conditions = []
    if date_from is not None:
        params.append(date_from)
        conditions.append(f"uploaded_at::date >= ${len(params)}")
    if date_to is not None:
        params.append(date_to)
        conditions.append(f"uploaded_at::date <= ${len(params)}")
    if event_name:
        params.append(event_name)
        conditions.append(f"event_name = ${len(params)}")
    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    params.append(threshold)
    threshold_idx = len(params)
    params.append(page_size)
    limit_idx = len(params)
    params.append(offset)
    offset_idx = len(params)

    rows = await conn.fetch(
        f"""
        WITH ranked AS (
            SELECT
                image_id,
                blob_path,
                source_url,
                bbox_x, bbox_y, bbox_w, bbox_h,
                1 - (embedding <=> $1::vector) AS similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY image_id
                    ORDER BY embedding <=> $1::vector
                ) AS rn
            FROM images
            {where_sql}
        ),
        best_per_image AS (
            SELECT image_id, blob_path, source_url, bbox_x, bbox_y, bbox_w, bbox_h, similarity
            FROM ranked
            WHERE rn = 1 AND similarity >= ${threshold_idx}
        )
        SELECT image_id, blob_path, source_url, bbox_x, bbox_y, bbox_w, bbox_h, similarity,
               COUNT(*) OVER () AS total_count
        FROM best_per_image
        ORDER BY similarity DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
        """,
        *params,
    )

    total_count = rows[0]["total_count"] if rows else 0
    results = [
        {
            "image_id": str(row["image_id"]),
            "blob_path": row["blob_path"],
            "source_url": row["source_url"],
            # Remotely-sourced rows (source_url set, no local blob_path) use
            # source_url directly as the image reference; local rows keep
            # the existing /pics-relative URL.
            "image_url": row["source_url"] or blob_path_to_url(row["blob_path"]),
            "similarity": round(float(row["similarity"]), 3),
            "bbox": {
                "x": row["bbox_x"],
                "y": row["bbox_y"],
                "width": row["bbox_w"],
                "height": row["bbox_h"],
            },
        }
        for row in rows
    ]

    return total_count, results


def embedding_from_face_list(faces) -> str:
    """Pick the largest detected face (in case of multiple) and return its
    embedding in pgvector text format."""
    query_face = max(faces, key=lambda f: f["bbox"]["width"] * f["bbox"]["height"])
    return embedding_to_pgvector(query_face["embedding"])


@router.get("/search/filters", dependencies=[Depends(require_admin_session)])
async def get_search_filters():
    """
    Filter options for the search page: the default similarity threshold
    (from .env, used to preset the UI slider) and every distinct event name
    currently tagged on test images (for the event dropdown). Admin-only --
    the public page's camera search doesn't offer these filters.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT event_name FROM images WHERE event_name IS NOT NULL ORDER BY event_name"
        )
    return {
        "default_threshold": SEARCH_THRESHOLD,
        "event_names": [row["event_name"] for row in rows],
    }


@router.get("/search_by_event", dependencies=[Depends(require_admin_session)])
async def search_by_event(
    event_name: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    """
    Browse every photo tagged with a given event -- no uploaded/reference
    face involved, just a plain filter deduped to one row per photo
    (images has one row per detected face, so several rows can share an
    image_id).
    """
    offset = (page - 1) * page_size

    params: list = [event_name]
    conditions = ["event_name = $1"]
    if date_from is not None:
        params.append(date_from)
        conditions.append(f"uploaded_at::date >= ${len(params)}")
    if date_to is not None:
        params.append(date_to)
        conditions.append(f"uploaded_at::date <= ${len(params)}")
    where_sql = " AND ".join(conditions)

    params.append(page_size)
    limit_idx = len(params)
    params.append(offset)
    offset_idx = len(params)

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH photos AS (
                SELECT image_id, blob_path, MIN(uploaded_at) AS uploaded_at
                FROM images
                WHERE {where_sql}
                GROUP BY image_id, blob_path
            )
            SELECT image_id, blob_path, COUNT(*) OVER () AS total_count
            FROM photos
            ORDER BY uploaded_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
        )

    total_count = rows[0]["total_count"] if rows else 0
    results = [
        {
            "image_id": str(row["image_id"]),
            "blob_path": row["blob_path"],
            "image_url": blob_path_to_url(row["blob_path"]),
        }
        for row in rows
    ]

    return {
        "event_name": event_name,
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_more": (page - 1) * page_size + len(results) < total_count,
        "results": results,
    }


@router.post("/search")
async def search_by_image(
    # Public route -- the main page's camera-capture search depends on it
    # being unauthenticated. Every other search mode below requires admin
    # login.
    file: UploadFile = File(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    event_name: str | None = Query(default=None),
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    faces = get_face_embeddings_from_array(img)
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image")

    embedding_str = embedding_from_face_list(faces)
    effective_threshold = threshold if threshold is not None else SEARCH_THRESHOLD

    pool = get_pool()
    async with pool.acquire() as conn:
        total_count, results = await run_embedding_search(
            conn, embedding_str, page, page_size, effective_threshold, date_from, date_to, event_name
        )

    return {
        "query_faces_detected": len(faces),
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_more": (page - 1) * page_size + len(results) < total_count,
        "results": results,
    }


@router.get("/search_by_employee", dependencies=[Depends(require_admin_session)])
async def search_by_employee(
    employee_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    event_name: str | None = Query(default=None),
):
    """
    Name/person search: loads this person's reference photo from disk and
    runs face detection on it fresh (same as image upload search), instead
    of reusing the embedding stored in persons.reference_embedding. This
    matches the accuracy of live image-upload search per user testing.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        person = await conn.fetchrow(
            "SELECT name, reference_image_path FROM persons WHERE employee_id = $1",
            employee_id,
        )
        if person is None:
            raise HTTPException(status_code=404, detail=f"No person found with employee_id '{employee_id}'")

        if not person["reference_image_path"]:
            raise HTTPException(
                status_code=404,
                detail=f"No reference image on file for '{employee_id}'. Re-run ingest_reference_images.py.",
            )

        full_path = (PROJECT_ROOT / person["reference_image_path"]).resolve()
        if not str(full_path).startswith(str(PROJECT_ROOT)) or not full_path.exists():
            raise HTTPException(status_code=404, detail="Reference image file not found on disk")

        img = cv2.imread(str(full_path))
        if img is None:
            raise HTTPException(status_code=500, detail="Could not read reference image file")

        faces = get_face_embeddings_from_array(img)
        if len(faces) == 0:
            raise HTTPException(status_code=500, detail="No face detected in the reference image")

        embedding_str = embedding_from_face_list(faces)
        effective_threshold = threshold if threshold is not None else SEARCH_THRESHOLD

        total_count, results = await run_embedding_search(
            conn, embedding_str, page, page_size, effective_threshold, date_from, date_to, event_name
        )

    return {
        "employee_id": employee_id,
        "name": person["name"],
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_more": (page - 1) * page_size + len(results) < total_count,
        "results": results,
    }
