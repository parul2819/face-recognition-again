import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

# Add project root to sys.path so "from core.face_utils import ..." works
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.face_utils import get_face_embeddings_from_array

from api.config import DEFAULT_PAGE_SIZE, PROJECT_ROOT, SEARCH_THRESHOLD
from api.db import get_pool
from api.utils import blob_path_to_url, embedding_to_pgvector

router = APIRouter()


async def run_embedding_search(conn, embedding_value, page: int, page_size: int):
    """
    Shared search logic: given an embedding, run a live cosine-similarity
    search against images.embedding, dedupe to the best face per photo, and
    paginate. Used by both /search (uploaded image) and /search_by_employee
    (reference image re-detected fresh from disk).
    """
    offset = (page - 1) * page_size

    rows = await conn.fetch(
        """
        WITH ranked AS (
            SELECT
                image_id,
                blob_path,
                bbox_x, bbox_y, bbox_w, bbox_h,
                1 - (embedding <=> $1::vector) AS similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY image_id
                    ORDER BY embedding <=> $1::vector
                ) AS rn
            FROM images
        ),
        best_per_image AS (
            SELECT image_id, blob_path, bbox_x, bbox_y, bbox_w, bbox_h, similarity
            FROM ranked
            WHERE rn = 1 AND similarity >= $2
        )
        SELECT image_id, blob_path, bbox_x, bbox_y, bbox_w, bbox_h, similarity,
               COUNT(*) OVER () AS total_count
        FROM best_per_image
        ORDER BY similarity DESC
        LIMIT $3 OFFSET $4
        """,
        embedding_value,
        SEARCH_THRESHOLD,
        page_size,
        offset,
    )

    total_count = rows[0]["total_count"] if rows else 0
    results = [
        {
            "image_id": str(row["image_id"]),
            "blob_path": row["blob_path"],
            "image_url": blob_path_to_url(row["blob_path"]),
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


@router.post("/search")
async def search_by_image(
    file: UploadFile = File(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
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

    pool = get_pool()
    async with pool.acquire() as conn:
        total_count, results = await run_embedding_search(conn, embedding_str, page, page_size)

    return {
        "query_faces_detected": len(faces),
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_more": (page - 1) * page_size + len(results) < total_count,
        "results": results,
    }


@router.get("/search_by_employee")
async def search_by_employee(
    employee_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=100),
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

        total_count, results = await run_embedding_search(conn, embedding_str, page, page_size)

    return {
        "employee_id": employee_id,
        "name": person["name"],
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_more": (page - 1) * page_size + len(results) < total_count,
        "results": results,
    }
