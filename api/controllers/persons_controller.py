from fastapi import APIRouter

from api.db import get_pool
from api.utils import blob_path_to_url

router = APIRouter()


@router.get("/persons")
async def list_persons():
    """
    Returns every known reference person, including a photo URL, for the
    name-search dropdown and the admin panel's person list.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT employee_id, name, reference_image_path FROM persons ORDER BY employee_id"
        )
    return {
        "persons": [
            {
                "employee_id": row["employee_id"],
                "name": row["name"],
                "photo_url": blob_path_to_url(row["reference_image_path"])
                if row["reference_image_path"]
                else None,
            }
            for row in rows
        ]
    }
