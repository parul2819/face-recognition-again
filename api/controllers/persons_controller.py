from fastapi import APIRouter

from api.db import get_pool

router = APIRouter()


@router.get("/persons")
async def list_persons():
    """Returns every known reference person, for the name-search dropdown."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT employee_id, name FROM persons ORDER BY employee_id"
        )
    return {
        "persons": [
            {"employee_id": row["employee_id"], "name": row["name"]} for row in rows
        ]
    }
