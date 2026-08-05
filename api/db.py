import asyncpg

from api.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """Called once on app startup."""
    global _pool
    _pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )


async def close_pool() -> None:
    """Called once on app shutdown."""
    if _pool is not None:
        await _pool.close()


def get_pool() -> asyncpg.Pool:
    """Controllers call this to get the shared connection pool."""
    if _pool is None:
        raise RuntimeError("Database pool has not been initialized yet")
    return _pool
