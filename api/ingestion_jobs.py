"""
DB-backed job tracking for OneDrive folder ingestion (see
docs/folder-batch-ingestion.md). Distinct from api/jobs.py's in-memory
tracker -- that one is fine for a manual bulk upload the admin is actively
watching, but a folder pull can be hundreds of images and needs progress
that survives longer than the in-memory dict's process lifetime.

No auto-polling: the admin UI calls get_job() on click ("Show me the
update"), not on a timer.
"""

from typing import Literal

import asyncpg

from api.config import ONEDRIVE_PROGRESS_BATCH_SIZE

JobTarget = Literal["test_images", "reference_persons"]


async def create_job(pool: asyncpg.Pool, target: JobTarget, folder_url: str) -> str:
    async with pool.acquire() as conn:
        job_id = await conn.fetchval(
            """
            INSERT INTO ingestion_jobs (target, folder_url, status)
            VALUES ($1, $2, 'pending')
            RETURNING job_id
            """,
            target,
            folder_url,
        )
    return str(job_id)


async def mark_processing(
    pool: asyncpg.Pool, job_id: str, total_images: int, skipped_images: int = 0
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'processing', total_images = $2, skipped_images = $3, updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            total_images,
            skipped_images,
        )


async def set_folder_name(pool: asyncpg.Pool, job_id: str, folder_name: str | None) -> None:
    """Best-effort: records the OneDrive folder's own display name once
    resolved, for the upload-history table. Called separately from
    create_job because the name is only known after an early Graph API call
    the background job makes -- a failure to resolve it just leaves the
    column null and the UI falls back to the share URL, it doesn't fail the
    job."""
    if not folder_name:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE ingestion_jobs SET folder_name = $2 WHERE job_id = $1",
            job_id,
            folder_name,
        )


async def mark_failed_outright(pool: asyncpg.Pool, job_id: str, error_message: str) -> None:
    """The job failed before per-image processing could start (e.g. the
    folder listing call itself failed)."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed', error_message = $2, completed_at = now(), updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            error_message,
        )


async def mark_completed(pool: asyncpg.Pool, job_id: str, note: str | None = None) -> None:
    """note reuses the error_message column for a non-error, informational
    remark on an otherwise-successful job -- e.g. "every image in this
    folder was already in the library." The admin UI shows it next to the
    status regardless of pass/fail, so it doubles as a heads-up here."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'completed', error_message = $2, completed_at = now(), updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            note,
        )


async def log_failure(pool: asyncpg.Pool, job_id: str, filename: str, error_message: str) -> None:
    """One row per failed image, written immediately -- failures are rare,
    so this is cheap regardless of ONEDRIVE_PROGRESS_BATCH_SIZE."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_job_failures (job_id, filename, error_message)
            VALUES ($1, $2, $3)
            """,
            job_id,
            filename,
            error_message,
        )


class BatchedProgressWriter:
    """
    Tracks per-image outcomes for one job and flushes processed/failed
    counters to the ingestion_jobs row every ONEDRIVE_PROGRESS_BATCH_SIZE
    images *attempted* (successes + failures combined), so a run with many
    failures still updates the header reasonably often. Not safe to share
    across processes -- only used within a single background task, where
    increments happen on the event loop thread between awaits.
    """

    def __init__(self, pool: asyncpg.Pool, job_id: str):
        self._pool = pool
        self._job_id = job_id
        self._processed_since_flush = 0
        self._failed_since_flush = 0

    async def record_success(self) -> None:
        self._processed_since_flush += 1
        await self._maybe_flush()

    async def record_failure(self) -> None:
        self._failed_since_flush += 1
        await self._maybe_flush()

    async def _maybe_flush(self) -> None:
        attempted = self._processed_since_flush + self._failed_since_flush
        if attempted >= ONEDRIVE_PROGRESS_BATCH_SIZE:
            await self.flush()

    async def flush(self) -> None:
        if self._processed_since_flush == 0 and self._failed_since_flush == 0:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs
                SET processed_images = processed_images + $2,
                    failed_images = failed_images + $3,
                    updated_at = now()
                WHERE job_id = $1
                """,
                self._job_id,
                self._processed_since_flush,
                self._failed_since_flush,
            )
        self._processed_since_flush = 0
        self._failed_since_flush = 0


async def list_jobs(pool: asyncpg.Pool, target: JobTarget, limit: int = 20) -> list[dict]:
    """Most recent jobs for one target (persons or test_images), newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, target, folder_url, folder_name, status, total_images,
                   processed_images, failed_images, skipped_images, error_message,
                   created_at, updated_at, completed_at
            FROM ingestion_jobs
            WHERE target = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            target,
            limit,
        )
    return [dict(row) for row in rows]


async def get_job(pool: asyncpg.Pool, job_id: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT job_id, target, folder_url, folder_name, status, total_images,
                   processed_images, failed_images, skipped_images, error_message,
                   created_at, updated_at, completed_at
            FROM ingestion_jobs
            WHERE job_id = $1
            """,
            job_id,
        )
    if row is None:
        return None
    return dict(row)


async def get_job_failures(pool: asyncpg.Pool, job_id: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT filename, error_message, failed_at
            FROM ingestion_job_failures
            WHERE job_id = $1
            ORDER BY failed_at
            """,
            job_id,
        )
    return [dict(row) for row in rows]
