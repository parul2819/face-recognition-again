# Folder-based batch ingestion (background job)

## Objective

Given a folder URL (source TBD -- see "Open question" below), fetch every
image in that folder, generate embeddings, and store them in the vector
database -- for folders containing hundreds to ~1,000 images. This must
never block or time out the triggering API request, and must let an admin
check progress on demand.

## Requirements (confirmed in discussion)

1. Submitting a folder URL returns immediately (a `job_id`), while the
   actual fetch/embed/store work runs in the background.
2. Progress must survive the request completing -- persisted in the
   database, not just in-memory (unlike the existing `api/jobs.py`
   in-memory tracker used for manual bulk uploads).
3. No automatic UI polling. The admin page shows a static "processing..."
   indicator; a **"Show me the update"** button triggers a GET call and
   renders the latest counts on click, nothing more.
4. Batched progress updates, not per-image: the shared job-progress row is
   only written once every **20 successfully processed images** (plus
   once at job start and once at job end), to avoid heavy/contended writes
   during concurrent processing.
5. Failures are logged individually (filename + error), since failures are
   expected to be rare (>99.9% success rate) -- this table stays small.
   Successes are NOT logged individually, only counted.
6. Must not slow down the actual fetch/embed/insert work per image --
   reuse the existing async + `asyncio.to_thread` + bounded-concurrency
   pattern from `docs/async-bulk-uploads.md` (see "Reuse existing
   patterns" below).

## Open question -- must be resolved before starting step 3

**What is the folder source?** OneDrive was the original target but is
blocked pending IT admin consent (see `docs/public-image-poc.md`'s
"Notes" and the earlier conversation about Graph Explorer's "Need admin
approval" error). Do NOT block the rest of this feature on that -- steps
1, 2, 4, 5, 6 below are fully buildable and testable right now using a
stub/placeholder folder-listing function. Only step 3 needs the real
answer, and it's isolated to one function so swapping it in later is
cheap.

## Database schema

### Table 1 -- `ingestion_jobs` (header, low write frequency)

```sql
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    folder_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    total_images INT NOT NULL DEFAULT 0,
    processed_images INT NOT NULL DEFAULT 0,  -- successes only
    failed_images INT NOT NULL DEFAULT 0,
    error_message TEXT,  -- set if the job itself fails outright (e.g. can't list the folder)
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    completed_at TIMESTAMP
);
```

Write pattern for this table during a run:
- 1 write at job creation (`status='pending'`, then `'processing'` once
  the folder listing resolves and `total_images` is known)
- 1 write per **batch of 20 successful images** (increment
  `processed_images` by the batch size, update `updated_at`)
- Writes for failures increment `failed_images` -- can piggyback on the
  same batched cadence rather than writing per-failure (batch every N
  images *attempted*, not just every N successes, so a run with many
  failures still updates the header reasonably often)
- 1 write at job completion (`status='completed'` or `'failed'`,
  `completed_at`)

### Table 2 -- `ingestion_job_failures` (small, failures only)

```sql
CREATE TABLE IF NOT EXISTS ingestion_job_failures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ingestion_jobs(job_id),
    filename TEXT NOT NULL,
    error_message TEXT,
    failed_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingestion_job_failures_job_id_idx
    ON ingestion_job_failures (job_id);
```

One row per failed image, written immediately when that image fails (not
batched -- failures are rare, so this is cheap regardless).

### `images` table

No new columns needed here beyond what already exists (`source_url`,
nullable `blob_path`) from the public-image POC work. Optionally add a
`job_id UUID REFERENCES ingestion_jobs(job_id)` column if it's useful to
trace which ingestion run a photo came from -- not required for the core
feature, include only if trivial.

## Backend implementation

### 1. Folder-listing interface (pluggable, per "Open question" above)

```python
# core/folder_source.py (or similar)
def list_folder_images(folder_url: str) -> list[str]:
    """
    Returns a list of downloadable image URLs for everything in the given
    folder. Swap the implementation once the real source is confirmed;
    everything else in this feature depends only on this function's
    signature, not on how it works internally.
    """
```

For now, implement a stub that works with something testable (e.g. a
Wikimedia Commons *category* page listing multiple files, reusing the
`resolve_commons_file_url` pattern from `core/remote_image.py`) so the
rest of the pipeline can be built and tested without waiting on OneDrive
access. Note clearly in the code that this is a placeholder.

### 2. `POST /admin/folder_jobs`

- Body: `{ "folder_url": "..." }`
- Inserts an `ingestion_jobs` row (`status='pending'`)
- Schedules the background processing (`asyncio.create_task`, same
  pattern as `add_test_images` in `admin_controller.py`)
- Returns `{ "job_id": ... }` immediately

### 3. Background processing function

1. Call `list_folder_images(folder_url)` → get the URL list. Update the
   job row: `total_images = len(urls)`, `status='processing'`.
2. Process the list with bounded concurrency (reuse
   `BULK_UPLOAD_CONCURRENCY` / the semaphore pattern already in
   `admin_controller.py`). For each image, off the event loop via
   `asyncio.to_thread`:
   - Download (reuse `core/remote_image.py`'s `download_image`)
   - Decode → `get_face_embeddings_from_array_bulk` (the single-threaded
     model instance already built for concurrent bulk processing)
   - Match against `persons`, insert into `images` with `source_url` set
     (same logic as `_process_one_upload_photo`, adapted for a URL
     instead of uploaded bytes)
3. Maintain an in-process counter per worker; every time the **running
   total of successes across all workers** crosses a multiple of 20 (use
   a shared counter + lock, or a single coordinating counter, to avoid
   double-counting across concurrent workers), issue one
   `UPDATE ingestion_jobs SET processed_images = ... WHERE job_id = ...`.
4. On any per-image failure, insert one row into
   `ingestion_job_failures` immediately (filename + error message).
5. At the end: final `UPDATE ingestion_jobs SET status='completed', completed_at=now()`
   (or `'failed'` with `error_message` if the folder listing itself
   failed in step 1).

### 4. `GET /admin/folder_jobs/{job_id}`

Returns the current `ingestion_jobs` row as-is (status, total_images,
processed_images, failed_images). This is the endpoint the UI's "Show me
the update" button calls -- no polling loop, a single fetch-on-click.

### 5. `GET /admin/folder_jobs/{job_id}/failures` (optional, if useful)

Returns the list of failed filenames + error messages from
`ingestion_job_failures` for this job, for anyone who wants to see which
specific files failed and why.

## Frontend (admin.html)

New panel (e.g. under the existing "Test Images" sidebar group, or its
own group):
- Folder URL input + "Start" button → calls `POST /admin/folder_jobs`,
  stores the returned `job_id`, shows a static "Processing in the
  background..." message (no auto-refresh).
- A **"Show me the update"** button → calls
  `GET /admin/folder_jobs/{job_id}` on click, renders:
  - A progress bar (`processed_images + failed_images` out of
    `total_images`)
  - Counts: total / processed / failed
  - Status text (pending / processing / completed / failed)
- If useful, a small "View failures" link/section that calls the
  failures endpoint and lists filenames + error messages.

## Reuse existing patterns (don't duplicate)

- Bounded concurrency + `asyncio.to_thread` for CPU-bound work: see
  `BULK_UPLOAD_CONCURRENCY`, `_process_one_upload_photo`,
  `_run_bulk_upload_job` in `api/controllers/admin_controller.py`, and
  `docs/async-bulk-uploads.md` for why this shape exists.
- Single-threaded face model instance for concurrent processing:
  `get_face_embeddings_from_array_bulk` in `core/face_utils.py`.
- Remote image download: `core/remote_image.py`'s `download_image`.
- Matching-against-persons + inserting into `images` with `source_url`:
  same shape as the existing bulk-upload insert logic, just fed a
  downloaded URL's bytes instead of an uploaded file's bytes.

## Explicitly out of scope

- Resuming a job after a server restart mid-run (job would be left in
  `status='processing'` with no worker actually running -- acceptable
  for now, flag as a known limitation, do not build recovery logic).
- Automatic UI polling (explicitly rejected per requirement 3).
- The real OneDrive folder-listing implementation (blocked on IT; use the
  stub per "Open question").
