# Async bulk uploads for /admin/test_images

## Problem

`POST /admin/test_images` processed every uploaded photo synchronously inside
the request handler: decode, save to disk, run face detection, insert DB
rows, all before responding. Two issues:

- A large batch keeps the HTTP request open until every photo finishes,
  risking client/proxy timeouts.
- Face detection (`insightface`'s `FaceAnalysis.get()`) is CPU-bound and was
  running directly on the asyncio event loop, so it blocked the *entire*
  server (including unrelated requests) while a batch was processing.

## Fix

The app is a single FastAPI process with no task queue (no Celery/Redis) —
so the fix stays in-process:

1. **In-memory job tracker** (`api/jobs.py`) — a dict of
   `job_id -> UploadJob`, tracking status/progress/results. Old finished
   jobs are pruned after an hour so the dict doesn't grow forever. State is
   memory-only: it resets if the server restarts.

2. **Endpoint split**:
   - `POST /admin/test_images` — reads all uploaded file bytes right away
     (fast), creates a job, schedules the processing as a background
     `asyncio` task, and returns `{job_id, total_files}` immediately instead
     of waiting for processing to finish.
   - `GET /admin/test_images/jobs/{job_id}` — polling endpoint returning
     current progress (`processed_files`/`total_files`) and, once
     `status == "done"`, the same summary fields the old synchronous
     endpoint used to return directly (`photos_added`, `faces_detected`,
     `faces_matched`).

3. **Actually unblocking the event loop**: per photo, the CPU-bound
   `cv2.imdecode`/`cv2.imwrite` and `insightface` detection calls are pushed
   off the event loop with `asyncio.to_thread(...)`. This is what lets the
   progress-polling endpoint (and everything else) stay responsive while a
   batch runs, not just moving the loop to a background task.

4. **Processing several photos at once** (`BULK_UPLOAD_CONCURRENCY` in
   `admin_controller.py`, currently `min(4, cpu_count)`): the first version
   of this fix still processed one photo at a time, which just made a slow
   batch non-blocking rather than actually faster. Testing showed each
   photo's face detection takes ~5s on CPU. Fixed by:
   - Running up to `BULK_UPLOAD_CONCURRENCY` photos through
     `_process_one_upload_photo` concurrently (`asyncio.gather` + a
     semaphore), each acquiring its own DB connection from the pool
     (`asyncpg` connections aren't safe to share across concurrent
     coroutines).
   - `core/face_utils.py` now builds a **second** `FaceAnalysis` instance
     (`_bulk_app`) whose onnxruntime sessions are each capped to 1 internal
     thread (`intra_op_num_threads = 1`). Without this, a single image's
     inference tries to use every CPU core by default, so running several
     images "concurrently" would just have them fight over the same cores
     instead of overlapping. The default `_app` (used by single-image
     endpoints like `/admin/persons`) is left uncapped, since using every
     core for a single image is still the right call there.
   - Measured result on a 14-core dev machine: a 20-photo batch went from
     ~103s (sequential, ~5s/photo) to ~11s (~0.5s/photo effective) with
     concurrency 4 -- roughly a 9-10x speedup.

5. **Frontend** (`ui/admin.html`, `uploadTestImages()`): starts the job,
   polls `GET /admin/test_images/jobs/{job_id}` every second, and updates
   the status line with live progress (`Processing 12 / 50 photos...`)
   before showing the final summary once `status == "done"`.

## Known limitations

- Job state is in-memory only — a server restart mid-batch loses progress
  tracking (acceptable for this local admin tool).
- If the admin page is refreshed while a job is running, the UI loses track
  of the `job_id` and stops polling (the job keeps running server-side
  regardless).
