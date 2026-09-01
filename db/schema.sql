CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS persons (
    person_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id TEXT NOT NULL,
    name TEXT,
    reference_embedding vector(512),
    reference_image_path TEXT
);

-- In case this table already existed from before this column was added.
ALTER TABLE persons ADD COLUMN IF NOT EXISTS reference_image_path TEXT;

CREATE TABLE IF NOT EXISTS images (
    face_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL,
    blob_path TEXT,
    original_filename TEXT,
    embedding vector(512) NOT NULL,
    matched_person_id UUID REFERENCES persons(person_id),
    confidence FLOAT,
    bbox_x INT,
    bbox_y INT,
    bbox_w INT,
    bbox_h INT,
    uploaded_at TIMESTAMP DEFAULT now()
);

-- In case this table already existed from before this column was added.
ALTER TABLE images ADD COLUMN IF NOT EXISTS original_filename TEXT;

-- Tags a photo with the event it was uploaded for (e.g. "Annual Day 2026"),
-- set optionally at bulk upload time. Used by the search-page event filter.
ALTER TABLE images ADD COLUMN IF NOT EXISTS event_name TEXT;

-- In case this table already existed from before blob_path allowed NULL.
-- Rows sourced from a remote URL (see source_url below) have no local file,
-- so blob_path can no longer be guaranteed NOT NULL.
ALTER TABLE images ALTER COLUMN blob_path DROP NOT NULL;

-- Public URL of a remotely-sourced image (e.g. a Wikimedia Commons file),
-- used instead of blob_path when the image isn't stored locally. Nullable
-- and independent of blob_path -- exactly one of the two should be set.
ALTER TABLE images ADD COLUMN IF NOT EXISTS source_url TEXT;

-- employee_id is the true unique identifier (names can repeat across people).
-- ON CONFLICT (employee_id) in ingest_reference_images.py relies on this index.
CREATE UNIQUE INDEX IF NOT EXISTS persons_employee_id_unique_idx
    ON persons (employee_id);

CREATE INDEX IF NOT EXISTS persons_reference_embedding_hnsw_idx
    ON persons USING hnsw (reference_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS images_embedding_hnsw_idx
    ON images USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS images_image_id_idx
    ON images USING btree (image_id);

-- For fast filename-based duplicate checking during upload.
CREATE INDEX IF NOT EXISTS images_original_filename_idx
    ON images (original_filename);

-- For the search-page date-range and event-name filters.
CREATE INDEX IF NOT EXISTS images_uploaded_at_idx
    ON images (uploaded_at);

CREATE INDEX IF NOT EXISTS images_event_name_idx
    ON images (event_name);

-- OneDrive folder ingestion jobs (see docs/folder-batch-ingestion.md).
-- One row per "pull this folder" run, for either test images or reference
-- persons. Progress is written in batches (every ONEDRIVE_PROGRESS_BATCH_SIZE
-- successes), not per-image, to avoid contended writes during a large
-- concurrent run. DB-backed (unlike api/jobs.py's in-memory tracker used for
-- manual bulk uploads) so progress survives a slow multi-hundred-image run.
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target TEXT NOT NULL CHECK (target IN ('test_images', 'reference_persons')),
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

-- The OneDrive folder's own display name (resolved best-effort once the
-- background job starts), so "View Upload History" can show something
-- readable instead of the raw share URL. Null if resolution failed or
-- hasn't happened yet -- the UI falls back to folder_url in that case.
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS folder_name TEXT;

-- Images skipped up front because they were already in the library (by
-- filename) before per-image processing started -- distinct from
-- failed_images, which counts images that were attempted and errored.
-- Lets the history table say "N duplicates skipped" instead of a bare
-- "completed" that gives no explanation for 0 added/0 failed.
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS skipped_images INT NOT NULL DEFAULT 0;

-- One row per failed image, written immediately (not batched -- failures
-- are expected to be rare, so this stays small).
CREATE TABLE IF NOT EXISTS ingestion_job_failures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    error_message TEXT,
    failed_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingestion_job_failures_job_id_idx
    ON ingestion_job_failures (job_id);
