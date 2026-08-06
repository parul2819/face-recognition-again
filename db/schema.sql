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
    blob_path TEXT NOT NULL,
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
