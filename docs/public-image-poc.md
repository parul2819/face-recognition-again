# Public-image POC (no auth needed) — validate before OneDrive

## Objective (per senior's direction)

Before pursuing OneDrive integration (blocked on IT approval for Graph API
access), validate the same end-to-end capability using **publicly
available images** that require no authentication at all. Same technical
question as the OneDrive POC: can the system fetch an image from a
**remote URL** (not local disk), generate its embedding, store it in
pgvector, and return the **correct source URL** as the search result
reference?

This supersedes `docs/onedrive-poc.md` for now — no OneDrive/Graph API
work is needed for this phase. Revisit that doc once IT grants access.

## Test images — Virat Kohli, from Wikimedia Commons

Do NOT hardcode a guessed `upload.wikimedia.org` URL (the path includes a
content hash that can't be predicted). Instead resolve it via Wikimedia's
own API, which is free, needs no key/auth, and returns the current direct
file URL:

```
GET https://commons.wikimedia.org/w/api.php
    ?action=query
    &titles=File:Virat_Kohli_portrait.jpg
    &prop=imageinfo
    &iiprop=url
    &format=json
```

The response's `query.pages.<id>.imageinfo[0].url` field is the direct,
downloadable image URL (a real `.jpg`, no login wall).

Candidate file titles to use for the 2-3 test images (all CC-licensed
portraits of Virat Kohli on Commons, confirmed to exist):
- `File:Virat_Kohli_portrait.jpg`
- `File:Virat_kohli.jpg`
- `File:Virat_Kohli_in_New_Delhi_in_December_2018.jpg`

(Swap in different files if a better angle/quality is needed — same API
call pattern works for any Commons file title.)

## Implementation tasks

### 1. URL → image bytes helper
A small function, e.g. in `core/remote_image.py`:

```
resolve_commons_file_url(file_title: str) -> str   # calls the API above
download_image(url: str) -> bytes                  # plain HTTP GET
```

No auth headers needed for either call.

### 2. Database — same as the OneDrive POC plan
Add a nullable `source_url TEXT` column to `images` (if not already added
from the earlier OneDrive POC work) to store the public image URL instead
of a local `blob_path`.

### 3. POC ingestion script
`scripts/public_image_poc_ingest.py`:
- For each of the 2-3 candidate file titles: resolve → download → decode
  → run through the **existing** `get_face_embeddings_from_array()` (do
  not duplicate this logic) → match against `persons` → insert into
  `images` with `source_url` set to the resolved direct URL.
- Print a summary per image (matched person, similarity).

### 4. Search path
Same as the OneDrive POC plan, step 5: when a result row has `source_url`
set, the search response should surface that URL as the image reference
instead of (or alongside) a local `image_url`.

### 5. Serving the image
Simplest for this POC: the frontend/response can just use `source_url`
directly as the `<img src>` — Wikimedia URLs are public and hotlink-
friendly, no proxy/redirect endpoint needed (unlike OneDrive, which
needed a token to download). Skip building a proxy endpoint for this
phase unless the annotated/box-drawing view is also required for these
remote images -- if it is, download-on-demand and draw the box the same
way the existing `/annotated` endpoint does for local files.

### 6. Validate end-to-end
- Ingest the 2-3 Wikimedia Virat images.
- Search using an existing local reference photo of Virat (already in
  `persons`).
- Confirm the results list includes the Wikimedia-sourced photos, each
  with `source_url` pointing to `upload.wikimedia.org` (not a local
  path), and that the image actually loads from that URL.

## Notes
- Wikimedia Commons images are CC-BY-SA licensed — fine for internal
  technical testing; note this if any output ever gets shared externally
  (attribution requirements apply for public reuse, not for internal
  validation).
- This same `source_url` pattern (nullable column, present alongside
  `blob_path`) is exactly what the OneDrive POC will reuse later — this
  phase is effectively derisking that part of the design without waiting
  on IT.
