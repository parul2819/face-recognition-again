import asyncio
import os
import sys
import uuid
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv

# Add project root to sys.path so "from core.face_utils import ..." works
# regardless of which folder this script is run from.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.face_utils import get_face_embeddings

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "face_recognition")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "face_recognition")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

IMAGES_DIR = os.getenv("IMAGES_DIR", "pics")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    """Convert a numpy embedding to the string format pgvector expects: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem


async def find_best_match(conn, embedding_str: str):
    row = await conn.fetchrow(
        """
        SELECT person_id, name,
               1 - (reference_embedding <=> $1::vector) AS similarity
        FROM persons
        ORDER BY reference_embedding <=> $1::vector
        LIMIT 1
        """,
        embedding_str,
    )
    return row


async def main():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )

    # TEMPORARY (for testing/calibration): wipe the images table clean at the
    # start of every run, so each run starts from a blank slate. Remove this
    # once we move past threshold-tuning and want incremental ingestion.
    await conn.execute("TRUNCATE TABLE images")
    print("images table cleared.\n")

    images_dir = Path(IMAGES_DIR)
    image_files = sorted(
        (f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS),
        key=sort_key,
    )

    total_images = 0
    total_faces = 0
    total_matched = 0
    total_unmatched = 0

    for image_path in image_files:
        blob_path = str(image_path).replace("\\", "/")

        faces = get_face_embeddings(str(image_path))

        if len(faces) == 0:
            print(f"{image_path.name}: no faces found, skipping")
            continue

        image_id = str(uuid.uuid4())

        print(f"{image_path.name}: {len(faces)} face(s)")

        for face in faces:
            embedding_str = embedding_to_pgvector(face["embedding"])
            match = await find_best_match(conn, embedding_str)

            matched_person_id = None
            similarity = None

            if match is not None:
                similarity = float(match["similarity"])
                print(f"    closest: {match['name']} (similarity={similarity:.3f})")

                if similarity >= MATCH_THRESHOLD:
                    matched_person_id = match["person_id"]
                    total_matched += 1
                else:
                    total_unmatched += 1

            bbox = face["bbox"]
            await conn.execute(
                """
                INSERT INTO images
                    (image_id, blob_path, embedding, matched_person_id, confidence,
                     bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9)
                """,
                image_id,
                blob_path,
                embedding_str,
                matched_person_id,
                similarity,
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"],
            )
            total_faces += 1

        total_images += 1

    print(
        f"\nDone. {total_images} images processed, {total_faces} faces detected, "
        f"{total_matched} matched, {total_unmatched} below threshold ({MATCH_THRESHOLD})."
    )

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
