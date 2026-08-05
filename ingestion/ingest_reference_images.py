import asyncio
import os
import sys
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

REFERENCE_IMAGES_DIR = os.getenv("REFERENCE_IMAGES_DIR", "pics/reference pics")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    """Convert a numpy embedding to the string format pgvector expects: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def parse_employee_id_and_name(filename_stem: str) -> tuple[str, str]:
    """
    Expected filename convention: "<employee_id>_<name>.jpg"
    e.g. "E1001_virat.jpg" -> employee_id="E1001", name="virat"
    """
    if "_" not in filename_stem:
        raise ValueError(
            f"Filename '{filename_stem}' does not follow the '<employee_id>_<name>' convention"
        )
    employee_id, name = filename_stem.split("_", 1)
    return employee_id, name


async def main():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )

    ref_dir = Path(REFERENCE_IMAGES_DIR)
    image_files = sorted(
        f for f in ref_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS
    )

    inserted = 0
    updated = 0
    skipped = 0

    for image_path in image_files:
        try:
            employee_id, person_name = parse_employee_id_and_name(image_path.stem)
        except ValueError as e:
            print(f"  WARNING: {e}, skipping")
            skipped += 1
            continue

        faces = get_face_embeddings(str(image_path))

        if len(faces) == 0:
            print(f"  WARNING: No face found in {image_path.name}, skipping")
            skipped += 1
            continue

        if len(faces) > 1:
            print(f"  WARNING: {len(faces)} faces found in {image_path.name}, skipping (expected exactly 1)")
            skipped += 1
            continue

        embedding = faces[0]["embedding"]
        embedding_str = embedding_to_pgvector(embedding)

        # Stored as a path relative to the project root (e.g.
        # "pics/reference pics/E1001_virat.jpg"), same convention as
        # images.blob_path, so the API can resolve it consistently.
        reference_image_path = str(image_path).replace("\\", "/")

        result = await conn.fetchrow(
            """
            INSERT INTO persons (employee_id, name, reference_embedding, reference_image_path)
            VALUES ($1, $2, $3::vector, $4)
            ON CONFLICT (employee_id)
            DO UPDATE SET name = EXCLUDED.name,
                          reference_embedding = EXCLUDED.reference_embedding,
                          reference_image_path = EXCLUDED.reference_image_path
            RETURNING (xmax = 0) AS inserted
            """,
            employee_id,
            person_name,
            embedding_str,
            reference_image_path,
        )

        if result["inserted"]:
            print(f"  Inserted (new): {employee_id} / {person_name} (confidence={faces[0]['confidence']:.3f})")
            inserted += 1
        else:
            print(f"  Updated (existing): {employee_id} / {person_name} (confidence={faces[0]['confidence']:.3f})")
            updated += 1

    print(f"\nDone. {inserted} new, {updated} updated, {skipped} skipped.")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
