import asyncio
import os
import sys
import uuid
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from dotenv import load_dotenv

# Add project root to sys.path so "from core..." imports work regardless of
# which folder this script is run from.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.face_utils import get_face_embeddings_from_array
from core.remote_image import download_image, resolve_commons_file_url

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "face_recognition")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "face_recognition")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

MATCH_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", os.getenv("MATCH_THRESHOLD", "0.35")))

# Candidate test images from docs/public-image-poc.md.
COMMONS_FILE_TITLES = [
    "File:Virat_Kohli_portrait.jpg",
    "File:Virat_kohli.jpg",
    "File:Virat_Kohli_in_New_Delhi_in_December_2018.jpg",
]


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    """Convert a numpy embedding to the string format pgvector expects: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


async def find_best_match(conn, embedding_str: str):
    return await conn.fetchrow(
        """
        SELECT person_id, name,
               1 - (reference_embedding <=> $1::vector) AS similarity
        FROM persons
        ORDER BY reference_embedding <=> $1::vector
        LIMIT 1
        """,
        embedding_str,
    )


async def main():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )

    for file_title in COMMONS_FILE_TITLES:
        print(f"{file_title}:")

        try:
            source_url = resolve_commons_file_url(file_title)
        except Exception as e:
            print(f"    could not resolve: {e}")
            continue

        try:
            image_bytes = download_image(source_url)
        except Exception as e:
            print(f"    could not download {source_url}: {e}")
            continue

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"    could not decode image from {source_url}")
            continue

        faces = get_face_embeddings_from_array(img)
        if len(faces) == 0:
            print(f"    no faces found, skipping ({source_url})")
            continue

        image_id = str(uuid.uuid4())
        print(f"    {len(faces)} face(s), source_url={source_url}")

        for face in faces:
            embedding_str = embedding_to_pgvector(face["embedding"])
            match = await find_best_match(conn, embedding_str)

            matched_person_id = None
            similarity = None

            if match is not None:
                similarity = float(match["similarity"])
                print(f"        closest: {match['name']} (similarity={similarity:.3f})")

                if similarity >= MATCH_THRESHOLD:
                    matched_person_id = match["person_id"]

            bbox = face["bbox"]
            await conn.execute(
                """
                INSERT INTO images
                    (image_id, source_url, embedding, matched_person_id, confidence,
                     bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9)
                """,
                image_id,
                source_url,
                embedding_str,
                matched_person_id,
                similarity,
                bbox["x"],
                bbox["y"],
                bbox["width"],
                bbox["height"],
            )

    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
