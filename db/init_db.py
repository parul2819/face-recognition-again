import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "face_recognition")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "face_recognition")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# schema.sql lives in the same folder as this script, so this resolves
# correctly no matter which directory you run the command from.
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


async def main():
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
    )

    schema_sql = SCHEMA_PATH.read_text()
    await conn.execute(schema_sql)

    rows = await conn.fetch(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )

    print("Tables in database:")
    for row in rows:
        print(f"  - {row['table_name']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
