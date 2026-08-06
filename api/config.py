import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "face_recognition")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "face_recognition")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

SEARCH_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", os.getenv("MATCH_THRESHOLD", "0.35")))
DEFAULT_PAGE_SIZE = int(os.getenv("SEARCH_PAGE_SIZE", "14"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICS_DIR = PROJECT_ROOT / "pics"
UI_DIR = PROJECT_ROOT / "ui"

REFERENCE_IMAGES_DIR = PROJECT_ROOT / os.getenv("REFERENCE_IMAGES_DIR", "pics/reference pics")
IMAGES_DIR = PROJECT_ROOT / os.getenv("IMAGES_DIR", "pics")
