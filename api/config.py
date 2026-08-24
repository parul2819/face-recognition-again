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

# Azure AD app registration for OneDrive folder ingestion (device-code flow,
# no client secret -- see core/onedrive_auth.py and scripts/onedrive_auth_setup.py).
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID", "")
MS_TENANT_ID = os.getenv("MS_TENANT_ID", "")
ONEDRIVE_TOKEN_CACHE_PATH = PROJECT_ROOT / os.getenv("ONEDRIVE_TOKEN_CACHE_PATH", ".onedrive_token_cache.json")

# How many photos a folder-based OneDrive ingestion job processes at once.
# Same reasoning as BULK_UPLOAD_CONCURRENCY in admin_controller.py.
ONEDRIVE_INGEST_CONCURRENCY = int(os.getenv("ONEDRIVE_INGEST_CONCURRENCY", "4"))

# Batched-progress cadence for OneDrive folder ingestion jobs (see
# docs/folder-batch-ingestion.md) -- the ingestion_jobs row is only updated
# once every this many successfully processed images, to avoid contended
# writes during a large concurrent run.
ONEDRIVE_PROGRESS_BATCH_SIZE = 20

# Admin panel credentials (HTTP Basic Auth -- see api/auth.py). MUST be set
# in .env for any real deployment; these fallbacks exist only so local dev
# doesn't hard-fail if .env is missing them, and are intentionally weak.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
