from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
API_DIR = PACKAGE_DIR.parent

LEGACY_DATA_DIR = API_DIR / "data"
PACKAGE_DATA_DIR = PACKAGE_DIR / "data"


def get_data_dir() -> Path:
    """Return the existing data directory, preferring the repo-level path."""
    if LEGACY_DATA_DIR.exists():
        return LEGACY_DATA_DIR
    return PACKAGE_DATA_DIR


DATA_DIR = get_data_dir()
PDF_PATH = DATA_DIR / "user_manual.pdf"
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.pkl"
