from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LLM_PATH = BASE_DIR / "models" / "exaone_2.4b" / "EXAONE-3.5-2.4B-Instruct-Q5_K_M.gguf"
EMBEDDING_PATH = BASE_DIR / "models" / "bge-m3"
UPLOAD_DIR = BASE_DIR / "uploads"
TMP_UPLOAD_DIR = UPLOAD_DIR / ".tmp"
SWAP_DIR = UPLOAD_DIR / ".swap"
VERSION_DIR = UPLOAD_DIR / ".versions"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
RETRIEVE_K = 4

# Chroma cosine distance (0 = identical). Worse than this → treat as no relevant chunk.
DISTANCE_THRESHOLD = 0.72

NO_INFO_MESSAGE = "정보가 없어서 답변할 수 없습니다"

ALLOWED_EXTENSIONS = {".pdf"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

COLLECTION_NAME = "pdf_rag"
