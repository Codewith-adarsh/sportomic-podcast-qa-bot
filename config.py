import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# YouTube Settings
YOUTUBE_VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID", "Rni7Fz7208c")
YOUTUBE_VIDEO_URL_BASE = "https://youtube.com/watch?v="

# Gemini API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "podcast_transcript"

# Transcript JSON Path
TRANSCRIPT_JSON_PATH = DATA_DIR / "transcript.json"

# Chunking Hyperparameters
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50
RETRIEVAL_TOP_K = 4

# Log Configuration
LOG_FILE_PATH = BASE_DIR / "podcast_bot.log"
