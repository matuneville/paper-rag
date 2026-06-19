from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the .env file relative to this file's location (project root),
# so it works regardless of which directory the process is launched from.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """Central configuration loaded from .env file."""

    # --- Auth ---
    gemini_api_key: str

    # --- Storage ---
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    # --- Models ---
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash-lite"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    retrieval_k: int = 4
    excerpt_length: int = 300

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8")


# Single instance shared across the app
settings = Settings()
