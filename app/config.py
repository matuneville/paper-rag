from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from .env file."""

    gemini_api_key: str
    chroma_persist_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Single instance shared across the app
settings = Settings()
