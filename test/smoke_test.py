"""
Quick smoke test — verifies config loads and data dirs exist.

Usage:
    ../.venv/bin/python test_phase1.py
"""
import os
from app.config import settings

def test_config():
    assert settings.gemini_api_key, "GEMINI_API_KEY is not set in .env"
    assert settings.chroma_persist_dir
    assert settings.upload_dir
    print("✅ Config loaded successfully")
    print(f"   chroma_persist_dir : {settings.chroma_persist_dir}")
    print(f"   upload_dir         : {settings.upload_dir}")

def test_dirs():
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    assert os.path.isdir(settings.chroma_persist_dir)
    assert os.path.isdir(settings.upload_dir)
    print("✅ Data directories exist")

if __name__ == "__main__":
    test_config()
    test_dirs()
