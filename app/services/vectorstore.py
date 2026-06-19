"""
Shared ChromaDB vectorstore factory.

Both IngestionService and RetrievalService need an identical
GoogleGenerativeAIEmbeddings + Chroma instance. Centralising it here:
  - avoids double-loading the embedding model
  - makes swapping to a different vector DB (Pinecone, Weaviate…) a single edit
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Return the shared Chroma vectorstore (singleton, cached for lifetime of process).
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )
