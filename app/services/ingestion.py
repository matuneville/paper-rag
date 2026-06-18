"""
Ingestion pipeline — Phase 2.

Loads a PDF, splits it into chunks, embeds them with Gemini, and stores
everything in a local ChromaDB collection.
"""

import os
import logging
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.config import settings

logger = logging.getLogger(__name__)


class IngestionService:
    """Handles the full PDF > chunks > embeddings-> ChromaDB vector database pipeline."""

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
        )

    async def ingest_pdf(self, file_path: str, paper_title: str) -> dict:
        """Load > chunk > embed > store a PDF.

        Args:
            file_path:   Absolute or relative path to the PDF on disk.
            paper_title: Human-readable title stored as chunk metadata.

        Returns:
            A dict with ingestion statistics:
                {
                    "paper_title": str,
                    "file_path":   str,
                    "pages":       int,   # total pages in the PDF
                    "chunks":      int,   # number of chunks stored
                    "ingested_at": str,   # ISO-8601 UTC timestamp
                }
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info("Starting ingestion for '%s' (%s)", paper_title, file_path)

        # 1. Load PDF - one Doc per page
        pages = PyPDFLoader(file_path).load()
        logger.info("Loaded %d pages from '%s'", len(pages), paper_title)

        # 2. Split into chunks
        chunks = self._splitter.split_documents(pages)
        logger.info("Split into %d chunks", len(chunks))

        # 3. Enrich metadata on every chunk
        filename = os.path.basename(file_path)
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["paper_title"] = paper_title

        # 4. Embed + store in ChromaDB:
        #    It automatically embeds with Google and writes to disk on the vector db
        self._vectorstore.add_documents(chunks)
        logger.info("Stored %d chunks in ChromaDB for '%s'", len(chunks), paper_title)

        return {
            "paper_title": paper_title,
            "file_path": file_path,
            "pages": len(pages),
            "chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
