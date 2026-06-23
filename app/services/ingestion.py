"""
Ingestion pipeline.

Loads a PDF, splits it into chunks, embeds them with Gemini, and stores
everything in a local ChromaDB collection.
"""

import os
import logging
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.services.vectorstore import get_vectorstore
from app.api.schemas import PaperMetadata

logger = logging.getLogger(__name__)


class IngestionService:
    """Handles the full PDF > chunks > embeddings > ChromaDB vector database pipeline."""

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._vectorstore = get_vectorstore()
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
        )

    async def ingest_pdf(self, file_path: str, paper_title: str | None = None) -> dict:
        """Load > chunk > embed > store a PDF.

        Args:
            file_path:   Absolute or relative path to the PDF on disk.
            paper_title: Optional human-readable title stored as chunk metadata.
                         If not provided, the LLM will extract it, falling back to filename.

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

        # Extract metadata using LLM
        filename = os.path.basename(file_path)
        fallback_title = paper_title or os.path.splitext(filename)[0]
        metadata_obj = PaperMetadata(title=fallback_title, authors=[], year=None, abstract=None)
        try:
            # Get text from first 2 pages, truncated to ~3000 chars
            first_pages_text = "\n".join([p.page_content for p in pages[:2]])[:3000]
            structured_llm = self._llm.with_structured_output(PaperMetadata)
            extracted = structured_llm.invoke(
                f"Extract metadata for the following research paper. "
                f"If a field like 'year' or 'abstract' is not clearly present, return null for it.\n\n"
                f"Paper Content:\n{first_pages_text}"
            )
            if extracted:
                if isinstance(extracted, dict):
                    metadata_obj = PaperMetadata(**extracted)
                elif isinstance(extracted, PaperMetadata):
                    metadata_obj = extracted
                if not metadata_obj.title:
                    metadata_obj.title = fallback_title
            logger.info("Extracted metadata for '%s'", metadata_obj.title)
        except Exception as exc:
            logger.warning("Metadata extraction failed for '%s': %s", fallback_title, exc)

        final_title = metadata_obj.title

        # Reject duplicate paper titles to prevent double-ingestion
        existing = self._vectorstore._collection.get(
            where={"paper_title": final_title}, include=[]
        )
        if existing.get("ids"):
            raise ValueError(f"'{final_title}' already indexed. Delete it first to re-ingest.")

        # 2. Split into chunks
        chunks = self._splitter.split_documents(pages)
        logger.info("Split into %d chunks", len(chunks))

        # 3. Enrich metadata on every chunk
        filename = os.path.basename(file_path)
        authors_str = ", ".join(metadata_obj.authors) if metadata_obj.authors else ""
        year_str = str(metadata_obj.year) if metadata_obj.year is not None else ""
        abstract_str = metadata_obj.abstract if metadata_obj.abstract else ""
        
        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["paper_title"] = final_title
            chunk.metadata["authors"] = authors_str
            chunk.metadata["year"] = year_str
            chunk.metadata["abstract"] = abstract_str

        # 4. Embed + store in ChromaDB:
        #    It automatically embeds with Google and writes to disk on the vector db
        self._vectorstore.add_documents(chunks)
        logger.info("Stored %d chunks in ChromaDB for '%s'", len(chunks), final_title)

        return {
            "paper_title": final_title,
            "file_path": file_path,
            "pages": len(pages),
            "chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata_obj,
        }
