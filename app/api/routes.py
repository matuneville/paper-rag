"""
API route definitions for paper-rag.

All business logic lives in the service layer; routes only handle
HTTP concerns (validation, file I/O, error mapping).
"""

import asyncio
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService
from app.services.vectorstore import get_vectorstore
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    PaperInfo,
    PaperListResponse,
    SourceDocument,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level service instances (reuses the shared vectorstore singleton)
_ingestion = IngestionService()
_retrieval = RetrievalService()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"]
)
async def health_check() -> HealthResponse:
    """Liveness probe — always returns 200 if the server is up."""
    return HealthResponse(
        status="ok",
        version="1.1.0"
    )


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

@router.post(
    "/api/papers/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["papers"],
)
async def upload_paper(
    file: UploadFile = File(..., description="PDF file to ingest"),
    paper_title: str | None = Form(None, description="Optional human-readable title for this paper"),
) -> UploadResponse:
    """Accept a PDF upload, run the ingestion pipeline, and store chunks in ChromaDB."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are accepted.",
        )

    # Persist the uploaded file to the uploads directory
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename

    # Read file content async, then write to disk in a thread pool
    # (avoids blocking the event loop during large uploads)
    try:
        content = await file.read()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, dest_path.write_bytes, content)
    except OSError as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {exc}",
        ) from exc
    finally:
        await file.close()

    # Run ingestion pipeline
    try:
        stats = await _ingestion.ingest_pdf(str(dest_path), paper_title)
    except Exception as exc:
        logger.exception("Ingestion failed")
        if "already indexed" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return UploadResponse(**stats)


@router.get(
    "/api/papers",
    response_model=PaperListResponse,
    tags=["papers"]
)
async def list_papers() -> PaperListResponse:
    """Return all papers currently indexed in ChromaDB with their chunk counts."""
    vectorstore = get_vectorstore()
    collection = vectorstore._collection  # Chroma internal — stable across versions

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    # Tally chunks and capture metadata per paper_title
    counts: dict[str, int] = {}
    paper_meta: dict[str, dict] = {}
    
    for meta in metadatas:
        title = str(meta.get("paper_title") or "Unknown") if meta else "Unknown"
        counts[title] = counts.get(title, 0) + 1
        
        if meta and title not in paper_meta and title != "Unknown":
            authors_str = str(meta.get("authors", ""))
            authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
            year_str = str(meta.get("year", ""))
            year = int(year_str) if year_str and year_str.isdigit() else None
            abstract = str(meta.get("abstract", "")) or None
            
            paper_meta[title] = {
                "authors": authors,
                "year": year,
                "abstract": abstract
            }

    papers = [
        PaperInfo(
            title=title, 
            chunk_count=count,
            authors=paper_meta.get(title, {}).get("authors", []),
            year=paper_meta.get(title, {}).get("year"),
            abstract=paper_meta.get(title, {}).get("abstract")
        )
        for title, count
        in sorted(counts.items())
    ]
    
    return PaperListResponse(
        papers=papers,
        total_chunks=sum(counts.values())
    )


@router.delete(
    "/api/papers/{paper_title:path}",
    response_model=DeleteResponse,
    tags=["papers"],
)
async def delete_paper(paper_title: str) -> DeleteResponse:
    """
    Remove all chunks for a given paper from ChromaDB (by paper_title metadata).
    """
    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    # Find IDs of chunks belonging to this paper
    results = collection.get(where={"paper_title": paper_title}, include=["metadatas"])
    ids_to_delete = results.get("ids") or []

    if not ids_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No paper found with title '{paper_title}'.",
        )

    collection.delete(ids=ids_to_delete)
    logger.info("Deleted %d chunks for paper '%s'", len(ids_to_delete), paper_title)

    return DeleteResponse(
        deleted=True,
        paper_title=paper_title,
        chunks_removed=len(ids_to_delete),
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["chat"]
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Semantic search over indexed papers + Gemini-generated answer with citations.
    """
    try:
        result = await _retrieval.query_papers(
            question=request.question,
            paper_filter=request.paper_filter,
        )
    except Exception as exc:
        logger.exception("Retrieval failed for question: %r", request.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    sources = [
        SourceDocument(**src)
        for src
        in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources
    )
