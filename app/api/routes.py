"""
API route definitions for paper-rag.

All business logic lives in the service layer; routes only handle
HTTP concerns (validation, file I/O, error mapping).
"""

import os
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
        version="1.0.0"
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
    paper_title: str = Form(..., min_length=1, description="Human-readable title for this paper"),
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

    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
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
        logger.exception("Ingestion failed for '%s'", paper_title)
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

    # Tally chunks per paper_title
    counts: dict[str, int] = {}
    for meta in metadatas:
        title = str(meta.get("paper_title") or "Unknown") if meta else "Unknown"
        counts[title] = counts.get(title, 0) + 1

    papers = [
        PaperInfo(title=title, chunk_count=count)
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
    Remove all chunks for a givne paper from ChromaDB (by paper_title metadata).
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
