"""
Pydantic request/response schemas for the paper-rag API.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural language question to ask about the uploaded papers."
    )
    paper_filter: str | None = Field(
        default=None,
        description="Restrict the search to a single paper by its title."
    )


class SourceDocument(BaseModel):
    paper_title: str
    page: int | None          # may be absent on some chunks
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

class PaperInfo(BaseModel):
    title: str
    chunk_count: int


class PaperListResponse(BaseModel):
    papers: list[PaperInfo]
    total_chunks: int


class UploadResponse(BaseModel):
    paper_title: str
    file_path: str
    pages: int
    chunks: int
    ingested_at: str


class DeleteResponse(BaseModel):
    deleted: bool
    paper_title: str
    chunks_removed: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
