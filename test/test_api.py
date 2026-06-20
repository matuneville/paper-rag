"""
API layer tests for paper-rag.

Uses FastAPI's TestClient (in-process, no real server) + unittest.mock to
isolate the HTTP layer from real Gemini calls or filesystem writes.

Run:
    .venv/bin/python test/test_api.py        # plain Python runner
    .venv/bin/pytest test/test_api.py -v     # if pytest is installed
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_vectorstore(metadatas=None, ids=None):
    """Return a MagicMock Chroma instance with a pre-configured collection."""
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "metadatas": metadatas if metadatas is not None else [],
        "ids": ids if ids is not None else [],
    }
    mock_vs = MagicMock()
    mock_vs._collection = mock_collection
    return mock_vs


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# GET /api/papers
# ---------------------------------------------------------------------------

def test_list_papers_empty():
    """Empty DB → empty list, zero chunks."""
    with patch("app.api.routes.get_vectorstore", return_value=_mock_vectorstore()):
        r = client.get("/api/papers")
    assert r.status_code == 200
    data = r.json()
    assert data["papers"] == []
    assert data["total_chunks"] == 0


def test_list_papers_with_data():
    """Multiple chunks for two papers → correct titles and counts."""
    metadatas = [
        {"paper_title": "Attention Is All You Need"},
        {"paper_title": "Attention Is All You Need"},
        {"paper_title": "BERT"},
    ]
    with patch("app.api.routes.get_vectorstore", return_value=_mock_vectorstore(metadatas=metadatas)):
        r = client.get("/api/papers")
    assert r.status_code == 200
    data = r.json()
    assert data["total_chunks"] == 3
    by_title = {p["title"]: p["chunk_count"] for p in data["papers"]}
    assert by_title["Attention Is All You Need"] == 2
    assert by_title["BERT"] == 1


# ---------------------------------------------------------------------------
# POST /api/papers/upload
# ---------------------------------------------------------------------------

def test_upload_rejects_non_pdf():
    """Non-PDF extension → 422 before hitting the service."""
    r = client.post(
        "/api/papers/upload",
        files={"file": ("report.docx", b"fake content", "application/octet-stream")},
        data={"paper_title": "Some Paper"},
    )
    assert r.status_code == 422


def test_upload_success():
    """Valid PDF → 201 with ingestion stats."""
    fake_stats = {
        "paper_title": "Test Paper",
        "file_path": "/data/uploads/test.pdf",
        "pages": 5,
        "chunks": 20,
        "ingested_at": "2026-01-01T00:00:00+00:00",
    }
    with (
        patch("app.api.routes._ingestion.ingest_pdf", new=AsyncMock(return_value=fake_stats)),
        patch("app.api.routes.shutil.copyfileobj"),  # skip real disk write
    ):
        r = client.post(
            "/api/papers/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"paper_title": "Test Paper"},
        )
    assert r.status_code == 201
    data = r.json()
    assert data["paper_title"] == "Test Paper"
    assert data["pages"] == 5
    assert data["chunks"] == 20


# ---------------------------------------------------------------------------
# DELETE /api/papers/{paper_title}
# ---------------------------------------------------------------------------

def test_delete_paper_not_found():
    """Unknown paper → 404."""
    with patch("app.api.routes.get_vectorstore", return_value=_mock_vectorstore(ids=[])):
        r = client.delete("/api/papers/nonexistent")
    assert r.status_code == 404


def test_delete_paper_success():
    """Existing paper → 200, correct chunk count, collection.delete called."""
    mock_vs = _mock_vectorstore(ids=["id1", "id2", "id3"])
    with patch("app.api.routes.get_vectorstore", return_value=mock_vs):
        r = client.delete("/api/papers/Attention Is All You Need")
    assert r.status_code == 200
    data = r.json()
    assert data["deleted"] is True
    assert data["chunks_removed"] == 3
    mock_vs._collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

def test_chat_empty_question_rejected():
    """Empty string → Pydantic min_length validation → 422."""
    r = client.post("/api/chat", json={"question": ""})
    assert r.status_code == 422


def test_chat_success():
    """Valid question → 200 with answer and sources."""
    fake_result = {
        "answer": "The encoder has 6 layers.",
        "sources": [
            {
                "paper_title": "Attention Is All You Need",
                "page": 2,
                "excerpt": "N=6 identical layers...",
            }
        ],
    }
    with patch("app.api.routes._retrieval.query_papers", new=AsyncMock(return_value=fake_result)):
        r = client.post("/api/chat", json={"question": "How many encoder layers?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "The encoder has 6 layers."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["paper_title"] == "Attention Is All You Need"


def test_chat_paper_filter_forwarded():
    """paper_filter is passed through to the retrieval service."""
    fake_result = {"answer": "Some answer.", "sources": []}
    with patch(
        "app.api.routes._retrieval.query_papers", new=AsyncMock(return_value=fake_result)
    ) as mock_q:
        r = client.post(
            "/api/chat",
            json={"question": "What is this?", "paper_filter": "BERT"},
        )
    assert r.status_code == 200
    mock_q.assert_called_once_with(question="What is this?", paper_filter="BERT")


# ---------------------------------------------------------------------------
# Plain-Python runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_health_ok,
        test_list_papers_empty,
        test_list_papers_with_data,
        test_upload_rejects_non_pdf,
        test_upload_success,
        test_delete_paper_not_found,
        test_delete_paper_success,
        test_chat_empty_question_rejected,
        test_chat_success,
        test_chat_paper_filter_forwarded,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
