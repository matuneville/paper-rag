"""
HTTP client for the paper-rag FastAPI backend.

All network calls go through here — one place to change the base URL,
add auth headers, or swap the HTTP library later.
"""

from typing import Any

import requests
from urllib.parse import quote

API_BASE = "http://localhost:8000"

# Reuse a single session for connection pooling
_session = requests.Session()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(path: str, **kwargs) -> tuple[Any, str | None]:
    try:
        r = _session.get(f"{API_BASE}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API. Is the FastAPI server running on port 8000?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


def _post(path: str, **kwargs) -> tuple[Any, str | None]:
    try:
        r = _session.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API. Is the FastAPI server running on port 8000?"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


def _delete(path: str) -> tuple[Any, str | None]:
    try:
        r = _session.delete(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach API."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, f"API error {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def health() -> tuple[dict, str | None]:
    """GET /health"""
    return _get("/health")


def list_papers() -> tuple[dict, str | None]:
    """GET /api/papers"""
    return _get("/api/papers")


def upload_paper(filename: str, content: bytes, paper_title: str | None = None) -> tuple[dict, str | None]:
    """POST /api/papers/upload"""
    data = {}
    if paper_title is not None:
        data["paper_title"] = paper_title
    return _post(
        "/api/papers/upload",
        files={"file": (filename, content, "application/pdf")},
        data=data,
    )


def delete_paper(paper_title: str) -> tuple[dict, str | None]:
    """DELETE /api/papers/{paper_title}"""
    encoded = quote(paper_title, safe="")
    return _delete(f"/api/papers/{encoded}")


def chat(question: str, paper_filter: str | None = None) -> tuple[dict, str | None]:
    """POST /api/chat"""
    payload: dict = {"question": question}
    if paper_filter:
        payload["paper_filter"] = paper_filter
    return _post("/api/chat", json=payload)
