"""
Phase 2 smoke test — ingestion pipeline.

Usage:
    ../.venv/bin/python test_phase2.py <path/to/paper.pdf>
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ingestion import IngestionService


async def main() -> None:
    service = IngestionService()

    # --- Test Error path ---
    try:
        await service.ingest_pdf("/nonexistent/fake.pdf", "Ghost Paper")
        raise AssertionError("Expected FileNotFoundError was not raised")
    except FileNotFoundError:
        print("✅ FileNotFoundError raised correctly for missing PDF")

    # --- Test Good path (requires a real PDF + GEMINI_API_KEY) ---
    if len(sys.argv) < 2:
        print("⚠️ No PDF file provided, should be like:")
        print("     ../.venv/bin/python test_phase2.py sample.pdf")
        return

    pdf_path = sys.argv[1]
    title = os.path.basename(pdf_path)
    print(f"\n📄 Ingesting: {pdf_path} as '{title}'")
    result = await service.ingest_pdf(file_path=pdf_path, paper_title=title)

    assert result["pages"] > 0
    assert result["chunks"] > 0
    print("✅ Ingestion succeeded")
    print(f"    Pages  : {result['pages']}")
    print(f"    Chunks : {result['chunks']}")


if __name__ == "__main__":
    asyncio.run(main())
