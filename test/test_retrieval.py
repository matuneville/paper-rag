"""
Phase 3 smoke test — retrieval & generation pipeline.

Requires the sample PDF to have already been ingested (run test_ingestion.py first).

Usage:
    ../.venv/bin/python test_retrieval.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.retrieval import RetrievalService


async def main() -> None:
    service = RetrievalService()

    question = "How many encoder layers does the classic transformer have?"
    print(f"❓ Question: {question}\n")

    result = await service.query_papers(question)

    assert "answer" in result
    assert "sources" in result

    print(f"💬 Answer:\n{result['answer']}\n")
    print(f"📚 Sources ({len(result['sources'])}):")
    for src in result["sources"]:
        print(f"  - {src['paper_title']} | page {src['page']}")
        print(f"    \"{src['excerpt'][:100]}...\"")

    print("\n✅ Retrieval succeeded")


if __name__ == "__main__":
    asyncio.run(main())
