import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.retrieval import RetrievalService


def main():
    service = RetrievalService()
    
    # Access the underlying Chroma collection
    collection = service._vectorstore._collection
    
    # Get all metadata without loading the heavy vector embeddings
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])
    
    if not metadatas:
        print("📭 The database is completely empty.")
        return

    # Extract unique paper titles and count chunks
    papers = {}
    for meta in metadatas:
        title = meta.get("paper_title", "Unknown")
        papers[title] = papers.get(title, 0) + 1

    print("📚 Papers currently in the database:")
    for title, chunk_count in papers.items():
        print(f"  - {title} ({chunk_count} chunks)")
        
    print(f"\nTotal chunks across all papers: {len(metadatas)}")


if __name__ == "__main__":
    main()
