"""
Retrieval & generation pipeline.

Searches ChromaDB for relevant chunks and uses Gemini to generate a grounded answer.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.prompts import RAG_PROMPT
from app.services.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Searches ChromaDB for relevant chunks and generates a Gemini-powered answer."""

    def __init__(self) -> None:
        self._vectorstore = get_vectorstore()
        self._llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
        )


    async def query_papers(
        self, question: str, paper_filter: str | None = None, k: int | None = None
    ) -> dict:
        """Search indexed papers and generate a grounded answer.

        Args:
            question:     The user's natural language question.
            paper_filter: Optional paper title to restrict the search to one paper.
            k:            Number of chunks to retrieve (default 4).

        Returns:
            A dict with the answer and its source citations::

                {
                    "answer":  str,
                    "sources": [
                        {"paper_title": str, "page": int, "excerpt": str}
                    ]
                }
        """
        k = k if k is not None else settings.retrieval_k
        logger.info("Querying papers for: %r (filter=%r, k=%d)", question, paper_filter, k)

        # 1. Build optional metadata filter
        search_kwargs: dict = {"k": k}
        if paper_filter:
            search_kwargs["filter"] = {"paper_title": paper_filter}

        # 2. Similarity search — returns the k most relevant chunks
        found_docs = self._vectorstore.similarity_search(question, **search_kwargs)
        logger.info("Retrieved %d chunks from ChromaDB", len(found_docs))

        if not found_docs:
            return {
                "answer": "I don't have enough information in the uploaded papers to answer this.",
                "sources": [],
            }

        # 3. Format context from retrieved chunks
        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('paper_title', 'Unknown')} | page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in found_docs
        )

        # 4. Build and invoke the prompt
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = await self._llm.ainvoke(prompt)
        
        answer = self._parse_answer(response.content)

        # 5. Build source citations from retrieved chunk metadata
        sources = [
            {
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "page": doc.metadata.get("page", None),
                "excerpt": doc.page_content[:settings.excerpt_length],
            }
            for doc in found_docs
        ]

        logger.info("Answer generated successfully")
        return {"answer": answer, "sources": sources}
    
    # ----------------- Helper methods -----------------
    @staticmethod
    def _parse_answer(answer) -> str:
        """
        Normalise LLM response content to a plain string.
        In newer SDK versions, content may be a list of parts rather than a string,
        or None when the response has no text content.
        """
        if answer is None:
            return "No text answer."
        if isinstance(answer, list):
            answer = "".join(
                part.get("text", "")
                for part in answer
                if isinstance(part, dict) and "text" in part
            )
        elif isinstance(answer, str) and answer.startswith("[{'type':"):
            import ast
            try:
                parts = ast.literal_eval(answer)
                answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
            except Exception:
                pass
        return answer
