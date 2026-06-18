"""
Retrieval & generation pipeline.

Searches ChromaDB for relevant chunks and uses Gemini to generate a grounded answer.
"""

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a research assistant. Answer the question using ONLY the context provided.
        If the answer is not in the context, say "I don't have enough information in the uploaded papers to answer this."

        Context:
        {context}

        Question: {question}

        Answer with citations (mention paper title and page number for each claim):
        """,
)


class RetrievalService:
    """Searches ChromaDB for relevant chunks and generates a Gemini-powered answer."""

    def __init__(self) -> None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
        )
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.gemini_api_key,
        )


    async def query_papers(
        self, question: str, paper_filter: str | None = None, k: int = 4
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
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
        response = await self._llm.ainvoke(prompt)
        
        answer = self._parse_answer(response.content)

        # 5. Build source citations from retrieved chunk metadata
        sources = [
            {
                "paper_title": doc.metadata.get("paper_title", "Unknown"),
                "page": doc.metadata.get("page", None),
                "excerpt": doc.page_content[:300],
            }
            for doc in found_docs
        ]

        logger.info("Answer generated successfully")
        return {"answer": answer, "sources": sources}
    
    # ----------------- Helper methods -----------------
    @staticmethod
    def _parse_answer(answer):
        """
        This method parses the answer to ensure it is a string.
        In newer SDK versions, content may be either a list of parts rather than a string.
        """
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
