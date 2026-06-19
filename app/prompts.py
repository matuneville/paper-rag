"""
Prompt templates used across the RAG pipeline.

Kept separate from services so prompts can be iterated without touching
business logic, and separate from config because they are app-logic
constants, not environment-dependent settings.
"""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""\
You are a research assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information in the \
uploaded papers to answer this."

Context:
{context}

Question: {question}

Answer with citations (mention paper title and page number for each claim):""",
)
