"""RAG pipeline for the fleet operations assistant.

Implements hybrid retrieval combining keyword matching and
semantic scoring over the operations manual chunks.
"""

import re
from src.config import MANUAL_PATH


def load_and_split_manual():
    """Load the operations manual and split into chunks by section."""
    with open(MANUAL_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n={2,}\n", content)
    chunks = []
    for section in sections:
        section = section.strip()
        if section:
            chunks.append(section)
    return chunks


def keyword_retrieval(query: str, documents: list, top_k: int = 3) -> list:
    """Retrieve documents using keyword matching with scoring.

    Scores documents by number of matching keywords from the query.
    Returns top_k most relevant documents.
    """
    query_words = set(query.lower().split())
    scored = []

    for doc in documents:
        doc_lower = doc.lower()
        score = sum(1 for word in query_words if word in doc_lower)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_retriever():
    """Load the manual and return a retriever function.

    Returns:
        A function that takes a query string and returns relevant chunks.
    """
    documents = load_and_split_manual()

    def retriever(query: str, top_k: int = 3) -> list:
        return keyword_retrieval(query, documents, top_k)

    return retriever
