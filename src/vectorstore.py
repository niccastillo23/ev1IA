"""Vector store (Mistral embeddings + Chroma) for the Seguros Express manual.

Builds a persistent Chroma index over the manual and exposes a similarity
search that can filter by a relevance threshold. The threshold keeps the
anti-hallucination behaviour: off-topic questions return no context.

The index is rebuilt only when the manual changes (fingerprint check), so
embeddings are computed once and then reused from disk.
"""

import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    MANUAL_PATH,
    VECTOR_DB_PATH,
)

COLLECTION_NAME = "manual_seguros"
FINGERPRINT_FILE = "fingerprint.txt"


def get_embeddings() -> MistralAIEmbeddings:
    """Return the Mistral embedding client (model `mistral-embed`)."""
    return MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=LLM_API_KEY)


def _manual_fingerprint() -> str:
    """Short hash of the manual so the index is rebuilt only when it changes."""
    return hashlib.sha256(MANUAL_PATH.read_bytes()).hexdigest()[:16]


def _load_documents() -> list:
    """Load the manual as one LangChain Document per section."""
    # Imported lazily to avoid a circular import at module load time.
    from src.rag_pipeline import load_and_split_manual

    documents = []
    for section in load_and_split_manual():
        title = section.splitlines()[0].strip() if section else ""
        documents.append(Document(page_content=section, metadata={"section": title}))
    return documents


def _split_documents(documents: list) -> list:
    """Chunk the sections into overlapping fragments for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Build (or load) the persistent Chroma index of the manual."""
    path = Path(VECTOR_DB_PATH)
    embeddings = get_embeddings()
    fingerprint = _manual_fingerprint()
    fingerprint_file = path / FINGERPRINT_FILE

    if (
        not force_rebuild
        and fingerprint_file.exists()
        and fingerprint_file.read_text(encoding="utf-8").strip() == fingerprint
    ):
        return Chroma(
            persist_directory=str(path),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    chunks = _split_documents(_load_documents())
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(path),
        collection_name=COLLECTION_NAME,
    )
    fingerprint_file.write_text(fingerprint, encoding="utf-8")
    return store


def build_vector_search(force_rebuild: bool = False):
    """Return a `search(query, k, threshold) -> list[(Document, score)]` function.

    Only fragments whose cosine relevance score reaches `threshold` are
    returned, which lets the hybrid retriever fall back to returning nothing
    for off-topic questions.
    """
    store = build_vectorstore(force_rebuild=force_rebuild)

    def search(query: str, k: int = 4, threshold: float = 0.0):
        results = store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, score) for doc, score in results if score >= threshold]

    return search
