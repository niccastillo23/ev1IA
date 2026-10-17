"""Vector store (NVIDIA embeddings + Chroma) for the Seguros Express manual.

Builds a persistent Chroma index over the manual and exposes a similarity
search that can filter by a relevance threshold. The threshold keeps the
anti-hallucination behaviour: off-topic questions return no context.

The index is rebuilt only when the manual OR the embedding model changes
(fingerprint check), so embeddings are computed once and then reused from disk.
"""

import hashlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    MANUAL_PATH,
    VECTOR_DB_PATH,
)

COLLECTION_NAME = "manual_seguros"
FINGERPRINT_FILE = "fingerprint.txt"
# Use cosine space so relevance scores map cleanly to cosine similarity in [0, 1].
COLLECTION_METADATA = {"hnsw:space": "cosine"}


class NVIDIAEmbeddings(Embeddings):
    """OpenAI-compatible embeddings wrapper for NVIDIA NIM.

    NVIDIA's asymmetric retrieval models (``*-embedqa-*`` and
    ``llama-nemotron-embed-vl-*``) require an ``input_type`` of ``query`` for
    search queries and ``passage`` for indexed documents. LangChain's
    ``Embeddings`` interface splits those two paths, so this wrapper maps
    ``embed_query`` -> ``query`` and ``embed_documents`` -> ``passage``.
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def _embed(self, texts: list, input_type: str) -> list:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            extra_body={"input_type": input_type},
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list) -> list:
        return self._embed(list(texts), "passage")

    def embed_query(self, text: str) -> list:
        return self._embed([text], "query")[0]


def get_embeddings() -> Embeddings:
    """Return the NVIDIA NIM embedding client (asymmetric query/passage)."""
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )


def _manual_fingerprint() -> str:
    """Hash of the manual AND the embedding model, so switching providers
    (or models) forces a rebuild instead of reusing incompatible vectors."""
    digest = hashlib.sha256()
    digest.update(MANUAL_PATH.read_bytes())
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    return digest.hexdigest()[:16]


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
            collection_metadata=COLLECTION_METADATA,
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
        collection_metadata=COLLECTION_METADATA,
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
