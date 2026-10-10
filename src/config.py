"""Configuration module for the Seguros Express assistant.

Loads environment variables from .env file with sensible defaults.
The LLM (chat) and the embeddings use Mistral, so a single
``LLM_API_KEY`` powers the whole pipeline (Mistral has an OpenAI-compatible
chat API and a native embedding endpoint).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.mistral.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")
LLM_MODEL_SMALL = os.getenv("LLM_MODEL_SMALL", "ministral-8b-latest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = PROJECT_ROOT / "data" / "manual_operaciones_seguros.txt"

# RAG with embeddings: Mistral embeddings stored in a persistent Chroma index.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
VECTOR_DB_PATH = PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "data/chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
# Minimum cosine relevance for a vector fragment to be considered on-topic.
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.68"))

# Conversational memory: sliding-window buffer (short-term) persisted to disk.
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "5"))
MEMORY_PERSIST_PATH = os.getenv("MEMORY_PERSIST_PATH", ".memory/session.json")

# Long-term memory: semantic store of past conversation summaries.
LONGTERM_DB_PATH = PROJECT_ROOT / os.getenv("LONGTERM_DB_PATH", "data/chroma_memory")
