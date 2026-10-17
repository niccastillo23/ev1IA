"""Configuration module for the Seguros Express assistant.

Loads environment variables from .env file with sensible defaults.
The chat LLM and the embeddings use an OpenAI-compatible endpoint
(NVIDIA NIM by default: https://integrate.api.nvidia.com/v1), so a single
``LLM_API_KEY`` powers the whole pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_MODEL_SMALL = os.getenv("LLM_MODEL_SMALL", "openai/gpt-oss-20b")

# --- Chat provider ---------------------------------------------------------
# "nvidia": remote OpenAI-compatible endpoint (see LLM_BASE_URL).
# "ollama": local models via http://localhost:11434 (fast, offline, free).
# Embeddings always use the OpenAI-compatible endpoint above.
PROVIDER = os.getenv("PROVIDER", "nvidia").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
# Context window for local models (Ollama's default can be too small for RAG).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Cheap model for the router/supervisor (local model when using Ollama).
CHAT_MODEL_SMALL = OLLAMA_MODEL if PROVIDER == "ollama" else LLM_MODEL_SMALL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = PROJECT_ROOT / "data" / "manual_operaciones_seguros.txt"
# Write-side store: siniestros registered by the agent (herramienta de escritura).
SINIESTROS_DB_PATH = PROJECT_ROOT / os.getenv("SINIESTROS_DB_PATH", "data/siniestros.json")

# RAG with embeddings: OpenAI-compatible embeddings in a persistent Chroma index.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
VECTOR_DB_PATH = PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "data/chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
# Minimum cosine relevance for a vector fragment to be considered on-topic.
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.30"))

# --- Cost / safety caps ----------------------------------------------------
# Hard cap on generated tokens per model call.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
# Max graph steps before LangGraph aborts (prevents runaway ReAct tool loops).
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "6"))
# Use the LLM supervisor only for ambiguous queries (keyword router runs first).
USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "false").lower() in ("1", "true", "yes")
# Enable the semantic long-term memory (adds embedding calls + injected context).
USE_LONG_TERM = os.getenv("USE_LONG_TERM", "false").lower() in ("1", "true", "yes")

# Conversational memory: sliding-window buffer (short-term) persisted to disk.
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "5"))
MEMORY_PERSIST_PATH = os.getenv("MEMORY_PERSIST_PATH", ".memory/session.json")

# Long-term memory: semantic store of past conversation summaries.
LONGTERM_DB_PATH = PROJECT_ROOT / os.getenv("LONGTERM_DB_PATH", "data/chroma_memory")
