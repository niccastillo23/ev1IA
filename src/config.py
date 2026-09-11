"""Configuration module for the fleet operations assistant.

Loads environment variables from .env file with sensible defaults
for development and testing purposes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small-latest")
LLM_EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "mistral-embed")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fleet_operations")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = PROJECT_ROOT / "data" / "manual_operaciones_logistica.txt"
