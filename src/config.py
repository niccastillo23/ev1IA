"""Configuration module for the fleet operations assistant.

Loads environment variables from .env file with sensible defaults
for development and testing purposes. Uses Groq as LLM provider.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-235b-a22b")
LLM_MODEL_SMALL = os.getenv("LLM_MODEL_SMALL", "qwen/qwen3-30b-a3b")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_PATH = PROJECT_ROOT / "data" / "manual_operaciones_seguros.txt"

# Conversational memory: sliding-window buffer (ConversationBufferWindowMemory)
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "5"))
MEMORY_PERSIST_PATH = os.getenv("MEMORY_PERSIST_PATH", ".memory/session.json")
