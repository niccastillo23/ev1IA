"""Conversational memory for the Seguros Express assistant.

Two complementary memories:

- ``ConversationBufferWindowMemory`` (short-term): sliding-window buffer that
  keeps the last N turns for the current session. Optionally persisted to disk.
- ``LongTermMemory`` (long-term): semantic store (Chroma + Mistral embeddings)
  of past exchanges, so relevant information survives across sessions and can
  be recalled by meaning.
"""

import json
import os
import time

from src.config import (
    LONGTERM_DB_PATH,
    MEMORY_MAX_TURNS,
    MEMORY_PERSIST_PATH,
)


class ConversationBufferWindowMemory:
    """Sliding-window buffer memory for multi-turn conversations.

    Stores the last `max_turns` exchanges (user + assistant) and exposes
    them in the OpenAI chat format. The window slides as new turns arrive.
    """

    def __init__(self, max_turns: int = 5, persist_path: str = None):
        self.max_turns = max_turns
        self.persist_path = persist_path
        self.messages = []

    def add_user(self, content: str) -> None:
        """Append a user message to the buffer."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Append an assistant message to the buffer."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        """Keep only the last `max_turns` exchanges (2 messages each)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self) -> list:
        """Return a copy of the current window in OpenAI chat format."""
        return list(self.messages)

    def clear(self) -> None:
        """Reset the conversation buffer."""
        self.messages = []

    def save(self) -> None:
        """Persist the buffer to disk if a path is configured."""
        if not self.persist_path:
            return
        directory = os.path.dirname(self.persist_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load the buffer from disk if the file exists."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
            self._trim()
        except (json.JSONDecodeError, OSError):
            self.messages = []


def build_memory() -> ConversationBufferWindowMemory:
    """Create the session memory with the configured window size."""
    memory = ConversationBufferWindowMemory(
        max_turns=MEMORY_MAX_TURNS,
        persist_path=MEMORY_PERSIST_PATH,
    )
    memory.load()
    return memory


class LongTermMemory:
    """Semantic long-term memory backed by Chroma + Mistral embeddings.

    Stores past exchanges as documents and recalls the most relevant ones by
    cosine similarity, so the assistant keeps useful context across sessions.
    """

    def __init__(self, persist_path=None, collection_name: str = "memoria_largo_plazo"):
        self.persist_path = str(persist_path or LONGTERM_DB_PATH)
        self.collection_name = collection_name
        self._store = None
        self._embeddings = None
        self._available = True

    def _ensure_store(self):
        if self._store is not None or not self._available:
            return
        try:
            from langchain_chroma import Chroma

            from src.vectorstore import get_embeddings

            self._embeddings = get_embeddings()
            os.makedirs(self.persist_path, exist_ok=True)
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_path,
            )
        except Exception as exc:  # noqa: BLE001 - memory is optional
            print(f"  [Memoria] Largo plazo deshabilitada: {exc}")
            self._available = False

    def remember(self, text: str, metadata: dict = None) -> None:
        """Store an exchange (question + answer) as a semantic memory."""
        if not text or not text.strip():
            return
        self._ensure_store()
        if not self._available or self._store is None:
            return
        meta = {"timestamp": time.time()}
        if metadata:
            meta.update(metadata)
        try:
            self._store.add_texts([text], metadatas=[meta])
        except Exception as exc:  # noqa: BLE001
            print(f"  [Memoria] No se pudo guardar en largo plazo: {exc}")

    def recall(self, query: str, k: int = 2, threshold: float = 0.6) -> list:
        """Return relevant past exchanges as a list of strings."""
        self._ensure_store()
        if not self._available or self._store is None:
            return []
        try:
            hits = self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:  # noqa: BLE001
            return []
        return [doc.page_content for doc, score in hits if score >= threshold]
