"""Conversational memory for the fleet operations assistant.

Implements a sliding-window buffer memory (ConversationBufferWindowMemory):
it keeps the last N conversation turns, discarding the oldest ones to
avoid growing the LLM context indefinitely. Optionally persists the
session to disk so it survives between runs.
"""

import json
import os

from src.config import MEMORY_MAX_TURNS, MEMORY_PERSIST_PATH


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
