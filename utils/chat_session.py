"""
chat/session_manager.py
Handles ephemeral chat session history (user ↔ assistant turns).
"""

from typing import List, Dict


class SessionManager:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        """Add a single message turn."""
        self.history.append({"role": role, "content": content})

    def get_context(self, limit: int = 5) -> str:
        """Return the last N turns as formatted text context."""
        recent = self.history[-limit * 2 :]  # limit pairs
        context_blocks = [
            f"{h['role'].capitalize()}: {h['content']}" for h in recent
        ]
        return "\n".join(context_blocks)

    def clear(self):
        """Reset session memory."""
        self.history.clear()
