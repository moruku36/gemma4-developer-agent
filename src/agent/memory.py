"""Conversation memory with token budgeting and tool-output pruning.

Long tool outputs (test logs, big files) are what blow up a 32k context.
Pruning strategy, cheapest first:

1. Truncate every oversized tool observation to head+tail.
2. If still over budget, drop the oldest non-essential turns, replacing
   them with a one-line marker so the model knows history was elided.

The system prompt (index 0) and the task message (index 1) are never
dropped: losing the issue text mid-task is unrecoverable.
"""

from __future__ import annotations

from tools.execution import truncate_middle


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for code-heavy English text)."""
    return max(1, len(text) // 4)


class ConversationMemory:
    """Ordered chat messages plus budget-aware pruning."""

    def __init__(
        self,
        max_context_tokens: int = 28000,
        tool_output_max_chars: int = 6000,
        keep_recent_turns: int = 6,
    ):
        self.messages: list[dict[str, str]] = []
        self.max_context_tokens = max_context_tokens
        self.tool_output_max_chars = tool_output_max_chars
        self.keep_recent_turns = keep_recent_turns
        self.pruned_turns = 0

    # -- mutation --------------------------------------------------------
    def add(self, role: str, content: str) -> None:
        if role == "user" and content.startswith("Observation:"):
            content = truncate_middle(content, self.tool_output_max_chars)
        self.messages.append({"role": role, "content": content})

    def add_observation(self, text: str) -> None:
        self.add("user", f"Observation: {text}")

    # -- accounting ------------------------------------------------------
    def total_tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.messages)

    # -- pruning ---------------------------------------------------------
    def prune_if_needed(self) -> bool:
        """Prune until under budget; return True if anything was removed."""
        pruned = False
        # Phase 1: hard-truncate every observation to half the usual cap.
        if self.total_tokens() > self.max_context_tokens:
            for msg in self.messages:
                if msg["content"].startswith("Observation:"):
                    shrunk = truncate_middle(msg["content"], self.tool_output_max_chars // 2)
                    if shrunk != msg["content"]:
                        msg["content"] = shrunk
                        pruned = True
        # Phase 2: drop oldest turns after the protected prefix (system+task).
        while self.total_tokens() > self.max_context_tokens and len(self.messages) > (
            2 + self.keep_recent_turns
        ):
            dropped = self.messages.pop(2)
            self.pruned_turns += 1
            pruned = True
            if len(self.messages) > 2 and self.messages[2]["content"] != _PRUNE_MARKER:
                self.messages.insert(2, {"role": "user", "content": _PRUNE_MARKER})
            _ = dropped
        return pruned


_PRUNE_MARKER = (
    "Observation: [earlier exploration history was pruned to fit the context "
    "budget; re-read any file you still need]"
)
