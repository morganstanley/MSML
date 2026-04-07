"""Hierarchical context management for alpha-lab.

Three tiers:
  1. Raw conversation — server-side via previous_response_id
  2. Summarized context — triggered when token count is high, forks the chain
  3. Persistent learnings — learnings.md in workspace, always in system prompt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_lab.provider import Provider

# ---------------------------------------------------------------------------
# Token Counting
# ---------------------------------------------------------------------------

# Try tiktoken, fall back to character estimate (~3.5 chars per token)
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        """Estimate token count for a string."""
        return len(_ENCODING.encode(text, disallowed_special=()))
except Exception:
    import logging as _logging
    _logging.getLogger("alpha_lab.context").warning(
        "tiktoken unavailable, using approximate character-based token estimation (~3.5 chars/token)"
    )
    # Offline fallback: ~3.5 characters per token on average
    def count_tokens(text: str) -> int:
        """Estimate token count for a string (character-based fallback)."""
        return int(len(text) / 3.5)


# ---------------------------------------------------------------------------
# Conversation Entry
# ---------------------------------------------------------------------------


@dataclass
class ConversationEntry:
    """A single turn in the conversation for local tracking."""

    role: str  # "user", "assistant", "tool"
    content: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


# ---------------------------------------------------------------------------
# Learnings Manager
# ---------------------------------------------------------------------------

LEARNINGS_SUMMARY_THRESHOLD = 20_000  # tokens


def load_learnings(workspace: str) -> str | None:
    """Load learnings.md from workspace, return content or None."""
    path = Path(workspace) / "learnings.md"
    if path.exists():
        content = path.read_text()
        if content.strip():
            return content
    return None


def summarize_learnings(provider: Any, learnings: str, model: str = "") -> str:
    """Summarize learnings.md if it's gotten too long."""
    if not model:
        raise ValueError("model must be specified for summarize_learnings")
    try:
        return provider.complete(
            model=model,
            system=(
                "You are a summarization assistant. Condense the following "
                "research notes into a concise but comprehensive summary. "
                "Preserve all key findings, data quality issues, and open "
                "questions. Remove redundancy and verbose descriptions. "
                "Keep the same markdown structure."
            ),
            messages=[{"role": "user", "content": learnings}],
            max_tokens=4000,
        )
    except Exception:
        return learnings  # Gracefully degrade: return unsummarized


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

# When cumulative tokens exceed this, trigger summarization + fork
SUMMARIZATION_THRESHOLD = 150_000


@dataclass
class ContextManager:
    """Manages conversation context, summarization, and chain forking."""

    provider: Any  # Provider protocol
    model: str
    workspace: str | None = None
    domain_description: str = ""  # e.g. "CUDA kernel optimization" — used in summarization

    # Server-side chain
    previous_response_id: str | None = None

    # Local tracking for summarization decisions
    entries: list[ConversationEntry] = field(default_factory=list)
    cumulative_tokens: int = 0

    # Summarized context from prior forks
    summary: str | None = None

    # Track API-reported usage for calibration
    last_input_tokens: int = 0
    last_output_tokens: int = 0

    def add_entry(self, role: str, content: str) -> None:
        """Track a conversation turn locally."""
        entry = ConversationEntry(role=role, content=content)
        self.entries.append(entry)
        self.cumulative_tokens += entry.token_count

    def update_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update with API-reported token usage."""
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens

    def should_summarize(self) -> bool:
        """Check if we should trigger summarization and fork."""
        return self.cumulative_tokens > SUMMARIZATION_THRESHOLD

    def summarize_and_fork(self) -> str:
        """Summarize older conversation entries and start a new chain.

        Returns the summary that was generated.
        """
        # Take the older ~60% of entries for summarization
        split_point = int(len(self.entries) * 0.6)
        if split_point < 2:
            split_point = min(2, len(self.entries))

        old_entries = self.entries[:split_point]
        kept_entries = self.entries[split_point:]

        # Build text to summarize
        text_parts = []
        if self.summary:
            text_parts.append(f"Previous summary:\n{self.summary}")
        for entry in old_entries:
            text_parts.append(f"[{entry.role}]: {entry.content}")
        text_to_summarize = "\n\n".join(text_parts)

        # Summarize using the same model as the main agent
        summarization_ok = False
        agent_desc = self.domain_description or "quant research"
        try:
            self.summary = self.provider.complete(
                model=self.model,
                system=(
                    f"Summarize this conversation between a {agent_desc} "
                    "agent and a user. Preserve: key findings, data "
                    "insights, decisions made, errors encountered and "
                    "resolved, current state of analysis. Be concise but "
                    "don't lose important details."
                ),
                messages=[{"role": "user", "content": text_to_summarize}],
                max_tokens=4000,
            )
            summarization_ok = True
        except Exception as e:
            import logging as _logging
            _logging.getLogger("alpha_lab.context").warning(
                "Context summarization failed, keeping full history: %s", e
            )

        if summarization_ok:
            # Only discard old entries if we successfully generated a summary
            self.entries = kept_entries
            self.cumulative_tokens = sum(e.token_count for e in kept_entries)
            # Fork the chain — caller needs to start a new response chain
            self.previous_response_id = None

        return self.summary or ""

    def get_learnings(self) -> str | None:
        """Load and potentially summarize learnings from workspace."""
        if not self.workspace:
            return None

        learnings = load_learnings(self.workspace)
        if learnings is None:
            return None

        token_count = count_tokens(learnings)
        if token_count > LEARNINGS_SUMMARY_THRESHOLD:
            learnings = summarize_learnings(self.provider, learnings, model=self.model)
            # Write summarized version back
            path = Path(self.workspace) / "learnings.md"
            path.write_text(learnings)

        return learnings
