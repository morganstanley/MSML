"""Hierarchical context management for alpha-lab.

Three tiers:
  1. Raw conversation — local history tracking, trimmed on summarization
  2. Summarized context — triggered when token count is high, forks the chain
  3. Persistent learnings — learnings.md in workspace, always in system prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.context")

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
        "tiktoken unavailable, using character-based token estimation (~3x less accurate)"
    )
    # Offline fallback: ~3.5 characters per token on average
    def count_tokens(text: str) -> int:
        """Estimate token count for a string (character-based fallback)."""
        return len(text) // 3


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
            try:
                from alpha_lab.memory import remember_workspace_file
                remember_workspace_file(
                    workspace,
                    "learnings.md",
                    tags=["phase1", "learnings"],
                    summary="Phase 1 learnings",
                    kind="finding",
                    phase="phase1",
                    agent="phase1",
                )
            except Exception as e:
                logger.warning("Failed to ingest learnings into memory: %s", e)
            return content
    return None


def summarize_learnings(provider: Any, learnings: str, model: str) -> str:
    """Summarize learnings.md if it's gotten too long.

    Parameters
    ----------
    model : str
        The model to use for summarization (caller passes config model).
    """
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

    def summarize_and_fork(
        self, history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Summarize older conversation entries and trim history.

        Parameters
        ----------
        history : list or None
            The provider-native ``_input_history`` from the agent loop.
            When provided (and summarization succeeds), a trimmed copy is
            returned so the caller can replace its history.

        Returns
        -------
        (summary_text, trimmed_history_or_None)
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

        # Summarize using the configured model
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
            logger.warning(
                "Context summarization failed, keeping full history: %s", e
            )

        trimmed_history: list[dict[str, Any]] | None = None
        if summarization_ok:
            # Only discard old entries if we successfully generated a summary
            self.entries = kept_entries
            self.cumulative_tokens = sum(e.token_count for e in kept_entries)
            # Fork the chain — caller needs to start a new response chain
            self.previous_response_id = None
            # Trim the actual provider history
            if history is not None and self.summary:
                trimmed_history = self.trim_history(history, self.summary)

        return self.summary or "", trimmed_history

    def trim_history(
        self,
        history: list[dict[str, Any]],
        summary_text: str,
    ) -> list[dict[str, Any]]:
        """Replace older history items with a summary message.

        Keeps the most recent ~40% of history items.  The older items are
        replaced by a single user message containing the summary.
        """
        split_point = int(len(history) * 0.6)
        if split_point < 2:
            split_point = min(2, len(history))

        summary_items = self.provider.build_user_items(
            f"[CONTEXT SUMMARY FROM EARLIER CONVERSATION]\n{summary_text}"
        )
        return summary_items + history[split_point:]

    # ------------------------------------------------------------------
    # Tool output compaction
    # ------------------------------------------------------------------

    @staticmethod
    def compact_tool_output(
        output: str,
        tool_name: str,
        max_chars: int = 8000,
    ) -> str:
        """Compress large tool outputs, keeping head and tail.

        For ``shell_exec`` and ``read_file``: keeps the first and last
        portions with a trimmed marker in between.  Other tools are
        simply truncated.
        """
        if len(output) <= max_chars:
            return output

        keep_each = max_chars // 2  # ~3000-4000 chars each side
        trimmed_count = len(output) - keep_each * 2

        if tool_name in ("shell_exec", "read_file"):
            return (
                output[:keep_each]
                + f"\n[...trimmed {trimmed_count} chars...]\n"
                + output[-keep_each:]
            )
        # Other tools: simple truncation
        return output[:max_chars] + f"\n[...truncated {len(output) - max_chars} chars...]"

    # ------------------------------------------------------------------
    # Learnings
    # ------------------------------------------------------------------

    def get_learnings(self) -> str | None:
        """Load and potentially summarize learnings from workspace.

        If summarization is triggered, the original ``learnings.md`` is
        archived under ``.memory/learnings_archive/`` before being
        overwritten.
        """
        if not self.workspace:
            return None

        learnings = load_learnings(self.workspace)
        if learnings is None:
            return None

        token_count = count_tokens(learnings)
        if token_count > LEARNINGS_SUMMARY_THRESHOLD:
            # Archive original before overwriting
            self._archive_learnings(learnings)
            learnings = summarize_learnings(self.provider, learnings, self.model)
            # Write summarized version back
            path = Path(self.workspace) / "learnings.md"
            path.write_text(learnings)

        return learnings

    def _archive_learnings(self, content: str) -> None:
        """Save a timestamped copy of learnings before summarization."""
        if not self.workspace:
            return
        archive_dir = Path(self.workspace) / ".memory" / "learnings_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"learnings_{timestamp}.md"
        try:
            archive_path.write_text(content)
            logger.info("Archived learnings to %s", archive_path)
        except OSError as e:
            logger.warning("Failed to archive learnings: %s", e)
