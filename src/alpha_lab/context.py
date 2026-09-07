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

from alpha_lab.utils import atomic_write

if TYPE_CHECKING:
    from alpha_lab.providers import Provider

logger = logging.getLogger("alpha_lab.context")


def _is_tool_result_item(item: object) -> bool:
    """True if a history item carries a tool RESULT (the answer to a tool call).

    Works across provider history formats:
      * Chat API: a message with ``{"role": "tool", ...}`` — chat endpoints
        reject a tool message whose originating assistant tool_call is
        missing from history.
      * Bedrock Converse: a message whose content has a ``toolResult`` block.
      * Anthropic Messages: a user message carrying a ``{"type": "tool_result"}``
        content block (providers/anthropic.py history shape).
      * OpenAI Responses: a flat ``{"type": "function_call_output", ...}`` item
        — the API rejects an output whose originating call is missing.

    Also treats image-injection user messages (injected after view_image
    results by build_tool_result_items: ``input_image`` parts on the OpenAI
    path, ``image_url`` parts on the local/chat path) as part of the
    tool-result block so trim_history never starts the kept tail on an
    orphan tool/image pair.
    """
    if not isinstance(item, dict):
        return False
    if item.get("role") == "tool":
        return True
    if item.get("type") == "function_call_output":
        return True
    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if "toolResult" in block or block.get("type") == "tool_result":
                return True
            if item.get("role") == "user" and block.get("type") in (
                "image_url", "input_image",
            ):
                return True
    return False


def _is_tool_call_item(item: object) -> bool:
    """Mirror of _is_tool_result_item for the call side, across formats:
    Chat API assistant messages with ``tool_calls``, Bedrock ``toolUse``
    blocks, Anthropic ``{"type": "tool_use"}`` content blocks, OpenAI
    Responses flat ``function_call`` items. The split must also never land
    INSIDE a batch of parallel tool calls (dropping one call while keeping
    its result — an orphan the providers 400 on)."""
    if not isinstance(item, dict):
        return False
    if item.get("type") == "function_call":
        return True
    if item.get("role") == "assistant" and item.get("tool_calls"):
        return True
    content = item.get("content")
    if isinstance(content, list):
        if any(
            isinstance(b, dict)
            and ("toolUse" in b or b.get("type") == "tool_use")
            for b in content
        ):
            return True
    return False


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
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not content.strip():
        return None
    return content


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

# Summary budget policy: target ~1/SUMMARY_COMPACTION_RATIO of the material
# being summarized, clamped to [SUMMARY_BUDGET_MIN_TOKENS,
# SUMMARY_BUDGET_MAX_TOKENS]. The floor keeps small summaries useful; the
# cap stays under provider output limits.
SUMMARY_BUDGET_MIN_TOKENS = 6_000
SUMMARY_BUDGET_MAX_TOKENS = 24_000
SUMMARY_COMPACTION_RATIO = 7


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
        """Check if we should trigger summarization and fork.

        Compares against the larger of two signals: the local
        ``cumulative_tokens`` accumulator and the API-reported
        ``last_input_tokens``. The local accumulator only sees tool outputs
        truncated to 500 chars at the agent-loop call site, so it
        structurally undercounts real context; the API's own reported
        prompt size is ground truth for whether the conversation is
        approaching the model's context limit.
        """
        return (
            self.cumulative_tokens > SUMMARIZATION_THRESHOLD
            or self.last_input_tokens > SUMMARIZATION_THRESHOLD
        )

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

        # Summarize using the configured model. The summary budget scales
        # with what is being summarized: a flat 4,000-token summary of a
        # 150k-token history is a ~37:1 crush that discards most of what
        # the model was asked to preserve.
        summarization_ok = False
        # Domain-agnostic default: the domain lives in domain_description
        # (adapter-provided); the kernel must not assume a field.
        agent_desc = self.domain_description or "research"
        tokens_to_summarize = sum(e.token_count for e in old_entries)
        if self.summary:
            tokens_to_summarize += count_tokens(self.summary)
        summary_budget = max(
            SUMMARY_BUDGET_MIN_TOKENS,
            min(SUMMARY_BUDGET_MAX_TOKENS,
                tokens_to_summarize // SUMMARY_COMPACTION_RATIO),
        )
        try:
            summary_text = self.provider.complete(
                model=self.model,
                system=(
                    f"Summarize this conversation between a {agent_desc} "
                    "agent and a user. Preserve: key findings, data "
                    "insights, decisions made, errors encountered and "
                    "resolved, current state of analysis. Be concise but "
                    "don't lose important details."
                ),
                messages=[{"role": "user", "content": text_to_summarize}],
                max_tokens=summary_budget,
            )
            # An empty completion is a failure, not a summary: treating it
            # as success would drop entries, reset last_input_tokens, and
            # clobber any previous summary — while the oversized provider
            # history survives untrimmed (trim requires a non-empty
            # summary), defeating the protection entirely.
            if summary_text and summary_text.strip():
                self.summary = summary_text
                summarization_ok = True
            else:
                logger.warning(
                    "Context summarization returned an empty summary, "
                    "keeping full history"
                )
        except Exception as e:
            logger.warning(
                "Context summarization failed, keeping full history: %s", e
            )

        trimmed_history: list[dict[str, Any]] | None = None
        if summarization_ok:
            # Only discard old entries if we successfully generated a summary
            self.entries = kept_entries
            self.cumulative_tokens = sum(e.token_count for e in kept_entries)
            # The last API-reported prompt size described the pre-fork
            # history; keeping it would re-trigger should_summarize (e.g.
            # the top-of-turn check) before any new response refreshes it.
            self.last_input_tokens = 0
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

        # The split point must never land between a tool call and the result
        # that answers it, in ANY provider format (OpenAI Responses flat
        # items, chat messages with role "tool", Bedrock toolUse/toolResult
        # blocks, Anthropic tool_use/tool_result blocks). Cutting at an
        # arbitrary index can keep a result whose originating call was
        # trimmed away — an orphan every provider rejects on each retry,
        # permanently poisoning the conversation (see PR #258 for the
        # production evidence). Back the split up to the nearest boundary
        # that does not start the kept tail on an orphan; if none exists,
        # leave history untouched rather than risk a corrupt payload.
        def _unsafe_boundary(it: object) -> bool:
            return _is_tool_result_item(it) or _is_tool_call_item(it)

        # split_point == len(history) is a legal blind cut (tiny histories:
        # min(2, len) can equal len) — an empty kept tail has no orphan, so
        # only walk boundaries strictly inside the list.
        while (
            1 < split_point < len(history)
            and _unsafe_boundary(history[split_point])
        ):
            split_point -= 1
        if split_point < len(history) and _unsafe_boundary(history[split_point]):
            logger.warning(
                "trim_history: no safe cut boundary found "
                "(history is a contiguous tool-call chain); keeping full "
                "history untrimmed rather than risking an orphaned tool item"
            )
            return history

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
        archived under ``.alpha_lab/memory/learnings_archive/`` before being
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
            atomic_write(path, learnings)

        return learnings

    def _archive_learnings(self, content: str) -> None:
        """Save a timestamped copy of learnings before summarization."""
        if not self.workspace:
            return
        archive_dir = Path(self.workspace) / ".alpha_lab" / "memory" / "learnings_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"learnings_{timestamp}.md"
        try:
            atomic_write(archive_path, content)
            logger.info("Archived learnings to %s", archive_path)
        except OSError as e:
            logger.warning("Failed to archive learnings: %s", e)
