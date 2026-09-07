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

# Default used when ContextManager is constructed without an explicit value
# (tests, ad-hoc callers). The production path threads
# config.learnings_summary_threshold_tokens into ContextManager(...).
LEARNINGS_SUMMARY_THRESHOLD = 20_000  # tokens


def load_learnings(workspace: str) -> str | None:
    """Load learnings.md from workspace, return content or None."""
    path = Path(workspace) / "learnings.md"
    if path.exists():
        content = path.read_text()
        if content.strip():
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

# Default used when ContextManager is constructed without an explicit value
# (tests, ad-hoc callers). The production path threads
# config.context_summarization_threshold_tokens into ContextManager(...).
SUMMARIZATION_THRESHOLD = 150_000


def _is_tool_result_item(item: object) -> bool:
    """True if a history item carries a tool RESULT (the answer to a tool call).

    Works across provider history formats:
      * Chat API: a message with ``{"role": "tool", ...}`` (ChatProvider).
      * Bedrock: a role/content message whose content has a ``toolResult`` block.
      * OpenAI Responses: a flat ``{"type": "function_call_output", ...}`` item.

    Also treats image-injection user messages (injected after view_image results
    by build_tool_result_items) as part of the tool-result block so trim_history
    never starts the kept tail on an orphan tool/image pair.

    Used by ``trim_history`` to avoid cutting between a tool call and its
    result, which would orphan the result and trigger a provider 400.
    """
    if not isinstance(item, dict):
        return False
    # Chat API tool result: {"role": "tool", "tool_call_id": ..., "content": ...}
    if item.get("role") == "tool":
        return True
    if item.get("type") == "function_call_output":
        return True
    content = item.get("content")
    if isinstance(content, list):
        # Bedrock toolResult block
        if any(isinstance(b, dict) and "toolResult" in b for b in content):
            return True
        # Image-injection user messages from build_tool_result_items: user message
        # whose content list contains only image_url / text blocks (no tool role,
        # but logically part of a tool sequence — cutting here is also unsafe
        # because the preceding tool message may be trimmed away).
        if item.get("role") == "user" and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in content
        ):
            return True
    return False


def _is_tool_call_item(item: object) -> bool:
    """True if a history item carries a tool CALL (a function/tool invocation) — the mirror
    of _is_tool_result_item for the call side, across provider formats:
      * Chat API: an assistant message with ``tool_calls``.
      * Bedrock: a role/content message whose content has a ``toolUse`` block.
      * OpenAI Responses: a flat ``{"type": "function_call", ...}`` item.

    Used by ``trim_history`` so the split also never lands INSIDE a batch of parallel tool
    calls (which would drop one call while keeping its result — an orphan the providers 400 on).
    """
    if not isinstance(item, dict):
        return False
    if item.get("type") == "function_call":
        return True
    if item.get("role") == "assistant" and item.get("tool_calls"):
        return True
    content = item.get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and "toolUse" in b for b in content):
            return True
    return False


@dataclass
class ContextManager:
    """Manages conversation context, summarization, and chain forking."""

    provider: Any  # Provider protocol
    model: str
    workspace: str | None = None
    domain_description: str = ""  # e.g. "CUDA kernel optimization" — used in summarization

    # Token thresholds — defaults match the previous module-level constants
    # and are overridden by TaskConfig.context_summarization_threshold_tokens
    # / .learnings_summary_threshold_tokens at construction time.
    summarization_threshold_tokens: int = SUMMARIZATION_THRESHOLD
    learnings_summary_threshold_tokens: int = LEARNINGS_SUMMARY_THRESHOLD

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
        ``last_input_tokens``. The former misses big tool outputs (they're
        truncated to 500 chars in the local accumulator at the agent-loop
        call site), so the API's own reported prompt size is the ground
        truth for whether we're approaching the model's context limit.
        """
        return (
            self.cumulative_tokens > self.summarization_threshold_tokens
            or self.last_input_tokens > self.summarization_threshold_tokens
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
        # with what is being summarized (target ~7:1 compaction, i.e.
        # within the user-mandated 5:1..10:1 band, 2026-08-02): a flat
        # 4,000-token summary of a 150k-token history was a ~37:1 crush
        # that destroyed most of what the article-style compaction is
        # supposed to preserve. Floor keeps small summaries useful; cap
        # stays under every provider's output ceiling.
        summarization_ok = False
        agent_desc = self.domain_description or "quant research"
        tokens_to_summarize = sum(e.token_count for e in old_entries)
        if self.summary:
            tokens_to_summarize += count_tokens(self.summary)
        summary_budget = max(6_000, min(24_000, tokens_to_summarize // 7))
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
                max_tokens=summary_budget,
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

        The split point must never land between a tool call and the result
        that answers it. Cutting at an arbitrary index can keep a
        tool-result whose originating tool-use was trimmed away, leaving an
        orphan that Bedrock (``toolResult`` with no ``toolUse``) and the
        OpenAI Responses API (``function_call_output`` with no
        ``function_call``) both reject with a 400. We back the split up to
        the nearest boundary that does not start the kept tail on an orphan.
        """
        split_point = int(len(history) * 0.6)
        if split_point < 2:
            split_point = min(2, len(history))

        # Back up off any tool-result OR tool-call boundary so the kept tail begins on a
        # CLEAN turn (an assistant text message or a genuine user message), keeping each
        # tool-call/tool-result pair — INCLUDING a batch of PARALLEL calls — fully intact.
        # (Backing off results alone is not enough: with parallel calls
        # [call_A, call_B, out_A, out_B] a split among the outputs stops at call_B and drops
        # call_A while keeping out_A — an orphaned function_call_output the OpenAI Responses
        # API rejects with "No tool call found for function call output".)
        def _unsafe_boundary(it: object) -> bool:
            return _is_tool_result_item(it) or _is_tool_call_item(it)
        while split_point > 1 and _unsafe_boundary(history[split_point]):
            split_point -= 1
        if split_point < len(history) and _unsafe_boundary(history[split_point]):
            # No safe boundary found (pathological) — don't risk a corrupt
            # payload; leave history untouched and let the next turn retry.
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
        archived under ``.memory/learnings_archive/`` before being
        overwritten.
        """
        if not self.workspace:
            return None

        learnings = load_learnings(self.workspace)
        if learnings is None:
            return None

        token_count = count_tokens(learnings)
        if token_count > self.learnings_summary_threshold_tokens:
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
