"""Tests for context management: token counting, summarization thresholds, learnings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.context import (
    LEARNINGS_SUMMARY_THRESHOLD,
    SUMMARIZATION_THRESHOLD,
    ContextManager,
    ConversationEntry,
    count_tokens,
    load_learnings,
)


class TestTokenCounting:
    def test_count_tokens_short(self) -> None:
        count = count_tokens("hello world")
        assert count > 0
        assert count < 10

    def test_count_tokens_empty(self) -> None:
        assert count_tokens("") == 0

    def test_count_tokens_long(self) -> None:
        text = "word " * 1000
        count = count_tokens(text)
        assert count > 500  # rough lower bound

    def test_count_tokens_code(self) -> None:
        code = "def foo():\n    return 42\n"
        count = count_tokens(code)
        assert count > 0


class TestConversationEntry:
    def test_auto_token_count(self) -> None:
        entry = ConversationEntry(role="user", content="Hello, how are you?")
        assert entry.token_count > 0

    def test_explicit_token_count(self) -> None:
        entry = ConversationEntry(role="assistant", content="Fine", token_count=99)
        assert entry.token_count == 99


class TestLoadLearnings:
    def test_load_existing(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("# Key Findings\n- Found stuff")
        result = load_learnings(tmp_workspace)
        assert result is not None
        assert "Key Findings" in result

    def test_load_empty(self, tmp_workspace: str) -> None:
        (Path(tmp_workspace) / "learnings.md").write_text("   \n  ")
        result = load_learnings(tmp_workspace)
        assert result is None

    def test_load_missing(self, tmp_workspace: str) -> None:
        result = load_learnings(tmp_workspace)
        assert result is None


def _make_mock_provider():
    """Create a mock provider with a working complete() method."""
    provider = MagicMock()

    def _complete(*, model, system, messages, max_tokens=4000):
        return "Summary of conversation"

    provider.complete.side_effect = _complete
    return provider


class TestContextManager:
    @pytest.fixture()
    def ctx(self, tmp_workspace: str) -> ContextManager:
        provider = _make_mock_provider()
        return ContextManager(
            provider=provider,
            model="gpt-4o",
            workspace=tmp_workspace,
        )

    def test_add_entry_tracks_tokens(self, ctx: ContextManager) -> None:
        ctx.add_entry("user", "Hello world")
        assert len(ctx.entries) == 1
        assert ctx.cumulative_tokens > 0

    def test_cumulative_tokens_grows(self, ctx: ContextManager) -> None:
        ctx.add_entry("user", "Message 1")
        tokens_after_one = ctx.cumulative_tokens
        ctx.add_entry("assistant", "Response 1")
        assert ctx.cumulative_tokens > tokens_after_one

    def test_should_summarize_below_threshold(self, ctx: ContextManager) -> None:
        ctx.add_entry("user", "short message")
        assert ctx.should_summarize() is False

    def test_should_summarize_above_threshold(self, ctx: ContextManager) -> None:
        ctx.cumulative_tokens = SUMMARIZATION_THRESHOLD + 1
        assert ctx.should_summarize() is True

    def test_update_usage(self, ctx: ContextManager) -> None:
        ctx.update_usage(1000, 500)
        assert ctx.last_input_tokens == 1000
        assert ctx.last_output_tokens == 500

    def test_previous_response_id_tracking(self, ctx: ContextManager) -> None:
        assert ctx.previous_response_id is None
        ctx.previous_response_id = "resp_abc123"
        assert ctx.previous_response_id == "resp_abc123"

    def test_summarize_and_fork_clears_chain(self, ctx: ContextManager) -> None:
        """After fork, previous_response_id should be None."""
        ctx.previous_response_id = "resp_old"
        # Add enough entries to summarize
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)

        ctx.summarize_and_fork()
        assert ctx.previous_response_id is None

    def test_summarize_and_fork_reduces_entries(self, ctx: ContextManager) -> None:
        for i in range(20):
            ctx.add_entry("user", f"Message {i} " * 50)
        original_count = len(ctx.entries)

        ctx.summarize_and_fork()
        assert len(ctx.entries) < original_count

    def test_summarize_and_fork_graceful_failure(self, ctx: ContextManager) -> None:
        """If API call fails, should not crash."""
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)
        ctx.provider.complete.side_effect = Exception("API down")

        # Should not raise
        result = ctx.summarize_and_fork()
        assert isinstance(result, str)

    def test_summarize_and_fork_min_split(self, ctx: ContextManager) -> None:
        """With very few entries, split_point should be clamped."""
        ctx.add_entry("user", "only one")

        # Should not crash even with 1 entry
        ctx.summarize_and_fork()

    def test_get_learnings_loads_file(self, ctx: ContextManager) -> None:
        (Path(ctx.workspace) / "learnings.md").write_text("# Findings")
        result = ctx.get_learnings()
        assert result is not None
        assert "Findings" in result

    def test_get_learnings_no_workspace(self) -> None:
        ctx = ContextManager(provider=MagicMock(), model="gpt-4o", workspace=None)
        assert ctx.get_learnings() is None

    def test_get_learnings_no_file(self, ctx: ContextManager) -> None:
        assert ctx.get_learnings() is None
