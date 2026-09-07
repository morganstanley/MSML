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
    """Create a mock provider with a working complete() and build_user_items()."""
    provider = MagicMock()

    def _complete(*, model, system, messages, max_tokens=4000):
        return "Summary of conversation"

    def _build_user_items(message):
        return [{"role": "user", "content": message}]

    provider.complete.side_effect = _complete
    provider.build_user_items.side_effect = _build_user_items
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

        summary, trimmed = ctx.summarize_and_fork()
        assert ctx.previous_response_id is None

    def test_summarize_and_fork_reduces_entries(self, ctx: ContextManager) -> None:
        for i in range(20):
            ctx.add_entry("user", f"Message {i} " * 50)
        original_count = len(ctx.entries)

        summary, trimmed = ctx.summarize_and_fork()
        assert len(ctx.entries) < original_count

    def test_summarize_and_fork_graceful_failure(self, ctx: ContextManager) -> None:
        """If API call fails, should not crash."""
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)
        ctx.provider.complete.side_effect = Exception("API down")

        # Should not raise
        summary, trimmed = ctx.summarize_and_fork()
        assert isinstance(summary, str)
        assert trimmed is None  # Failed summarization should not trim history

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


class TestHistoryTrimming:
    @pytest.fixture()
    def ctx(self, tmp_workspace: str) -> ContextManager:
        provider = _make_mock_provider()
        return ContextManager(
            provider=provider,
            model="gpt-4o",
            workspace=tmp_workspace,
        )

    def test_trim_history_produces_smaller_list(self, ctx: ContextManager) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = ctx.trim_history(history, "Summary of earlier conversation")
        assert len(trimmed) < len(history)

    def test_trim_history_starts_with_summary(self, ctx: ContextManager) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        trimmed = ctx.trim_history(history, "Summary text")
        assert "[CONTEXT SUMMARY" in trimmed[0]["content"]

    def test_summarize_and_fork_returns_trimmed_history(self, ctx: ContextManager) -> None:
        for i in range(20):
            ctx.add_entry("user", f"Message {i} " * 50)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]

        summary, trimmed = ctx.summarize_and_fork(history=history)
        assert trimmed is not None
        assert len(trimmed) < len(history)
        assert summary == "Summary of conversation"

    def test_summarize_and_fork_failure_preserves_history(self, ctx: ContextManager) -> None:
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        ctx.provider.complete.side_effect = Exception("API down")

        summary, trimmed = ctx.summarize_and_fork(history=history)
        assert trimmed is None  # History should NOT be trimmed on failure

    def test_summarize_and_fork_without_history(self, ctx: ContextManager) -> None:
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)

        summary, trimmed = ctx.summarize_and_fork()  # No history passed
        assert trimmed is None  # No history to trim
        assert summary == "Summary of conversation"

    # -- Orphaned tool-result protection (regression: Bedrock/OpenAI 400) -----

    @staticmethod
    def _bedrock_history() -> list:
        """[user(init), (assistant toolUse / user toolResult) x 9] = 19 msgs.

        For i >= 1: odd index = assistant toolUse, even index = user toolResult.
        0.6 * 19 == 11 (an assistant index here); we tune length below so the
        naive split lands ON a toolResult to exercise the back-up path.
        """
        h = [{"role": "user", "content": [{"text": "do the task"}]}]
        for i in range(1, 10):
            tid = f"tooluse_{i:03d}"
            h.append({"role": "assistant", "content": [
                {"text": f"calling tool {i}"},
                {"toolUse": {"toolUseId": tid, "name": "shell_exec", "input": {}}},
            ]})
            h.append({"role": "user", "content": [
                {"toolResult": {"toolUseId": tid, "content": [{"text": "ok"}]}},
            ]})
        return h

    @staticmethod
    def _orphan_tool_results(trimmed: list) -> list[str]:
        """Return toolResult ids in `trimmed` with no preceding toolUse."""
        seen_uses: set[str] = set()
        orphans: list[str] = []
        for msg in trimmed:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if "toolUse" in b:
                    seen_uses.add(b["toolUse"]["toolUseId"])
                elif "toolResult" in b:
                    tid = b["toolResult"]["toolUseId"]
                    if tid not in seen_uses:
                        orphans.append(tid)
        return orphans

    def test_trim_history_never_orphans_tool_result_bedrock(
        self, ctx: ContextManager
    ) -> None:
        # Make build_user_items return Bedrock-shaped blocks so the merged
        # leading message matches production format.
        ctx.provider.build_user_items.side_effect = lambda m: [
            {"role": "user", "content": [{"text": m}]}
        ]
        history = self._bedrock_history()
        # Force the naive split onto a toolResult to exercise the back-up.
        trimmed = ctx.trim_history(history, "summary")
        assert self._orphan_tool_results(trimmed) == [], (
            "trimmed history must not contain a toolResult whose toolUse was cut"
        )

    def test_trim_history_orphan_check_across_many_split_points(
        self, ctx: ContextManager
    ) -> None:
        ctx.provider.build_user_items.side_effect = lambda m: [
            {"role": "user", "content": [{"text": m}]}
        ]
        # Vary length so int(len*0.6) lands on assistant AND on toolResult.
        for n_pairs in range(2, 12):
            h = [{"role": "user", "content": [{"text": "init"}]}]
            for i in range(1, n_pairs + 1):
                tid = f"tooluse_{i:03d}"
                h.append({"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": tid, "name": "x", "input": {}}}]})
                h.append({"role": "user", "content": [
                    {"toolResult": {"toolUseId": tid, "content": []}}]})
            trimmed = ctx.trim_history(h, "summary")
            assert self._orphan_tool_results(trimmed) == [], (
                f"orphan with {n_pairs} pairs (len={len(h)}, "
                f"split={int(len(h)*0.6)})"
            )

    def test_is_tool_result_item_detects_both_formats(self) -> None:
        from alpha_lab.context import _is_tool_result_item
        # Bedrock
        assert _is_tool_result_item(
            {"role": "user", "content": [{"toolResult": {"toolUseId": "x"}}]}
        )
        # OpenAI Responses flat item
        assert _is_tool_result_item({"type": "function_call_output", "call_id": "x"})
        # Negatives
        assert not _is_tool_result_item({"role": "user", "content": [{"text": "hi"}]})
        assert not _is_tool_result_item(
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "x"}}]}
        )
        assert not _is_tool_result_item("a plain string")


class TestCompactToolOutput:
    def test_short_output_unchanged(self) -> None:
        output = "hello world"
        assert ContextManager.compact_tool_output(output, "shell_exec") == output

    def test_long_shell_output_keeps_head_and_tail(self) -> None:
        output = "A" * 5000 + "B" * 5000
        compacted = ContextManager.compact_tool_output(output, "shell_exec", max_chars=8000)
        assert len(compacted) < len(output)
        assert compacted.startswith("A")
        assert compacted.endswith("B")
        assert "trimmed" in compacted

    def test_long_read_file_output_keeps_head_and_tail(self) -> None:
        output = "X" * 20000
        compacted = ContextManager.compact_tool_output(output, "read_file", max_chars=8000)
        assert len(compacted) < len(output)
        assert "trimmed" in compacted

    def test_other_tool_truncated(self) -> None:
        output = "Z" * 20000
        compacted = ContextManager.compact_tool_output(output, "view_image", max_chars=8000)
        assert len(compacted) < len(output)
        assert "truncated" in compacted

    def test_exact_threshold(self) -> None:
        output = "A" * 8000
        assert ContextManager.compact_tool_output(output, "shell_exec", max_chars=8000) == output


class TestNonDestructiveLearnings:
    @pytest.fixture()
    def ctx(self, tmp_workspace: str) -> ContextManager:
        provider = _make_mock_provider()
        return ContextManager(
            provider=provider,
            model="gpt-4o",
            workspace=tmp_workspace,
        )

    def test_summarize_archives_original(self, ctx: ContextManager) -> None:
        # Write a large learnings file
        learnings_path = Path(ctx.workspace) / "learnings.md"
        large_content = "# Findings\n" + "Important finding. " * 5000
        learnings_path.write_text(large_content)

        ctx.get_learnings()

        # Check archive was created
        archive_dir = Path(ctx.workspace) / ".memory" / "learnings_archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("learnings_*.md"))
        assert len(archives) == 1
        assert archives[0].read_text() == large_content

    def test_archive_directory_created_lazily(self, ctx: ContextManager) -> None:
        archive_dir = Path(ctx.workspace) / ".memory" / "learnings_archive"
        assert not archive_dir.exists()

        # Small learnings should NOT trigger archiving
        learnings_path = Path(ctx.workspace) / "learnings.md"
        learnings_path.write_text("# Short findings")
        ctx.get_learnings()
        assert not archive_dir.exists()


class TestModelSelection:
    def test_summarize_uses_configured_model(self, tmp_workspace: str) -> None:
        provider = _make_mock_provider()
        ctx = ContextManager(
            provider=provider,
            model="gpt-5.4",
            workspace=tmp_workspace,
        )
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)

        ctx.summarize_and_fork()

        # Verify the model passed to complete() is the configured one
        provider.complete.assert_called_once()
        call_kwargs = provider.complete.call_args
        assert call_kwargs.kwargs["model"] == "gpt-5.4"


class TestTotalPromptTrigger:
    """Regression: the summarization trigger must see TOTAL prompt tokens.

    The agent loop passes fresh + cache_read + cache_write; a fresh-only
    number blinded the trigger for cached Anthropic histories (fresh=2
    while the true prompt hit 250k; campaign audit 2026-08-02).
    """

    def test_cache_heavy_usage_trips_trigger(self) -> None:
        from unittest.mock import MagicMock
        ctx = ContextManager(provider=MagicMock(), model="m", workspace=None)
        # Simulate the agent-loop sum for a cached history: 2 fresh +
        # 200k cache reads. update_usage receives the caller's total.
        ctx.update_usage(2 + 200_000 + 0, 50)
        assert ctx.should_summarize() is True

    def test_fresh_only_small_prompt_does_not_trip(self) -> None:
        from unittest.mock import MagicMock
        ctx = ContextManager(provider=MagicMock(), model="m", workspace=None)
        ctx.update_usage(60_000, 50)
        assert ctx.should_summarize() is False


class TestSummaryBudgetScales:
    """Summary max_tokens scales ~7:1 with the summarized text (5:1..10:1
    band per user ruling 2026-08-02), never the old flat 4,000."""

    def test_budget_grows_with_history(self) -> None:
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.complete.return_value = "summary text"
        ctx = ContextManager(provider=provider, model="m", workspace=None)
        # ~100k tokens of entries (content sized via 3.5 chars/token fallback
        # or tiktoken; either way >> floor)
        for _ in range(50):
            ctx.add_entry("tool", "x" * 8000)
        ctx.summarize_and_fork()
        budget = provider.complete.call_args.kwargs["max_tokens"]
        total = sum(e.token_count for e in ctx.entries) or 1
        assert budget > 4000, budget
        assert 6000 <= budget <= 24000, budget

    def test_budget_floor_for_small_histories(self) -> None:
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.complete.return_value = "summary text"
        ctx = ContextManager(provider=provider, model="m", workspace=None)
        for _ in range(4):
            ctx.add_entry("assistant", "short entry")
        ctx.summarize_and_fork()
        budget = provider.complete.call_args.kwargs["max_tokens"]
        assert budget == 6000, budget
