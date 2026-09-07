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
from alpha_lab import deps


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

    def test_load_is_read_only(self, tmp_workspace: str) -> None:
        # load_learnings only reads the file; it must not write to memory. Ingestion
        # happens once at the Phase 1 boundary (run.py), not on every prompt read.
        (Path(tmp_workspace) / "learnings.md").write_text("# Key Findings\n- Leakage risk")
        result = load_learnings(tmp_workspace)
        assert result is not None
        assert list(deps.memory_store) == []


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

    def test_should_summarize_on_api_reported_prompt_size(
        self, ctx: ContextManager
    ) -> None:
        # The local estimate only sees 500-char tool-output stubs, so it can
        # sit far below the threshold while the real prompt is huge. The
        # API-reported prompt size must trigger on its own.
        ctx.add_entry("user", "short message")
        assert ctx.should_summarize() is False
        ctx.update_usage(SUMMARIZATION_THRESHOLD + 1, 50)
        assert ctx.should_summarize() is True

    def test_api_reported_size_below_threshold_does_not_trigger(
        self, ctx: ContextManager
    ) -> None:
        ctx.update_usage(SUMMARIZATION_THRESHOLD - 1, 50)
        assert ctx.should_summarize() is False

    def test_successful_fork_clears_stale_reported_size(
        self, ctx: ContextManager
    ) -> None:
        # The pre-fork prompt size must not re-trigger summarization before
        # the next API response refreshes it.
        for i in range(4):
            ctx.add_entry("user", f"message {i}")
        ctx.update_usage(SUMMARIZATION_THRESHOLD + 1, 50)
        assert ctx.should_summarize() is True
        ctx.summarize_and_fork()
        assert ctx.last_input_tokens == 0
        assert ctx.should_summarize() is False

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

    def test_summary_budget_scales_with_history(self, ctx: ContextManager) -> None:
        """The summary budget tracks the material being summarized (~7:1
        compaction, clamped to [6_000, 24_000]) instead of a flat 4_000
        cap — a flat cap crushes a 150k-token history ~37:1."""
        seen: list[int] = []

        def capture(*, model, system, messages, max_tokens=4000):
            seen.append(max_tokens)
            return "Summary of conversation"

        ctx.provider.complete.side_effect = capture

        def run_with(entry_tokens: int) -> int:
            # 10 entries; summarize_and_fork takes the older 60% (6 entries).
            ctx.entries = [
                ConversationEntry("user", "x", token_count=entry_tokens)
                for _ in range(10)
            ]
            ctx.summary = None
            ctx.summarize_and_fork()
            return seen[-1]

        assert run_with(1_000) == 6_000    # 6k to summarize -> floor
        assert run_with(14_000) == 12_000  # 84k -> 84k // 7 inside the band
        assert run_with(35_000) == 24_000  # 210k -> capped

    def test_summary_budget_counts_prior_summary(self, ctx: ContextManager) -> None:
        """A carried-forward summary is re-summarized too, so it counts
        toward the budget's input size."""
        seen: list[int] = []

        def capture(*, model, system, messages, max_tokens=4000):
            seen.append(max_tokens)
            return "Summary of conversation"

        ctx.provider.complete.side_effect = capture
        ctx.entries = [
            ConversationEntry("user", "x", token_count=14_000)
            for _ in range(10)
        ]
        ctx.summary = "carried summary " * 1000
        ctx.summarize_and_fork()
        assert seen[-1] > 12_000  # strictly above the no-prior-summary budget

    def test_empty_summary_is_failure_not_success(self, ctx: ContextManager) -> None:
        """An empty completion must not drop entries, reset the trigger, or clobber the old summary."""
        for i in range(10):
            ctx.add_entry("user", f"Message {i} " * 100)
        ctx.summary = "previous summary"
        ctx.last_input_tokens = 999_999
        entries_before = len(ctx.entries)
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        ctx.provider.complete.side_effect = lambda **kw: ""

        summary, trimmed = ctx.summarize_and_fork(history=history)

        assert trimmed is None                       # history untouched
        assert len(ctx.entries) == entries_before    # entries kept
        assert ctx.summary == "previous summary"     # old summary preserved
        assert ctx.last_input_tokens == 999_999      # trigger not reset

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

    def test_trim_history_never_orphans_function_call_output(
        self, ctx: ContextManager
    ) -> None:
        """A cut landing between a function_call and its output must move
        past the pair: an orphaned function_call_output draws a 400 from
        the Responses API on every subsequent request (190 rejection lines
        in d6_cuda_sol_msml; fatal to d2v_sol_msml attempt 1, 2026-08-06)."""
        # 20 items; blind split lands at index 12, in the middle of the
        # call/output run (indices 10-13).
        history = [
            {"role": "user", "content": f"msg {i}"} for i in range(10)
        ]
        history += [
            {"type": "function_call", "call_id": "call_a", "name": "t"},
            {"type": "function_call", "call_id": "call_b", "name": "t"},
            {"type": "function_call_output", "call_id": "call_a", "output": "x"},
            {"type": "function_call_output", "call_id": "call_b", "output": "y"},
        ]
        history += [
            {"role": "user", "content": f"msg {i}"} for i in range(6)
        ]
        assert int(len(history) * 0.6) == 12  # cut would land mid-pair

        trimmed = ctx.trim_history(history, "Summary text")

        kept_output_ids = {
            it.get("call_id")
            for it in trimmed
            if isinstance(it, dict)
            and it.get("type") == "function_call_output"
        }
        kept_call_ids = {
            it.get("call_id")
            for it in trimmed
            if isinstance(it, dict) and it.get("type") == "function_call"
        }
        assert kept_output_ids <= kept_call_ids, (
            f"orphaned function_call_output(s): {kept_output_ids - kept_call_ids}"
        )

    def test_trim_history_plain_messages_unaffected_by_pairing_guard(
        self, ctx: ContextManager
    ) -> None:
        """Chat/bedrock-style message dicts carry no "type" key; the
        pairing guard must not change where their cut lands."""
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        trimmed = ctx.trim_history(history, "Summary text")
        assert len(trimmed) == 1 + 8  # summary item + most recent 40%

    def test_trim_history_never_orphans_chat_tool_message(
        self, ctx: ContextManager
    ) -> None:
        """Chat-format histories (glm/kimi local path) have the same hazard:
        a role-"tool" message whose assistant tool_calls message was trimmed.
        The kimi-k3 server rejects the orphan outright (probed 2026-08-06)."""
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        history += [
            {"role": "assistant", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]
        history += [{"role": "user", "content": f"msg {i}"} for i in range(6)]
        assert int(len(history) * 0.6) == 12  # cut would land on a tool msg

        trimmed = ctx.trim_history(history, "Summary text")

        kept_result_ids = {
            m.get("tool_call_id")
            for m in trimmed
            if isinstance(m, dict) and m.get("role") == "tool"
        }
        kept_call_ids = set()
        for m in trimmed:
            if isinstance(m, dict) and m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    kept_call_ids.add(c.get("id"))
        assert kept_result_ids <= kept_call_ids, (
            f"orphaned tool message(s): {kept_result_ids - kept_call_ids}"
        )

    def test_trim_history_all_tool_chain_returns_history_untouched(
        self, ctx: ContextManager
    ) -> None:
        """If no safe boundary exists, trim must refuse rather than corrupt."""
        history = [{"role": "user", "content": "start"}]
        for i in range(10):
            history += [
                {"role": "assistant", "tool_calls": [{"id": f"c{i}"}]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": "r"},
            ]
        # Every index from 1 on is a call/result item: backing up reaches
        # index 1, which is still unsafe, so trim must refuse outright.
        trimmed = ctx.trim_history(history, "Summary text")
        assert trimmed == history

    def test_trim_history_two_item_history_does_not_crash(
        self, ctx: ContextManager
    ) -> None:
        """len(history) == 2 makes the blind split equal len(history); the
        boundary walk must not read one past the end (IndexError)."""
        history = [
            {"type": "function_call", "call_id": "c1", "name": "t"},
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
        ]
        trimmed = ctx.trim_history(history, "Summary text")
        outputs = [
            it for it in trimmed
            if isinstance(it, dict) and it.get("type") == "function_call_output"
        ]
        calls = [
            it for it in trimmed
            if isinstance(it, dict) and it.get("type") == "function_call"
        ]
        assert bool(outputs) == bool(calls)  # both kept or both dropped

    def test_trim_history_never_orphans_anthropic_tool_result(
        self, ctx: ContextManager
    ) -> None:
        """Native Anthropic Messages format (providers/anthropic.py): the
        assistant's "tool_use" content block is answered by a user message
        carrying a "tool_result" block. The pairing guard must recognize
        both block types or the cut can orphan the result."""
        history = [{"role": "user", "content": f"msg {i}"} for i in range(11)]
        history += [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "shell", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r"},
            ]},
        ]
        history += [{"role": "user", "content": f"msg {i}"} for i in range(8)]
        assert int(len(history) * 0.6) == 12  # cut would land on the result

        trimmed = ctx.trim_history(history, "Summary text")

        kept_result_ids: set[str] = set()
        kept_use_ids: set[str] = set()
        for m in trimmed:
            content = m.get("content") if isinstance(m, dict) else None
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    kept_result_ids.add(b.get("tool_use_id"))
                if b.get("type") == "tool_use":
                    kept_use_ids.add(b.get("id"))
        assert kept_result_ids <= kept_use_ids, (
            f"orphaned tool_result block(s): {kept_result_ids - kept_use_ids}"
        )

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
    # Lower the threshold so these tests exercise the archive-on-overflow logic
    # without depending on which tokenizer (real tiktoken vs char fallback) is
    # active — production threshold is 20k tokens, but here any non-trivial
    # content trips it.
    @pytest.fixture(autouse=True)
    def _low_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("alpha_lab.context.LEARNINGS_SUMMARY_THRESHOLD", 10)

    @pytest.fixture()
    def ctx(self, tmp_workspace: str) -> ContextManager:
        provider = _make_mock_provider()
        return ContextManager(
            provider=provider,
            model="gpt-4o",
            workspace=tmp_workspace,
        )

    def test_summarize_archives_original(self, ctx: ContextManager) -> None:
        learnings_path = Path(ctx.workspace) / "learnings.md"
        large_content = "# Findings\n" + "Important finding. " * 50
        learnings_path.write_text(large_content)

        # Force summarization path to avoid tokenizer-dependent thresholds.
        with patch("alpha_lab.context.LEARNINGS_SUMMARY_THRESHOLD", 1):
            ctx.get_learnings()

        archive_dir = Path(ctx.workspace) / ".alpha_lab" / "memory" / "learnings_archive"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("learnings_*.md"))
        assert len(archives) == 1
        assert archives[0].read_text() == large_content

    def test_archive_directory_created_lazily(
        self, ctx: ContextManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Restore production threshold for this test — we want to verify that
        # SHORT content does NOT trigger archiving.
        monkeypatch.setattr("alpha_lab.context.LEARNINGS_SUMMARY_THRESHOLD", 20_000)
        archive_dir = Path(ctx.workspace) / ".alpha_lab" / "memory" / "learnings_archive"
        assert not archive_dir.exists()

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
