"""Tests for the meta/ filesystem layout helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.meta_layout import (
    ack_path,
    annotations_path,
    backups_dir,
    directives_path,
    ensure_meta_layout,
    from_user_path,
    instructions_dir,
    last_seen_path,
    meta_dir,
    meta_log_jsonl_path,
    meta_log_md_path,
    notes_inbox_path,
    notes_to_user_path,
    read_throttle,
    scratch_dir,
    throttle_path,
)


class TestPathHelpers:
    """All path helpers derive from workspace and return predictable shapes."""

    def test_meta_dir_under_workspace(self, tmp_path: Path) -> None:
        assert meta_dir(tmp_path) == tmp_path / "meta"

    def test_path_helpers_all_under_meta(self, tmp_path: Path) -> None:
        md = meta_dir(tmp_path)
        for path_fn in (
            directives_path,
            annotations_path,
            notes_to_user_path,
            notes_inbox_path,
            meta_log_jsonl_path,
            meta_log_md_path,
            throttle_path,
        ):
            p = path_fn(tmp_path)
            assert p.parent == md or p.parent.parent == md

    def test_instructions_subdir(self, tmp_path: Path) -> None:
        assert instructions_dir(tmp_path) == tmp_path / "meta" / "instructions"
        assert from_user_path(tmp_path).parent == instructions_dir(tmp_path)
        assert ack_path(tmp_path).parent == instructions_dir(tmp_path)
        assert last_seen_path(tmp_path).parent == instructions_dir(tmp_path)

    def test_scratch_and_backups_under_meta(self, tmp_path: Path) -> None:
        assert scratch_dir(tmp_path).parent == meta_dir(tmp_path)
        assert backups_dir(tmp_path).parent == meta_dir(tmp_path)


class TestEnsureMetaLayout:
    """Bootstrap creates the tree without overwriting existing content."""

    def test_creates_directories(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        assert meta_dir(tmp_path).is_dir()
        assert instructions_dir(tmp_path).is_dir()
        assert scratch_dir(tmp_path).is_dir()
        assert backups_dir(tmp_path).is_dir()

    def test_creates_default_files(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        for p in (
            directives_path(tmp_path),
            annotations_path(tmp_path),
            notes_to_user_path(tmp_path),
            notes_inbox_path(tmp_path),
            meta_log_jsonl_path(tmp_path),
            meta_log_md_path(tmp_path),
            from_user_path(tmp_path),
            ack_path(tmp_path),
            throttle_path(tmp_path),
        ):
            assert p.exists(), f"{p} should have been created"

    def test_returns_meta_path(self, tmp_path: Path) -> None:
        result = ensure_meta_layout(tmp_path)
        assert result == meta_dir(tmp_path)

    def test_idempotent(self, tmp_path: Path) -> None:
        # Running twice must not change anything.
        ensure_meta_layout(tmp_path)
        directives = directives_path(tmp_path)
        directives.write_text("custom content")
        ensure_meta_layout(tmp_path)
        assert directives.read_text() == "custom content"

    def test_does_not_overwrite_user_instructions(self, tmp_path: Path) -> None:
        """The user may have written instructions. Re-bootstrapping must
        never wipe them — that would silently lose user input."""
        ensure_meta_layout(tmp_path)
        from_user_path(tmp_path).write_text("please prioritize cold clients")
        ensure_meta_layout(tmp_path)
        assert "cold clients" in from_user_path(tmp_path).read_text()

    def test_does_not_overwrite_meta_log(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        log = meta_log_jsonl_path(tmp_path)
        log.write_text('{"some": "entry"}\n')
        ensure_meta_layout(tmp_path)
        assert log.read_text() == '{"some": "entry"}\n'

    def test_default_from_user_is_a_dummy_with_examples(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        content = from_user_path(tmp_path).read_text()
        assert "Conductor" in content
        # Must include at least one usage hint so the user knows what to write
        assert "#" in content  # comment-style hints
        assert "instruct" in content.lower() or "prioritize" in content.lower()

    def test_default_throttle_is_valid_json_with_none_levels(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        data = json.loads(throttle_path(tmp_path).read_text())
        assert data == {"gpu": "none", "cpu": "none"}

    def test_default_annotations_is_empty_dict(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        data = json.loads(annotations_path(tmp_path).read_text())
        assert data == {}


class TestReadThrottle:
    """The dispatcher reads throttle.json on the hot path. Must be robust."""

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        # Don't bootstrap — file doesn't exist
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}

    def test_default_throttle(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}

    def test_slow_gpu(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text('{"gpu": "slow", "cpu": "none"}')
        assert read_throttle(tmp_path) == {"gpu": "slow", "cpu": "none"}

    def test_halt_new_cpu(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text('{"gpu": "none", "cpu": "halt-new"}')
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "halt-new"}

    def test_corrupt_json_falls_back_to_default(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text("{not valid json")
        # Must not raise — the dispatcher's hot path can never crash on a
        # malformed throttle file. Default behavior == no throttling.
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}

    def test_unknown_level_coerces_to_none(self, tmp_path: Path) -> None:
        """A typo from the conductor must not accidentally halt the pipeline."""
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text('{"gpu": "stahp", "cpu": "none"}')
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}

    def test_non_dict_falls_back_to_default(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text('["not", "a", "dict"]')
        assert read_throttle(tmp_path) == {"gpu": "none", "cpu": "none"}

    def test_partial_dict_fills_missing_with_none(self, tmp_path: Path) -> None:
        ensure_meta_layout(tmp_path)
        throttle_path(tmp_path).write_text('{"gpu": "slow"}')
        assert read_throttle(tmp_path) == {"gpu": "slow", "cpu": "none"}


class TestTokenUsageSummary:
    """``meta/token_usage_summary.jsonl`` is the aggregated counterpart of
    the per-call ``token_usage.jsonl``. One JSONL line per agent role plus
    a final GRAND_TOTAL row.
    """

    def test_summary_aggregates_by_role(self, tmp_path: Path) -> None:
        from alpha_lab.meta_layout import (
            record_token_usage, refresh_token_usage_summary,
            token_usage_summary_path,
        )
        # Seed three calls from two agent roles with different providers.
        record_token_usage(tmp_path,
            log_name="strategist", provider="grok", model="Grok-4.3",
            input_tokens=100, output_tokens=10, reasoning_tokens=5,
        )
        record_token_usage(tmp_path,
            log_name="worker_worker_0_implement_x", provider="grok",
            model="Grok-4.3", input_tokens=200, output_tokens=20, reasoning_tokens=3,
        )
        record_token_usage(tmp_path,
            log_name="worker_worker_1_analyze_y", provider="grok",
            model="Grok-4.3", input_tokens=300, output_tokens=30, reasoning_tokens=7,
        )
        refresh_token_usage_summary(tmp_path)
        out = token_usage_summary_path(tmp_path)
        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        roles = {r["role"]: r for r in lines}
        assert "strategist" in roles
        assert "worker_implement" in roles
        assert "worker_analyze" in roles
        assert "GRAND_TOTAL" in roles
        # Per-role math
        assert roles["strategist"]["turns"] == 1
        assert roles["strategist"]["input_tokens"] == 100
        assert roles["worker_implement"]["input_tokens"] == 200
        assert roles["worker_analyze"]["input_tokens"] == 300
        # Grand total
        gt = roles["GRAND_TOTAL"]
        assert gt["turns"] == 3
        assert gt["input_tokens"] == 600
        assert gt["output_tokens"] == 60
        assert gt["reasoning_tokens"] == 15
        assert gt["total_tokens"] == 660

    def test_summary_idempotent_overwrites(self, tmp_path: Path) -> None:
        """Refresh rewrites the file — no append, no duplicate roles."""
        from alpha_lab.meta_layout import (
            record_token_usage, refresh_token_usage_summary,
            token_usage_summary_path,
        )
        for _ in range(5):
            record_token_usage(tmp_path,
                log_name="strategist", provider="grok", model="Grok-4.3",
                input_tokens=100, output_tokens=10,
            )
        refresh_token_usage_summary(tmp_path)
        refresh_token_usage_summary(tmp_path)
        refresh_token_usage_summary(tmp_path)
        out = token_usage_summary_path(tmp_path)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        # Exactly 2 lines: strategist + GRAND_TOTAL (no duplicates)
        assert len(lines) == 2
        roles = [l["role"] for l in lines]
        assert roles == ["strategist", "GRAND_TOTAL"]
        assert lines[0]["turns"] == 5
        assert lines[1]["turns"] == 5

    def test_summary_handles_missing_source_gracefully(self, tmp_path: Path) -> None:
        """If token_usage.jsonl doesn't exist (early in the run), refresh
        silently returns — never raises, never writes anything."""
        from alpha_lab.meta_layout import (
            refresh_token_usage_summary, token_usage_summary_path,
        )
        refresh_token_usage_summary(tmp_path)
        assert not token_usage_summary_path(tmp_path).exists()
