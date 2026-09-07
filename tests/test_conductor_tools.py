"""Tests for the Conductor's helper functions (conductor_tools.py).

These cover the file-mutation primitives: meta_log entry building and
appending, annotations JSON merge, directive prepending, throttle state
updates, backup-with-restore safety, and bounded read helpers
(read_meta_log, read_experiment_summary, peek_experiment_log,
read_system_load).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from alpha_lab import conductor_tools as ct
from alpha_lab import meta_layout as ml
from alpha_lab.experiment_db import ExperimentDB


# ---------------------------------------------------------------------------
# meta_log
# ---------------------------------------------------------------------------


class TestMetaLogEntry:
    def test_basic_fields(self) -> None:
        e = ct.meta_log_entry("park", target=42, reason="duplicate", evidence="see #41")
        assert e["decision_type"] == "park"
        assert e["target"] == 42
        assert e["reason"] == "duplicate"
        assert e["evidence"] == "see #41"
        assert e["prior_decision_id"] is None
        assert "ts" in e

    def test_truncates_oversize_reason_and_evidence(self) -> None:
        long_reason = "x" * (ct.MAX_REASON_CHARS + 1000)
        long_evidence = "y" * (ct.MAX_EVIDENCE_CHARS + 1000)
        e = ct.meta_log_entry("park", target=1, reason=long_reason, evidence=long_evidence)
        assert len(e["reason"]) == ct.MAX_REASON_CHARS
        assert len(e["evidence"]) == ct.MAX_EVIDENCE_CHARS

    def test_unrecognized_decision_type_records_anyway(self, caplog) -> None:
        # Don't reject — meta_log is append-only and we'd rather have a
        # weird entry than a silent drop.
        e = ct.meta_log_entry("frobnicate", reason="r")
        assert e["decision_type"] == "frobnicate"


class TestMetaLogAppendAndRead:
    def test_append_and_read_roundtrip(self, tmp_path: Path) -> None:
        for i in range(3):
            ct.meta_log_append(
                tmp_path,
                ct.meta_log_entry("park", target=i, reason=f"r{i}"),
            )
        entries = ct.meta_log_read(tmp_path, last_n=10)
        assert [e["target"] for e in entries] == [0, 1, 2]

    def test_read_last_n(self, tmp_path: Path) -> None:
        for i in range(20):
            ct.meta_log_append(
                tmp_path,
                ct.meta_log_entry("note_to_user", target=i, reason="x"),
            )
        last5 = ct.meta_log_read(tmp_path, last_n=5)
        assert len(last5) == 5
        assert [e["target"] for e in last5] == [15, 16, 17, 18, 19]

    def test_read_with_sample_older(self, tmp_path: Path) -> None:
        for i in range(50):
            ct.meta_log_append(
                tmp_path,
                ct.meta_log_entry("park", target=i, reason="r"),
            )
        # last_n=10 + sample_older=True should give 10 + ≤10 stratified
        result = ct.meta_log_read(tmp_path, last_n=10, sample_older=True)
        assert len(result) > 10
        assert len(result) <= 20
        # Sampled entries have the marker, recent ones don't
        sampled = [e for e in result if e.get("_sampled")]
        recent = [e for e in result if not e.get("_sampled")]
        assert len(sampled) > 0
        assert len(recent) == 10
        # Recent should be the actual last 10
        assert [e["target"] for e in recent] == list(range(40, 50))

    def test_read_clamps_last_n(self, tmp_path: Path) -> None:
        for i in range(5):
            ct.meta_log_append(
                tmp_path,
                ct.meta_log_entry("park", target=i, reason="r"),
            )
        # Pass a huge last_n; should clamp without erroring
        result = ct.meta_log_read(tmp_path, last_n=10**9)
        assert len(result) == 5

    def test_read_missing_file_returns_empty(self, tmp_path: Path) -> None:
        # Don't bootstrap; file doesn't exist
        assert ct.meta_log_read(tmp_path) == []

    def test_read_skips_malformed_lines(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        path = ml.meta_log_jsonl_path(tmp_path)
        path.write_text(
            json.dumps({"decision_type": "park", "target": 1, "ts": 1.0, "reason": "ok"}) + "\n"
            + "{not valid json\n"
            + json.dumps({"decision_type": "park", "target": 2, "ts": 2.0, "reason": "ok"}) + "\n"
        )
        result = ct.meta_log_read(tmp_path, last_n=10)
        assert [e["target"] for e in result] == [1, 2]


class TestMetaLogRenderMd:
    def test_renders_with_recent_first(self, tmp_path: Path) -> None:
        for i in range(3):
            ct.meta_log_append(
                tmp_path,
                ct.meta_log_entry("park", target=i, reason=f"reason_{i}"),
            )
        ct.meta_log_render_md(tmp_path)
        md = ml.meta_log_md_path(tmp_path).read_text()
        assert "Conductor decision log" in md
        # Most recent (i=2) should appear before i=0 in the rendered doc
        idx_2 = md.find("reason_2")
        idx_0 = md.find("reason_0")
        assert idx_2 != -1 and idx_0 != -1
        assert idx_2 < idx_0

    def test_truncates_long_reasons_in_digest(self, tmp_path: Path) -> None:
        ct.meta_log_append(
            tmp_path,
            ct.meta_log_entry("park", target=1, reason="X" * 5000),
        )
        ct.meta_log_render_md(tmp_path)
        md = ml.meta_log_md_path(tmp_path).read_text()
        # The full 5000-char reason must NOT appear verbatim in the digest.
        assert "X" * 5000 not in md
        assert "..." in md  # truncation marker


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_round_trip(self, tmp_path: Path) -> None:
        ct.set_annotation(tmp_path, 42, "champion")
        ct.set_annotation(tmp_path, 43, "control")
        ann = ct.read_annotations(tmp_path)
        assert ann == {"42": "champion", "43": "control"}

    def test_set_overwrites_existing(self, tmp_path: Path) -> None:
        ct.set_annotation(tmp_path, 1, "exploration")
        ct.set_annotation(tmp_path, 1, "champion")
        ann = ct.read_annotations(tmp_path)
        assert ann == {"1": "champion"}

    def test_clear(self, tmp_path: Path) -> None:
        ct.set_annotation(tmp_path, 1, "champion")
        ct.set_annotation(tmp_path, 2, "control")
        ct.clear_annotation(tmp_path, 1)
        ann = ct.read_annotations(tmp_path)
        assert ann == {"2": "control"}

    def test_corrupt_annotations_returns_empty_dict(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        ml.annotations_path(tmp_path).write_text("{bad json")
        assert ct.read_annotations(tmp_path) == {}


# ---------------------------------------------------------------------------
# Directives & notes
# ---------------------------------------------------------------------------


class TestDirectives:
    def test_append_creates_block(self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "strategist", "diversify into new families")
        text = ml.directives_path(tmp_path).read_text()
        assert "strategist" in text
        assert "diversify into new families" in text

    def test_unknown_role_coerces_to_all(self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "captain_planet", "save the world")
        text = ml.directives_path(tmp_path).read_text()
        assert "all" in text

    def test_most_recent_first(self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "strategist", "FIRST_DIRECTIVE")
        time.sleep(1.1)  # ensure different second-precision timestamps
        ct.append_directive(tmp_path, "strategist", "SECOND_DIRECTIVE")
        text = ml.directives_path(tmp_path).read_text()
        # Both present
        assert "FIRST_DIRECTIVE" in text
        assert "SECOND_DIRECTIVE" in text
        # Most recent is closer to the top
        assert text.find("SECOND_DIRECTIVE") < text.find("FIRST_DIRECTIVE")


class TestDirectiveScopeAndAck:
    def test_append_directive_returns_id(self, tmp_path: Path) -> None:
        did = ct.append_directive(tmp_path, "strategist", "do thing", scope="one-shot")
        assert did.startswith("d-")
        text = ml.directives_path(tmp_path).read_text()
        assert did in text
        assert "scope=one-shot" in text

    def test_invalid_scope_defaults_to_standing(self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "strategist", "x", scope="garbage")
        text = ml.directives_path(tmp_path).read_text()
        assert "scope=standing" in text

    def test_per_experiment_scope_accepted(self, tmp_path: Path) -> None:
        ct.append_directive(
            tmp_path, "worker", "rerun 178 with seed=42", scope="per-experiment:178",
        )
        text = ml.directives_path(tmp_path).read_text()
        assert "scope=per-experiment:178" in text

    def test_parse_round_trips(self, tmp_path: Path) -> None:
        d1 = ct.append_directive(tmp_path, "strategist", "FIRST", scope="standing")
        time.sleep(1.1)
        d2 = ct.append_directive(tmp_path, "worker", "SECOND", scope="one-shot")
        parsed = ct.parse_directives(tmp_path)
        ids = [d["id"] for d in parsed]
        assert d1 in ids and d2 in ids
        by_id = {d["id"]: d for d in parsed}
        assert by_id[d1]["scope"] == "standing"
        assert by_id[d2]["scope"] == "one-shot"
        assert "FIRST" in by_id[d1]["body"]
        assert "SECOND" in by_id[d2]["body"]

    def test_role_filtering(self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "strategist", "S")
        ct.append_directive(tmp_path, "worker", "W")
        ct.append_directive(tmp_path, "all", "A")
        out_s = ct.directives_for_role(tmp_path, "strategist")
        out_w = ct.directives_for_role(tmp_path, "worker")
        roles_s = {d["role"] for d in out_s}
        roles_w = {d["role"] for d in out_w}
        assert "strategist" in roles_s and "all" in roles_s and "worker" not in roles_s
        assert "worker" in roles_w and "all" in roles_w and "strategist" not in roles_w

    def test_one_shot_filtered_after_ack(self, tmp_path: Path) -> None:
        did = ct.append_directive(
            tmp_path, "strategist", "propose 3", scope="one-shot",
        )
        active_before = ct.directives_for_role(tmp_path, "strategist")
        assert any(d["id"] == did for d in active_before)
        ct.append_directive_ack(
            tmp_path,
            directive_id=did,
            actor_role="strategist",
            actor_id="strategist_a",
            action="proposed 3 new experiments",
        )
        active_after = ct.directives_for_role(tmp_path, "strategist")
        assert not any(d["id"] == did for d in active_after)

    def test_one_shot_visible_to_other_role(self, tmp_path: Path) -> None:
        did = ct.append_directive(
            tmp_path, "all", "do once", scope="one-shot",
        )
        # strategist acks
        ct.append_directive_ack(
            tmp_path, directive_id=did, actor_role="strategist",
            actor_id="strategist_a", action="done",
        )
        # worker should still see it (different role's ack does not claim it)
        active_w = ct.directives_for_role(tmp_path, "worker")
        assert any(d["id"] == did for d in active_w)
        # second strategist turn should NOT see it
        active_s = ct.directives_for_role(tmp_path, "strategist")
        assert not any(d["id"] == did for d in active_s)

    def test_per_experiment_only_matches_id(self, tmp_path: Path) -> None:
        did = ct.append_directive(
            tmp_path, "worker", "rerun #178", scope="per-experiment:178",
        )
        # Worker on a different experiment doesn't see it
        active_other = ct.directives_for_role(tmp_path, "worker", experiment_id=999)
        assert not any(d["id"] == did for d in active_other)
        # Worker without an experiment id doesn't see it either
        active_none = ct.directives_for_role(tmp_path, "worker")
        assert not any(d["id"] == did for d in active_none)
        # Worker on the matching experiment sees it
        active_match = ct.directives_for_role(tmp_path, "worker", experiment_id=178)
        assert any(d["id"] == did for d in active_match)

    def test_standing_always_visible(self, tmp_path: Path) -> None:
        did = ct.append_directive(
            tmp_path, "strategist", "always include cold slice", scope="standing",
        )
        # Even if "acked" (should be a no-op for standing), still visible
        ct.append_directive_ack(
            tmp_path, directive_id=did, actor_role="strategist",
            actor_id="strategist_a", action="noted",
        )
        active = ct.directives_for_role(tmp_path, "strategist")
        assert any(d["id"] == did for d in active)

    def test_read_directive_acks(self, tmp_path: Path) -> None:
        ct.append_directive_ack(
            tmp_path, directive_id="d-1-a", actor_role="worker",
            actor_id="worker_2", action="ran experiment",
        )
        acks = ct.read_directive_acks(tmp_path)
        assert len(acks) == 1
        assert acks[0]["directive_id"] == "d-1-a"
        assert acks[0]["actor_id"] == "worker_2"

    def test_render_for_prompt_contains_scope_help(self, tmp_path: Path) -> None:
        did = ct.append_directive(
            tmp_path, "strategist", "propose 3", scope="one-shot",
        )
        active = ct.directives_for_role(tmp_path, "strategist")
        rendered = ct.render_directives_for_prompt(active, acks=[])
        assert did in rendered
        assert "ack_directive" in rendered
        assert "one-shot" in rendered


class TestDirectiveUptakeEscalation:
    """Zero-ack runs lost every audited pair (2026-08-08, 9-run audit):
    after N consecutive zero-ack turns with active directives, the harness
    escalates with a mandatory banner instead of failing silently."""

    def test_escalates_after_threshold_and_resets_on_ack(
            self, tmp_path: Path) -> None:
        ct.append_directive(tmp_path, "strategist", "diversify", scope="standing")
        active = ct.directives_for_role(tmp_path, "strategist")
        assert active
        for _ in range(ct.ESCALATE_AFTER_ZERO_ACK_TURNS - 1):
            assert ct.directive_uptake_escalation(tmp_path, active, []) is None
        banner = ct.directive_uptake_escalation(tmp_path, active, [])
        assert banner and "ESCALATION" in banner
        # an ack resets the counter
        assert ct.directive_uptake_escalation(
            tmp_path, active, [{"directive_id": "d-1"}]) is None
        assert ct.directive_uptake_escalation(tmp_path, active, []) is None

    def test_no_active_directives_never_escalates(self, tmp_path: Path) -> None:
        for _ in range(ct.ESCALATE_AFTER_ZERO_ACK_TURNS + 2):
            assert ct.directive_uptake_escalation(tmp_path, [], []) is None


class TestDirectiveRetirement:
    """The Conductor owns the directive lifecycle: directives never expire
    on their own. ``retire_directive`` is the explicit mechanism for
    taking a directive out of force."""

    def test_retire_filters_directive_from_role_view(self, tmp_path: Path) -> None:
        d1 = ct.append_directive(tmp_path, "worker", "use aligned future_ret")
        d2 = ct.append_directive(tmp_path, "worker", "retry seed 42", scope="one-shot")
        # Both visible before retirement
        active = ct.directives_for_role(tmp_path, "worker")
        assert {d["id"] for d in active} == {d1, d2}
        # Retire one
        ct.append_directive_retirement(tmp_path, d1, "phase moved on")
        active = ct.directives_for_role(tmp_path, "worker")
        assert {d["id"] for d in active} == {d2}

    def test_retired_directive_ids_dedupes(self, tmp_path: Path) -> None:
        d1 = ct.append_directive(tmp_path, "worker", "x")
        ct.append_directive_retirement(tmp_path, d1, "no longer applicable")
        ct.append_directive_retirement(tmp_path, d1, "still not applicable")
        assert ct.retired_directive_ids(tmp_path) == {d1}

    def test_retirement_is_idempotent_per_filter(self, tmp_path: Path) -> None:
        d1 = ct.append_directive(tmp_path, "all", "Phase 1 mid-run note")
        ct.append_directive_retirement(tmp_path, d1, "Phase 1 over")
        # A different role should also not see it
        assert ct.directives_for_role(tmp_path, "worker") == []
        assert ct.directives_for_role(tmp_path, "strategist") == []


class TestFromUserBoilerplateStrip:
    """The from_user.md file is bootstrapped with a `#`-commented help
    block. If we pass it raw to the Conductor it treats the boilerplate
    as a real user instruction. ``_strip_from_user_boilerplate`` removes
    just the boilerplate while preserving any real user content,
    including user-written `#`-prefixed lines (markdown headers etc.)."""

    def test_default_returns_empty(self) -> None:
        from alpha_lab.conductor import _strip_from_user_boilerplate
        from alpha_lab.meta_layout import DEFAULT_FROM_USER
        assert _strip_from_user_boilerplate(DEFAULT_FROM_USER) == ""

    def test_real_user_text_survives_below_boilerplate(self) -> None:
        from alpha_lab.conductor import _strip_from_user_boilerplate
        from alpha_lab.meta_layout import DEFAULT_FROM_USER
        text = (
            DEFAULT_FROM_USER
            + "\nPlease prioritize cold-client experiments.\n"
            "Skip won-only filtering for total flow targets.\n"
        )
        out = _strip_from_user_boilerplate(text)
        assert out.startswith("Please prioritize cold-client experiments")
        assert "won-only filtering" in out
        # The boilerplate header phrase should be gone
        assert "Instructions from user to Conductor" not in out

    def test_user_markdown_headers_preserved(self) -> None:
        """A user who replaces the boilerplate with their own free text
        (including markdown `#` headers) must see all of it preserved."""
        from alpha_lab.conductor import _strip_from_user_boilerplate
        text = (
            "# Goal\n"
            "Prioritize cold-client experiments.\n"
            "\n"
            "# Constraints\n"
            "- No Sharpe > 5 as champion threshold\n"
        )
        out = _strip_from_user_boilerplate(text)
        assert "# Goal" in out
        assert "# Constraints" in out
        assert "Prioritize cold-client experiments." in out

    def test_handles_empty_and_whitespace(self) -> None:
        from alpha_lab.conductor import _strip_from_user_boilerplate
        assert _strip_from_user_boilerplate("") == ""
        assert _strip_from_user_boilerplate("   \n  ") == ""


class TestNotesToUser:
    def test_appends_to_notes_to_user(self, tmp_path: Path) -> None:
        ct.append_note_to_user(tmp_path, "Plateau observed at 0.183")
        text = ml.notes_to_user_path(tmp_path).read_text()
        assert "Plateau observed at 0.183" in text


class TestNotesInbox:
    def test_strategist_note_appears(self, tmp_path: Path) -> None:
        ct.append_note_to_conductor(tmp_path, "strategist", "Don't preempt #181")
        inbox = ct.read_notes_inbox(tmp_path)
        assert "strategist" in inbox
        assert "Don't preempt #181" in inbox

    def test_inbox_size_bounded(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        # Write a huge inbox
        inbox = ml.notes_inbox_path(tmp_path)
        inbox.write_text("x" * (ct.MAX_DIGEST_CHARS * 2))
        result = ct.read_notes_inbox(tmp_path)
        assert len(result) <= ct.MAX_DIGEST_CHARS


# ---------------------------------------------------------------------------
# User instructions
# ---------------------------------------------------------------------------


class TestUserInstructions:
    def test_initial_dummy_is_not_new(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        # Conductor hasn't seen anything yet; the dummy IS new on first read
        current, is_new = ct.read_from_user_diff(tmp_path)
        assert is_new is True
        ct.mark_user_instructions_seen(tmp_path)
        # After marking seen, no diff
        _, is_new = ct.read_from_user_diff(tmp_path)
        assert is_new is False

    def test_user_writes_new_content_triggers_diff(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        ct.mark_user_instructions_seen(tmp_path)
        # User writes new content
        ml.from_user_path(tmp_path).write_text(
            "please prioritize cold-client experiments"
        )
        current, is_new = ct.read_from_user_diff(tmp_path)
        assert is_new is True
        assert "cold-client" in current

    def test_ack_appends(self, tmp_path: Path) -> None:
        ct.append_ack(tmp_path, "Read user instruction; translated to directive.")
        text = ml.ack_path(tmp_path).read_text()
        assert "translated to directive" in text


# ---------------------------------------------------------------------------
# Throttle state
# ---------------------------------------------------------------------------


class TestThrottle:
    def test_set_gpu_only_preserves_cpu(self, tmp_path: Path) -> None:
        ct.set_throttle_state(tmp_path, gpu="slow")
        result = ml.read_throttle(tmp_path)
        assert result == {"gpu": "slow", "cpu": "none"}

    def test_set_both(self, tmp_path: Path) -> None:
        ct.set_throttle_state(tmp_path, gpu="halt-new", cpu="slow")
        result = ml.read_throttle(tmp_path)
        assert result == {"gpu": "halt-new", "cpu": "slow"}

    def test_invalid_level_coerces_to_none(self, tmp_path: Path) -> None:
        ct.set_throttle_state(tmp_path, gpu="banana")
        result = ml.read_throttle(tmp_path)
        assert result["gpu"] == "none"


# ---------------------------------------------------------------------------
# Backups & deletes (the safety-critical tools)
# ---------------------------------------------------------------------------


class TestBackupAndDelete:
    def test_backup_copies_file(self, tmp_path: Path) -> None:
        # Create a file in the workspace
        target = tmp_path / "stale_cache.parquet"
        target.write_text("cache contents")
        backup = ct.backup_workspace_path(tmp_path, "stale_cache.parquet")
        assert backup.exists()
        assert backup.read_text() == "cache contents"
        # Source still exists
        assert target.exists()

    def test_backup_copies_directory(self, tmp_path: Path) -> None:
        srcdir = tmp_path / "stale_dir"
        srcdir.mkdir()
        (srcdir / "a.txt").write_text("a")
        (srcdir / "b.txt").write_text("b")
        backup = ct.backup_workspace_path(tmp_path, "stale_dir")
        assert backup.is_dir()
        assert (backup / "a.txt").read_text() == "a"
        # Source still exists
        assert srcdir.exists()

    def test_delete_with_backup_makes_a_backup_first(self, tmp_path: Path) -> None:
        target = tmp_path / "stale.parquet"
        target.write_text("delete me")
        backup = ct.safe_delete_with_backup(tmp_path, "stale.parquet")
        # Source is gone
        assert not target.exists()
        # Backup contains the contents
        assert backup.exists()
        assert backup.read_text() == "delete me"

    def test_refuses_to_delete_meta_dir(self, tmp_path: Path) -> None:
        ml.ensure_meta_layout(tmp_path)
        with pytest.raises(PermissionError, match="protected"):
            ct.safe_delete_with_backup(tmp_path, "meta")

    def test_refuses_to_delete_db(self, tmp_path: Path) -> None:
        (tmp_path / "experiments.db").write_text("fake")
        with pytest.raises(PermissionError, match="protected"):
            ct.safe_delete_with_backup(tmp_path, "experiments.db")

    def test_refuses_to_delete_adapter(self, tmp_path: Path) -> None:
        (tmp_path / "adapter").mkdir()
        with pytest.raises(PermissionError, match="protected"):
            ct.safe_delete_with_backup(tmp_path, "adapter")

    def test_refuses_absolute_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            ct.backup_workspace_path(tmp_path, "/etc/passwd")

    def test_refuses_path_traversal(self, tmp_path: Path) -> None:
        # Create a file outside the workspace
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("you can't have me")
        try:
            with pytest.raises(ValueError, match="escapes workspace"):
                ct.backup_workspace_path(tmp_path, "../outside.txt")
        finally:
            outside.unlink()

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ct.backup_workspace_path(tmp_path, "nonexistent.txt")


# ---------------------------------------------------------------------------
# Phase rewind
# ---------------------------------------------------------------------------


class TestPhaseRewind:
    def test_writes_marker_file(self, tmp_path: Path) -> None:
        # Bootstrap a minimal workspace with an adapter
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "manifest.json").write_text("{}")
        ct.request_phase_rewind(tmp_path, "phase0", "wrong domain", "exhibit A")
        marker = ml.meta_dir(tmp_path) / ct.PHASE_REWIND_MARKER
        assert marker.exists()
        payload = json.loads(marker.read_text())
        assert payload["target_phase"] == "phase0"
        assert "wrong domain" in payload["reason"]

    def test_backs_up_adapter(self, tmp_path: Path) -> None:
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "manifest.json").write_text('{"v":1}')
        result = ct.request_phase_rewind(tmp_path, "phase0", "r", "e")
        backup_dir = Path(result["backup_dir"])
        assert (backup_dir / "adapter" / "manifest.json").exists()

    def test_phase2_rewind_backs_up_adapter_framework_dir(self, tmp_path: Path) -> None:
        """The Phase 2 framework dir name comes from the adapter — caller
        must pass it. Without an adapter, only universally-named
        artifacts (adapter/, learnings.md, etc.) get backed up."""
        from unittest.mock import MagicMock
        (tmp_path / "adapter").mkdir()
        (tmp_path / "harness").mkdir()
        (tmp_path / "harness" / "metrics.py").write_text("# metrics")
        adapter = MagicMock()
        adapter.experiment.framework_dir = "harness"
        result = ct.request_phase_rewind(tmp_path, "phase2", "r", "e", adapter=adapter)
        backup_dir = Path(result["backup_dir"])
        assert (backup_dir / "harness" / "metrics.py").exists()

    def test_phase2_rewind_without_adapter_skips_framework_dir(self, tmp_path: Path) -> None:
        """Without an adapter the framework_dir name is unknown — the
        rewind succeeds but the framework backup is skipped (the
        caller is responsible for passing adapter when phase 2 rewind
        matters)."""
        (tmp_path / "adapter").mkdir()
        (tmp_path / "harness").mkdir()
        (tmp_path / "harness" / "metrics.py").write_text("# metrics")
        result = ct.request_phase_rewind(tmp_path, "phase2", "r", "e")
        backup_dir = Path(result["backup_dir"])
        # adapter backup still happens (universal name)
        assert (backup_dir / "adapter").exists()
        # but harness/ backup does not — no adapter to resolve it from
        assert not (backup_dir / "harness").exists()

    def test_invalid_target_phase_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="phase0/1/2"):
            ct.request_phase_rewind(tmp_path, "phase3", "r", "e")


# ---------------------------------------------------------------------------
# read_system_load
# ---------------------------------------------------------------------------


class TestReadSystemLoad:
    def test_returns_string(self, tmp_path: Path) -> None:
        result = ct.read_system_load(tmp_path)
        assert isinstance(result, str)
        # Always includes throttle state
        assert "throttle" in result

    def test_does_not_crash_on_missing_nvidia(self, tmp_path: Path) -> None:
        # On a host without nvidia-smi, the function still returns a sensible
        # string and doesn't raise.
        result = ct.read_system_load(tmp_path)
        assert "cpu_load" in result or "cpu_load unavailable" in result


# ---------------------------------------------------------------------------
# peek_experiment_log
# ---------------------------------------------------------------------------


class TestPeekExperimentLog:
    def test_reads_local_job_out(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "local_job.out").write_text("\n".join([f"line {i}" for i in range(20)]))
        result = ct.peek_experiment_log(tmp_path, "my_exp", last_n_lines=5)
        assert "line 19" in result
        assert "line 14" not in result.split("\n")[0]  # should be near the end

    def test_returns_no_log_message_when_missing(self, tmp_path: Path) -> None:
        result = ct.peek_experiment_log(tmp_path, "nonexistent_exp")
        assert "no log" in result.lower()

    def test_falls_back_to_other_log_files(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "exp_a"
        exp_dir.mkdir(parents=True)
        # No local_job.out, but a generic .log
        (exp_dir / "training.log").write_text("epoch 1\nepoch 2\nepoch 3\n")
        result = ct.peek_experiment_log(tmp_path, "exp_a", last_n_lines=5)
        assert "epoch" in result


# ---------------------------------------------------------------------------
# read_experiment_summary (bounded summary)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_one(tmp_path: Path) -> tuple[ExperimentDB, int]:
    db = ExperimentDB(str(tmp_path / "db.sqlite"))
    eid = db.create(
        "test_exp",
        "A long description " * 50,
        "Hypothesis: " + "x" * 1000,
        json.dumps({"model_type": "lstm", "epochs": 50, "features": list(range(20))}),
    )
    db.set_results(eid, json.dumps({"sharpe": 1.5, "mae": 0.02, "extras": {"k": "v"}}))
    return db, eid


class TestReadExperimentSummary:
    def test_truncates_description(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        s = ct.read_experiment_summary(tmp_path, db, eid)
        assert len(s["description"]) <= 800

    def test_truncates_hypothesis(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        s = ct.read_experiment_summary(tmp_path, db, eid)
        assert len(s["hypothesis"]) <= 800

    def test_includes_paths_for_followup_reads(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        s = ct.read_experiment_summary(tmp_path, db, eid)
        paths = s["paths"]
        assert "experiment_dir" in paths
        assert "run_experiment_py" in paths
        assert "metrics_json" in paths

    def test_does_not_include_full_results_json(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        s = ct.read_experiment_summary(tmp_path, db, eid)
        # metrics_summary holds at most 8 scalar keys; the nested dict 'extras'
        # must not have leaked in unbounded form.
        assert isinstance(s["metrics_summary"], dict)
        # We DON'T expect the full nested structure preserved verbatim
        # — read_file is for that.
        assert "sharpe" in s["metrics_summary"]

    def test_returns_error_for_missing_id(self, tmp_path: Path) -> None:
        db = ExperimentDB(str(tmp_path / "db.sqlite"))
        s = ct.read_experiment_summary(tmp_path, db, 9999)
        assert "error" in s

    def test_includes_annotation(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        ct.set_annotation(tmp_path, eid, "champion")
        s = ct.read_experiment_summary(tmp_path, db, eid)
        assert s["annotation"] == "champion"

    def test_reflects_parked_state(self, db_with_one, tmp_path: Path) -> None:
        db, eid = db_with_one
        db.park(eid)
        s = ct.read_experiment_summary(tmp_path, db, eid)
        assert s["parked_at"] is not None
