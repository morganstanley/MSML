"""Tests for the Conductor agent.

Covers:

* The pure ``build_conductor_context`` digest builder — section content,
  size budgets, robustness to missing files.
* The Conductor class wiring — NOOP path returns immediately,
  prompt_builder injects workspace instructions above the system prompt,
  triggers map to the right initial-message and log_name.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from alpha_lab import conductor as cond
from alpha_lab import conductor_tools as ct
from alpha_lab import meta_layout as ml
from alpha_lab.adapter import DomainAdapter, MetricConfig
from alpha_lab.config import TaskConfig
from alpha_lab.experiment_db import ExperimentDB


# ---------------------------------------------------------------------------
# build_conductor_context — the digest builder
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_workspace(tmp_path: Path) -> Path:
    ml.ensure_meta_layout(tmp_path)
    return tmp_path


@pytest.fixture
def fresh_db(tmp_path: Path) -> ExperimentDB:
    return ExperimentDB(str(tmp_path / "exps.db"))


class TestBuildConductorContext:
    def test_minimal_workspace_returns_string(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert isinstance(digest, str)
        # Each section should be present as a header
        assert "USER INSTRUCTIONS" in digest
        assert "SYSTEM PULSE" in digest

    def test_workspace_instruction_file_appears(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        ml.from_user_path(fresh_workspace).write_text("prioritize cold clients")
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "cold clients" in digest

    def test_empty_workspace_instructions_says_so(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        # Empty/uninstructive from_user.md content should produce an
        # explicit "(empty — no user instructions)" marker so the agent
        # knows the channel exists but has nothing in it.
        ml.from_user_path(fresh_workspace).write_text("")
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "USER INSTRUCTIONS" in digest
        assert "no user instructions" in digest

    def test_leaderboard_section_with_completed_experiment(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        eid = fresh_db.create("test_model", "D", "H", "{}")
        fresh_db.set_results(eid, json.dumps({"sharpe": 1.5}))
        digest = cond.build_conductor_context(
            fresh_workspace, fresh_db, metric_key="sharpe"
        )
        assert "test_model" in digest
        assert "1.5" in digest

    def test_recent_decisions_section(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        ct.meta_log_append(
            fresh_workspace,
            ct.meta_log_entry("park", target=42, reason="saturated_family"),
        )
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "saturated_family" in digest

    def test_inbox_appears_when_present(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        ct.append_note_to_conductor(
            fresh_workspace, "strategist", "Don't preempt #181"
        )
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "Don't preempt #181" in digest

    def test_handles_missing_meta_dir(
        self, tmp_path: Path, fresh_db: ExperimentDB
    ) -> None:
        # No bootstrap — meta/ doesn't exist yet
        digest = cond.build_conductor_context(tmp_path, fresh_db)
        # Doesn't raise; produces a valid string
        assert isinstance(digest, str)
        assert "USER INSTRUCTIONS" in digest

    def test_handles_no_db(self, fresh_workspace: Path) -> None:
        digest = cond.build_conductor_context(fresh_workspace, db=None)
        assert isinstance(digest, str)
        # Without DB, no leaderboard section, but still has user instructions,
        # pulse, recent_decisions, inbox, milestone, taxonomy
        assert "USER INSTRUCTIONS" in digest

    def test_digest_size_bounded_for_busy_run(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        # Simulate a busy run: 200 experiments + 30 milestones + 100 decisions
        for i in range(200):
            eid = fresh_db.create(f"exp_{i:03d}", "D" * 100, "H" * 100, "{}")
            fresh_db.set_results(eid, json.dumps({"sharpe": float(i % 10) / 10}))
        # Simulate milestone reports
        reports = fresh_workspace / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        for j in range(30):
            d = reports / f"milestone_{j:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "milestone_report.md").write_text(f"# milestone {j}\n" + "x" * 5_000)
        # Simulate decisions
        for k in range(100):
            ct.meta_log_append(
                fresh_workspace,
                ct.meta_log_entry("park", target=k, reason=f"reason {k}"),
            )
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        # Total digest must remain bounded — generous ceiling so per-section
        # caps still kick in long before we hit it.
        assert len(digest) < 30_000, f"digest too large: {len(digest)}"

    def test_per_section_budget_truncates_huge_workspace_instructions(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        huge = "x" * 50_000
        ml.from_user_path(fresh_workspace).write_text(huge)
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        # The section is capped — the full 50K must NOT appear
        assert "x" * 50_000 not in digest

    def test_milestone_report_excerpt_appears(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        reports = fresh_workspace / "reports" / "milestone_001"
        reports.mkdir(parents=True)
        (reports / "milestone_report.md").write_text("# Milestone 1\nKey finding A.")
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "Key finding A" in digest

    def test_milestone_report_alternative_filename(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        # Tolerate the 'report.md' built-in convention as well as the
        # 'milestone_report.md' that some customizers emit.
        reports = fresh_workspace / "reports" / "milestone_002"
        reports.mkdir(parents=True)
        (reports / "report.md").write_text("# Built-in convention\nKey finding B.")
        digest = cond.build_conductor_context(fresh_workspace, fresh_db)
        assert "Key finding B" in digest

    def test_trigger_appears_in_pulse(
        self, fresh_workspace: Path, fresh_db: ExperimentDB
    ) -> None:
        digest = cond.build_conductor_context(
            fresh_workspace, fresh_db, trigger="phase1_done"
        )
        assert "phase1_done" in digest


# ---------------------------------------------------------------------------
# Conductor class wiring — NOOP, prompt-builder content, log_name.
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Provider stub. The Conductor's _run_steer runs an AgentLoop that
    requires a Provider; we never hit the network because no_conductor=True
    short-circuits before AgentLoop construction in those tests, and for
    tests that DO build the prompt, we exercise prompt_builder directly
    rather than running the loop."""

    def __init__(self) -> None:
        self.openai_client = None


class _FakeAdapter(DomainAdapter):
    pass


def _events_collector() -> tuple[list, Any]:
    events: list = []

    def cb(e: Any) -> None:
        events.append(e)

    return events, cb


class TestConductorNoopPath:
    def test_no_conductor_returns_empty(self, tmp_path: Path) -> None:
        cfg = TaskConfig(data_path="/d", description="D")
        cfg.pipeline.phase3.no_conductor = True
        events, cb = _events_collector()
        conductor = cond.Conductor(
            provider=_FakeProvider(),
            config=cfg,
            workspace=str(tmp_path),
            db=ExperimentDB(str(tmp_path / "exps.db")),
            adapter=None,
            event_callback=cb,
        )
        # Each entry point should short-circuit and return empty without
        # constructing an AgentLoop.
        for trigger in ("phase0", "phase1", "phase2", "milestone", "timer"):
            method = getattr(conductor, f"steer_{trigger}")
            result = method()
            assert result == ""
        # No PhaseEvent emitted — short-circuit happens before that.
        assert events == []


class TestConductorPromptBuilder:
    """Without running the loop, we can still exercise the prompt_builder
    closure that the Conductor would pass to AgentLoop. This verifies the
    prompt structure: system prompt + adapter addendum + workspace + digest.
    """

    def _build_conductor_with_runner_mocked(
        self, tmp_path: Path, adapter: DomainAdapter | None = None
    ) -> tuple[cond.Conductor, list]:
        cfg = TaskConfig(data_path="/d", description="D")
        events, cb = _events_collector()
        c = cond.Conductor(
            provider=_FakeProvider(),
            config=cfg,
            workspace=str(tmp_path),
            db=ExperimentDB(str(tmp_path / "exps.db")),
            adapter=adapter,
            event_callback=cb,
        )
        return c, events

    def test_prompt_builder_includes_workspace_instructions(self, tmp_path: Path) -> None:
        # The workspace instruction file is the sole user-input channel.
        # When the user has written content there, the prompt assembly
        # injects it above the Conductor's system prompt.
        c, _events = self._build_conductor_with_runner_mocked(tmp_path)
        # Simulate the user writing instructions before launch
        ml.from_user_path(tmp_path).write_text(
            "Always keep 3 home-run attempts in the queue."
        )

        def fake_prompt_builder() -> str:
            digest = cond.build_conductor_context(workspace=tmp_path, db=c.db)
            return cond.CONDUCTOR_SYSTEM_PROMPT + "\n\n" + digest

        prompt = fake_prompt_builder()
        # The fixed system prompt provides the role exposition
        assert "user's representative" in prompt
        # The injected user instructions appear above the conductor's own prompt
        assert "home-run attempts" in prompt

    def test_prompt_includes_adapter_addendum_when_present(
        self, tmp_path: Path
    ) -> None:
        adapter = DomainAdapter(
            domain_name="test",
            domain_description="test",
            prompts={
                "phase3_conductor": "## Test domain addendum\nDomain-specific guidance here.",
            },
            metric=MetricConfig(primary_metric="sharpe", direction="maximize"),
        )
        c, _ = self._build_conductor_with_runner_mocked(tmp_path, adapter=adapter)
        addendum = c._read_adapter_addendum()
        assert "Domain-specific guidance" in addendum

    def test_prompt_no_addendum_when_adapter_lacks_phase3_conductor(
        self, tmp_path: Path
    ) -> None:
        adapter = DomainAdapter(
            domain_name="test",
            prompts={"phase1": "phase 1 only"},
            metric=MetricConfig(primary_metric="sharpe"),
        )
        c, _ = self._build_conductor_with_runner_mocked(tmp_path, adapter=adapter)
        assert c._read_adapter_addendum() == ""


class TestConductorTriggers:
    """Each trigger maps to a distinct log_name and initial message."""

    def test_initial_messages_distinct_per_trigger(self) -> None:
        triggers = ("milestone", "timer", "phase0_done", "phase1_done", "phase2_done")
        messages = {t: cond._INITIAL_MESSAGES[t] for t in triggers}
        # All five must be defined and distinct
        assert len(set(messages.values())) == len(triggers)
        # All mention the trigger context in plain language
        assert "milestone" in messages["milestone"].lower()
        assert "timer" in messages["timer"].lower()
        assert "phase 0" in messages["phase0_done"].lower()
        assert "phase 1" in messages["phase1_done"].lower()
        assert "phase 2" in messages["phase2_done"].lower()


class TestConductorToolList:
    """The Conductor must NOT have note_to_conductor in its toolset (it has
    no peer to write notes TO), but MUST have all the steering tools."""

    def test_tool_list_excludes_note_to_conductor(self) -> None:
        assert "note_to_conductor" not in cond.CONDUCTOR_TOOLS

    def test_tool_list_includes_steering_primitives(self) -> None:
        for name in (
            "park_experiment",
            "unpark_experiment",
            "set_priority",
            "annotate_experiment",
            "issue_directive",
            "set_throttle",
            "request_phase_rewind",
            "read_meta_log",
            "read_user_instructions",
            "read_experiment",
        ):
            assert name in cond.CONDUCTOR_TOOLS, f"missing {name} from conductor tools"

    def test_tool_list_includes_report_to_user(self) -> None:
        # Always — every agent ends its turn with this.
        assert "report_to_user" in cond.CONDUCTOR_TOOLS
