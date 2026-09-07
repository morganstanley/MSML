"""Multi-agent pipeline orchestrator for alpha-lab.

Runs the builder→critic→tester loop for Phase 2, creating fresh AgentLoop
instances per step with their own prompt, tool set, and JSONL log.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent, PhaseEvent, StatusEvent
from alpha_lab.prompts import build_step_prompt
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.pipeline")


# ---------------------------------------------------------------------------
# Verdict extraction helper
# ---------------------------------------------------------------------------


def _extract_verdict(content: str) -> str:
    """Extract the final verdict from review.md content.

    Looks for structured verdict patterns to avoid false positives from
    instruction text that mentions 'NEEDS FIXES' in examples.

    Patterns matched (case-insensitive):
    - "Verdict: PASS" / "Verdict: NEEDS FIXES"
    - "Final verdict: PASS" / "Final verdict: NEEDS FIXES"
    - "**Verdict**: PASS" / "**Verdict**: NEEDS FIXES"
    - Lines starting with "## Verdict" followed by PASS/NEEDS FIXES
    - Markdown heading with just the verdict: "# PASS"
    - Bare PASS/NEEDS FIXES on its own line

    Returns "PASS", "NEEDS FIXES", or "UNCLEAR".
    """
    # Normalize whitespace
    content = content.strip()
    tail_content = content[-500:]

    # (pattern, scope) — "tail" restricts the match to the last 500 chars so
    # weak patterns (bold, heading-only, bare-line) don't pick up verdicts
    # that appear in instruction text, examples, or code blocks earlier in
    # the document. Strong patterns that require the word "verdict" nearby
    # remain "all" because they're unambiguous anywhere in the review.
    verdict_patterns = [
        # "Final verdict: PASS" / "Verdict: NEEDS FIXES" — a bounded run of
        # non-word characters is allowed between the label and the verdict
        # so decorations survive: an archived GLM critic wrote
        # "Verdict: 🟢 PASS" at the top of a long review, the old
        # colon/space/asterisk-only gap missed it, every weak pattern was
        # tail-restricted, and seven clean builds were re-"fixed" across an
        # 8-round loop (phase-2 study, 2026-08-13).
        (r'final\s+verdict[^\w]{1,16}(PASS|NEEDS\s*FIXES)', "all"),
        (r'(?<!\w)verdict[^\w]{1,16}(PASS|NEEDS\s*FIXES)', "all"),
        # Markdown header "## Verdict" followed by verdict on same/next line
        (r'##\s*(?:final\s+)?verdict[^\w]{0,16}(PASS|NEEDS\s*FIXES)', "all"),
        # Bold verdict "**PASS**" or "**NEEDS FIXES**" — tail only
        (r'\*\*(PASS|NEEDS\s*FIXES)\*\*', "tail"),
        # Markdown heading with just the verdict: "# PASS" or "## NEEDS FIXES"
        # — tail only: instruction text / templates often contain example
        # headings like "## PASS" to illustrate the format.
        (r'^#{1,3}\s+(PASS|NEEDS\s*FIXES)\s*$', "tail"),
        # Bare verdict on its own line (no surrounding words)
        # — tail only: same reason; code blocks and examples can emit
        # standalone PASS / NEEDS FIXES lines.
        (r'^\s*(PASS|NEEDS\s*FIXES)\s*$', "tail"),
    ]

    for pattern, scope in verdict_patterns:
        haystack = tail_content if scope == "tail" else content
        match = re.search(pattern, haystack, re.IGNORECASE | re.MULTILINE)
        if match:
            verdict = match.group(1).upper()
            if "NEEDS" in verdict:
                return "NEEDS FIXES"
            return "PASS"

    # Fallback: look in the last 500 characters for unstructured verdict
    # (more likely to be the actual conclusion, not instruction examples)
    tail = content[-500:].upper()
    if "NEEDS FIXES" in tail and "PASS" not in tail.split("NEEDS FIXES")[-1]:
        return "NEEDS FIXES"
    if "VERDICT" in tail and "PASS" in tail:
        return "PASS"

    return "UNCLEAR"


# ---------------------------------------------------------------------------
# Workspace state detection
# ---------------------------------------------------------------------------

def detect_phase1_complete(workspace: str) -> bool:
    """Check if Phase 1 output exists: learnings.md + data_report/ with content."""
    ws = Path(workspace)
    learnings = ws / "learnings.md"
    report_dir = ws / "data_report"

    if not learnings.exists() or not learnings.read_text().strip():
        return False
    if not report_dir.is_dir():
        return False
    md_files = list(report_dir.glob("*.md"))
    return len(md_files) > 0


def detect_phase2_progress(workspace: str, adapter: Any | None = None) -> str:
    """Detect how far Phase 2 has progressed.

    Returns the step to resume from:
      "builder"  — nothing built yet
      "critic"   — framework dir exists, needs review
      "tester"   — review passed, needs tests
      "done"     — tests exist and pass
    """
    ws = Path(workspace)

    # Use adapter framework config or defaults
    framework_dir_name = "backtest"
    framework_files = ["strategy.py", "engine.py", "metrics.py"]
    review_file = "review.md"
    if adapter is not None:
        framework_dir_name = adapter.experiment.framework_dir
        framework_files = adapter.experiment.framework_files
        review_file = adapter.phase2_review_file

    backtest = ws / framework_dir_name

    # Check if builder output exists. The manifest's ``framework_files`` is
    # the builder's checklist, but Phase 0 customizers occasionally leave
    # stale template names there (e.g. ``strategy.py``/``engine.py`` from
    # the time_series template) while the builder produces semantically
    # equivalent files under different names. To avoid wiping a working
    # framework on restart, also accept the framework as built when a
    # ``review.md`` already exists (the critic only writes it on top of a
    # built framework).
    review_path = backtest / review_file
    review_exists = backtest.is_dir() and review_path.exists()
    key_files = framework_files[:3] if len(framework_files) >= 3 else framework_files
    listed_present = (
        backtest.is_dir()
        and bool(key_files)
        and all((backtest / f).exists() for f in key_files)
    )
    if not (listed_present or review_exists):
        return "builder"

    # Check if critic has reviewed
    if not review_exists:
        return "critic"

    # Check review verdict - look for structured verdict patterns only
    # to avoid false positives from instruction text that mentions "NEEDS FIXES"
    content = review_path.read_text()
    verdict = _extract_verdict(content)
    if verdict == "NEEDS FIXES":
        return "builder"  # needs another builder pass
    if verdict != "PASS":
        return "critic"  # unclear, re-review

    # Check if tests exist and pass
    tests_dir = backtest / "tests"
    if not tests_dir.is_dir() or not list(tests_dir.glob("test_*.py")):
        return "tester"

    # Try running tests to see if they pass
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", f"{framework_dir_name}/tests/", "-v", "--tb=no", "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return "done"
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass  # Tests not runnable yet

    return "tester"


# ---------------------------------------------------------------------------
# Step configuration
# ---------------------------------------------------------------------------

@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""

    name: str                       # "builder", "critic", "tester"
    prompt_key: str                 # key into PROMPT_REGISTRY
    tool_names: list[str]           # tool names from TOOL_REGISTRY
    include_web_search: bool = False
    reasoning_effort: str = "low"
    min_report_attempts: int = 1    # critics/testers can finish on first call


@dataclass
class StepResult:
    """Result from a completed pipeline step."""

    step: str
    completed: bool
    summary: str = ""


# Step definitions
_MEMORY_TOOLS = ["memory_store", "memory_search", "memory_read"]

# Phase 2 step toolsets: each step gets ``note_to_conductor`` (so it
# can flag concerns mid-loop) and ``ack_directive`` (so it can claim
# one-shot Conductor directives that target its role). The injection
# of currently-active directives into the step's prompt context
# happens in ``_run_step`` so each agent sees what the Conductor has
# asked of *its* role specifically.
_CONDUCTOR_TOOLS_PHASE2 = ["note_to_conductor", "ack_directive"]

BUILDER_STEP = StepConfig(
    name="builder",
    prompt_key="phase2_builder",
    tool_names=(
        ["shell_exec", "view_image", "read_file", "grep_file", "report_to_user"]
        + _MEMORY_TOOLS + _CONDUCTOR_TOOLS_PHASE2
    ),
)

CRITIC_STEP = StepConfig(
    name="critic",
    prompt_key="phase2_critic",
    tool_names=(
        ["read_file", "grep_file", "shell_exec", "report_to_user"]
        + _MEMORY_TOOLS + _CONDUCTOR_TOOLS_PHASE2
    ),
    reasoning_effort="low",
)

TESTER_STEP = StepConfig(
    name="tester",
    prompt_key="phase2_tester",
    tool_names=(
        ["read_file", "grep_file", "shell_exec", "report_to_user"]
        + _MEMORY_TOOLS + _CONDUCTOR_TOOLS_PHASE2
    ),
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrates the multi-agent Phase 2 pipeline.

    Flow:
        Builder → Critic → [needs fixes? → Builder → Critic] →
        Tester → [tests fail? → Builder → Tester] → Done
    """

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        event_callback: Callable[[AgentEvent], None],
        adapter: Any | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.event_callback = event_callback
        self.adapter = adapter
        self._current_agent: AgentLoop | None = None
        self._stop_requested = False

    def stop(self) -> None:
        """Stop the currently running agent."""
        self._stop_requested = True
        if self._current_agent is not None:
            self._current_agent.stop()

    def emit(self, event: AgentEvent) -> None:
        """Emit an event via the callback."""
        self.event_callback(event)

    def run_phase2(self) -> None:
        """Run the full Phase 2 builder→critic→tester loop.

        Auto-detects progress and resumes from the right step.
        """
        max_iters = self.config.pipeline.max_fix_iterations

        # Domain-specific framework name/description come from the adapter, not
        # hardcoded finance vocabulary. Fall back to the time_series defaults only
        # when there is no adapter at all (legacy/backward-compat path).
        framework_dir = (
            self.adapter.experiment.framework_dir
            if self.adapter is not None else "backtest"
        )
        framework_desc = (
            self.adapter.phase2_framework_description
            if self.adapter is not None else "walk-forward backtesting framework"
        )

        # Detect where to start
        start_step = detect_phase2_progress(self.workspace, adapter=self.adapter)
        if start_step == "done":
            logger.info("Phase 2 already complete — all tests pass")
            self.emit(PhaseEvent(
                phase="phase2",
                step="complete",
                iteration=0,
                status="completed",
                detail="Phase 2 already complete (tests pass)",
            ))
            return

        logger.info(f"Phase 2 resuming from: {start_step}")
        self.emit(PhaseEvent(
            phase="phase2",
            step=start_step,
            iteration=0,
            status="starting",
            detail=f"Starting Phase 2 from {start_step}",
        ))

        # Load Phase 1 context from files
        phase1_context = self._load_phase1_context()

        # Skip to critic or tester if builder is already done
        skip_builder = start_step in ("critic", "tester")
        skip_critic = start_step == "tester"

        # --- Builder → Critic loop ---
        if not skip_critic:
            for iteration in range(max_iters):
                if self._stop_requested:
                    return

                # Builder (skip on first iteration if resuming from critic)
                if not (iteration == 0 and skip_builder):
                    builder_msg = f"Build the {framework_desc} in {framework_dir}/. Go."
                    if iteration > 0:
                        # Feed review feedback
                        review_content = self._read_review()
                        builder_msg = (
                            f"The critic found issues in your {framework_dir}/ code. "
                            f"Fix them and rebuild. Here is the review:\n\n{review_content}\n\n"
                            f"Fix all issues. Go."
                        )

                    self.emit(PhaseEvent(
                        phase="phase2",
                        step="builder",
                        iteration=iteration,
                        status="starting",
                        detail=f"Builder iteration {iteration}",
                    ))

                    builder_result = self._run_step(
                        BUILDER_STEP,
                        initial_message=builder_msg,
                        extra_context=phase1_context,
                        iteration=iteration,
                    )

                    if self._stop_requested:
                        return

                    self.emit(PhaseEvent(
                        phase="phase2",
                        step="builder",
                        iteration=iteration,
                        status="completed" if builder_result.completed else "failed",
                        detail=builder_result.summary[:200],
                    ))

                    if not builder_result.completed:
                        logger.error(
                            f"Builder failed on iteration {iteration} — aborting Phase 2"
                        )
                        self.emit(PhaseEvent(
                            phase="phase2",
                            step="complete",
                            iteration=0,
                            status="failed",
                            detail="Phase 2 aborted: builder failed",
                        ))
                        return

                # Critic
                self.emit(PhaseEvent(
                    phase="phase2",
                    step="critic",
                    iteration=iteration,
                    status="starting",
                    detail=f"Critic iteration {iteration}",
                ))

                critic_result = self._run_step(
                    CRITIC_STEP,
                    initial_message=f"Review the {framework_dir}/ directory for correctness. Go.",
                    extra_context=None,
                    iteration=iteration,
                )

                if self._stop_requested:
                    return

                self.emit(PhaseEvent(
                    phase="phase2",
                    step="critic",
                    iteration=iteration,
                    status="completed",
                    detail=critic_result.summary[:200],
                ))

                # Check review.md for verdict
                if self._review_passes():
                    logger.info("Critic passed — moving to tester")
                    break
                else:
                    logger.info(f"Critic found issues — iteration {iteration + 1}")
                    skip_builder = False  # force builder on subsequent iterations
            else:
                logger.error("Max fix iterations reached during critic loop — aborting Phase 2")
                self.emit(PhaseEvent(
                    phase="phase2",
                    step="complete",
                    iteration=0,
                    status="failed",
                    detail="Phase 2 aborted: max critic iterations reached without passing",
                ))
                return

        if self._stop_requested:
            return

        # --- Tester (with retry loop) ---
        for iteration in range(max_iters):
            if self._stop_requested:
                return

            tester_msg = f"Write tests for {framework_dir}/ and run them. Go."
            if iteration > 0:
                # Feed test failure output
                test_output = self._run_tests()
                tester_msg = (
                    f"Tests failed. Here is the output:\n\n{test_output}\n\n"
                    f"Fix the {framework_dir} code and/or tests, then re-run. Go."
                )
                # Re-run builder to fix, then re-test
                self.emit(PhaseEvent(
                    phase="phase2",
                    step="builder",
                    iteration=iteration,
                    status="starting",
                    detail=f"Builder fix iteration {iteration} (test failures)",
                ))

                fix_msg = (
                    f"Tests failed. Fix the {framework_dir} code. "
                    f"Test output:\n\n{test_output}\n\nFix the issues. Go."
                )
                fix_result = self._run_step(
                    BUILDER_STEP,
                    initial_message=fix_msg,
                    extra_context=phase1_context,
                    iteration=iteration,
                )

                if self._stop_requested:
                    return

                self.emit(PhaseEvent(
                    phase="phase2",
                    step="builder",
                    iteration=iteration,
                    status="completed" if fix_result.completed else "failed",
                    detail="Builder fix completed" if fix_result.completed else "Builder fix failed",
                ))

                if not fix_result.completed:
                    logger.error(
                        f"Builder fix failed on iteration {iteration} — aborting Phase 2"
                    )
                    self.emit(PhaseEvent(
                        phase="phase2",
                        step="complete",
                        iteration=0,
                        status="failed",
                        detail="Phase 2 aborted: builder fix failed",
                    ))
                    return

            self.emit(PhaseEvent(
                phase="phase2",
                step="tester",
                iteration=iteration,
                status="starting",
                detail=f"Tester iteration {iteration}",
            ))

            tester_result = self._run_step(
                TESTER_STEP,
                initial_message=tester_msg,
                extra_context=None,
                iteration=iteration,
            )

            if self._stop_requested:
                return

            self.emit(PhaseEvent(
                phase="phase2",
                step="tester",
                iteration=iteration,
                status="completed",
                detail=tester_result.summary[:200],
            ))

            # Check if tests pass
            test_output = self._run_tests()
            if self._tests_pass(test_output):
                logger.info("All tests pass")
                self.emit(PhaseEvent(
                    phase="phase2",
                    step="complete",
                    iteration=0,
                    status="completed",
                    detail="Phase 2 complete — all tests pass",
                ))
                return
            else:
                logger.info(f"Tests failed — iteration {iteration + 1}")
        else:
            logger.error("Max fix iterations reached during test loop — aborting Phase 2")
            self.emit(PhaseEvent(
                phase="phase2",
                step="complete",
                iteration=0,
                status="failed",
                detail="Phase 2 aborted: max test iterations reached without passing",
            ))
            return

        # If we get here, tests never fully passed
        final_output = self._run_tests()
        if self._tests_pass(final_output):
            self.emit(PhaseEvent(
                phase="phase2",
                step="complete",
                iteration=0,
                status="completed",
                detail="Phase 2 complete — tests pass after final check",
            ))
        else:
            self.emit(PhaseEvent(
                phase="phase2",
                step="complete",
                iteration=0,
                status="failed",
                detail="Phase 2 finished — some tests may still fail",
            ))

    def _run_step(
        self,
        step: StepConfig,
        initial_message: str,
        extra_context: str | None,
        iteration: int = 0,
    ) -> StepResult:
        """Run a single pipeline step as a fresh AgentLoop."""
        log_name = f"phase2_{step.name}_{iteration}"

        # Inject Conductor directives targeted at this step's role
        # (e.g. ``builder``/``critic``/``tester``, plus ``all``) into
        # the step's prompt context. Mirrors the strategist/worker
        # behavior so the agent sees what the Conductor asked of its
        # role without spending a tool call to read ``meta/directives.md``
        # itself, and so one-shot directives already claimed by another
        # iteration of the same step are filtered out.
        conductor_section = ""
        try:
            from alpha_lab import conductor_tools as _ct
            active = _ct.directives_for_role(self.workspace, step.name)
            acks = _ct.read_directive_acks(self.workspace)
            if active or acks:
                conductor_section = "\n" + _ct.render_directives_for_prompt(
                    active, acks=acks,
                )[:4000]
        except Exception as e:
            logger.debug("Phase 2 directive injection skipped: %s", e)

        merged_context = (
            (extra_context or "") + ("\n" + conductor_section if conductor_section else "")
        ) or None

        # Build prompt builder closure that includes extra_context and adapter
        def prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
        ) -> str:
            return build_step_prompt(
                step.prompt_key,
                workspace,
                learnings,
                config,
                merged_context,
                adapter=self.adapter,
            )

        tools = get_tool_schemas(step.tool_names, include_web_search=step.include_web_search)

        context = ContextManager(
            provider=self.provider,
            model=self.config.model,
            workspace=self.workspace,
            summarization_threshold_tokens=self.config.context_summarization_threshold_tokens,
            learnings_summary_threshold_tokens=self.config.learnings_summary_threshold_tokens,
        )

        agent = AgentLoop(
            provider=self.provider,
            model=self.config.model,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort=step.reasoning_effort,
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name=log_name,
            min_report_attempts=step.min_report_attempts,
            adapter=self.adapter,
        )

        self._current_agent = agent
        try:
            summary = agent.run(initial_message)
        finally:
            self._current_agent = None

        # Check if agent actually finished successfully (called report_to_user)
        agent_succeeded = agent._done and not self._stop_requested
        return StepResult(
            step=step.name,
            completed=agent_succeeded,
            summary=summary or "",
        )

    def _load_phase1_context(self) -> str:
        """Load Phase 1 output files as context for Phase 2."""
        parts: list[str] = []

        # learnings.md
        learnings_path = Path(self.workspace) / "learnings.md"
        if learnings_path.exists():
            parts.append(f"# learnings.md\n{learnings_path.read_text()}")

        # data_report/*.md
        report_dir = Path(self.workspace) / "data_report"
        if report_dir.is_dir():
            for md_file in sorted(report_dir.glob("*.md")):
                parts.append(f"# data_report/{md_file.name}\n{md_file.read_text()}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def _read_review(self) -> str:
        """Read the review file from the framework directory."""
        framework_dir = "backtest"
        review_file = "review.md"
        if self.adapter is not None:
            framework_dir = self.adapter.experiment.framework_dir
            review_file = self.adapter.phase2_review_file
        review_path = Path(self.workspace) / framework_dir / review_file
        if review_path.exists():
            return review_path.read_text()
        return f"(no {review_file} found)"

    def _review_passes(self) -> bool:
        """Check if backtest/review.md indicates PASS verdict."""
        content = self._read_review()
        verdict = _extract_verdict(content)
        return verdict == "PASS"

    def _run_tests(self) -> str:
        """Run pytest on the framework's tests/ directory and return output."""
        import subprocess
        framework_dir = "backtest"
        if self.adapter is not None:
            framework_dir = self.adapter.experiment.framework_dir

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", f"{framework_dir}/tests/", "-v"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            output += f"\n[exit code: {result.returncode}]"
            return output
        except Exception as e:
            return f"[ERROR] Failed to run tests: {e}"

    def _tests_pass(self, test_output: str) -> bool:
        """Check if pytest output indicates all tests passed."""
        # pytest exit code 0 = all passed — check only the suffix to avoid
        # matching "[exit code: 0]" in stdout content
        return test_output.rstrip().endswith("[exit code: 0]")

    def save_canonical_baselines(self) -> bool:
        """Run the backtest and save metrics to ``output/baseline_metrics.csv``.

        Called after Phase 2 completes. Invokes the framework's entry point
        with its own defaults — the Phase-2 agent owns any domain-specific
        options. Falls back to discovering whatever the agent already produced.

        Returns True if a canonical CSV was written.
        """
        import subprocess

        out_dir = Path(self.workspace) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical = out_dir / "baseline_metrics.csv"

        # --- Step 1: Run the framework entry point with its own defaults -----
        # adapter.experiment.entry_point is the *per-experiment* script that
        # lives in each experiment dir (e.g. run_experiment.py). For the
        # framework-level baseline we want a runner that lives in framework_dir
        # itself — conventionally named run_*.py (e.g. run_backtest.py).
        entry_module = "backtest.run_backtest"
        if self.adapter is not None:
            framework = self.adapter.experiment.framework_dir
            framework_files = self.adapter.experiment.framework_files or []
            framework_path = Path(self.workspace) / framework if framework else None
            runner = next(
                (
                    f for f in framework_files
                    if f.startswith("run_") and f.endswith(".py")
                    and framework_path and (framework_path / f).exists()
                ),
                None,
            )
            if runner and framework:
                entry_module = f"{framework}.{Path(runner).stem}"
            elif framework and framework_path and (framework_path / "run_backtest.py").exists():
                entry_module = f"{framework}.run_backtest"

        result = None
        try:
            result = subprocess.run(
                [sys.executable, "-m", entry_module],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.warning(
                    "%s exited %d: %s",
                    entry_module,
                    result.returncode,
                    (result.stderr or result.stdout)[-500:],
                )
        except Exception as e:
            logger.warning("%s failed: %s", entry_module, e)

        # --- Step 2: Find and copy the metrics to canonical path -------------
        # Search well-known locations first, then broad discovery. The schema
        # check below is domain-neutral (strategy column + a recognized metric);
        # if nothing matches, this logs a warning and returns False — the richer
        # framework-agnostic discovery lives in output_generator.
        import csv as _csv
        import glob

        # A canonical baseline table is keyed by "strategy" and carries at least
        # one recognized metric column. Recognized = the adapter's own metrics
        # plus a small generic set — so this works for any domain, not just the
        # exchange-rates "country"/"strategy" schema.
        metric_cols = {"mae", "rmse", "r2", "sharpe", "accuracy", "loss", "score"}
        if self.adapter is not None:
            m = self.adapter.metric
            metric_cols.update({m.primary_metric, m.extract_key, *(m.secondary_metrics or [])})
        metric_cols = {c.lower() for c in metric_cols if c}

        def _is_baseline_table(cols) -> bool:
            cl = {str(c).lower() for c in cols}
            return "strategy" in cl and bool(cl & metric_cols)

        framework_dir_name = "backtest"
        if self.adapter is not None and self.adapter.experiment.framework_dir:
            framework_dir_name = self.adapter.experiment.framework_dir
        search_paths = [
            glob.glob(str(Path(self.workspace) / "plots" / framework_dir_name / "metrics_summary.csv")),
            glob.glob(str(Path(self.workspace) / "plots" / framework_dir_name / "*.csv")),
            glob.glob(str(Path(self.workspace) / framework_dir_name / "metrics_summary.csv")),
        ]
        for candidates in search_paths:
            for c in candidates:
                try:
                    with open(c) as f:
                        rows = list(_csv.DictReader(f))
                    if rows and _is_baseline_table(rows[0].keys()):
                        import shutil
                        shutil.copy2(c, canonical)
                        logger.info(
                            "Canonical baselines: copied %d rows from %s", len(rows), c
                        )
                        return True
                except Exception:
                    continue

        # Fallback: search for parquet files
        try:
            import pandas as pd
            for pq in sorted(Path(self.workspace).rglob("*.parquet")):
                if "experiment" in str(pq):
                    continue
                try:
                    df = pd.read_parquet(pq)
                    if _is_baseline_table(df.columns):
                        df.to_csv(canonical, index=False)
                        logger.info(
                            "Canonical baselines: converted %d rows from %s",
                            len(df), pq,
                        )
                        return True
                except Exception:
                    continue
        except ImportError:
            pass

        logger.warning("Could not find any baseline metrics to save")
        return False
