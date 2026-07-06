"""Multi-agent pipeline orchestrator for alpha-lab.

Runs the builder→critic→tester loop for Phase 2, running each step as a fresh
agent via ``sandbox.run_agent`` (sandboxed when bwrap is available) with its own
prompt, tool set, and JSONL log.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.agents import load_agent
from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig
from alpha_lab.events import AgentEvent, PhaseEvent, StatusEvent
from alpha_lab.provider import Provider

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
        # "Final verdict: PASS" or "Final verdict: NEEDS FIXES"
        (r'final\s+verdict[:\s*]+\**(PASS|NEEDS\s*FIXES)\**', "all"),
        # "Verdict: PASS" or "Verdict: NEEDS FIXES"
        (r'(?<!\w)verdict[:\s*]+\**(PASS|NEEDS\s*FIXES)\**', "all"),
        # Markdown header "## Verdict" followed by verdict on same/next line
        (r'##\s*(?:final\s+)?verdict[:\s]*(PASS|NEEDS\s*FIXES)', "all"),
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

    # Check if builder output exists
    key_files = framework_files[:3] if len(framework_files) >= 3 else framework_files
    if not backtest.is_dir() or not all((backtest / f).exists() for f in key_files):
        return "builder"

    # Check if critic has reviewed
    review_path = backtest / review_file
    if not review_path.exists():
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
    """Configuration for a single pipeline step.

    Tools, reasoning_effort, min_report_attempts, and include_web_search are
    loaded from the agent file at `agents/phase2/{name}.md` at runtime.
    """

    name: str                       # "builder", "critic", "tester"
    prompt_key: str                 # key into PROMPT_REGISTRY


@dataclass
class StepResult:
    """Result from a completed pipeline step."""

    step: str
    completed: bool
    summary: str = ""


# Step definitions
BUILDER_STEP = StepConfig(name="builder", prompt_key="phase2_builder")
CRITIC_STEP = StepConfig(name="critic", prompt_key="phase2_critic")
TESTER_STEP = StepConfig(name="tester", prompt_key="phase2_tester")


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
        self._current_agent: sandbox.AgentRunHandle | None = None
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
                    fw_dir = self._framework_dir_name()
                    builder_msg = (
                        f"Build the evaluation framework in {fw_dir}/. Go."
                    )
                    if iteration > 0:
                        # Feed review feedback
                        review_content = self._read_review()
                        builder_msg = (
                            f"The critic found issues in your {fw_dir}/ code. "
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
                    initial_message=(
                        f"Review the {self._framework_dir_name()}/ directory "
                        f"for correctness. Go."
                    ),
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

            fw_dir = self._framework_dir_name()
            tester_msg = f"Write tests for {fw_dir}/ and run them. Go."
            if iteration > 0:
                # Feed test failure output
                test_output = self._run_tests()
                tester_msg = (
                    f"Tests failed. Here is the output:\n\n{test_output}\n\n"
                    f"Fix the {fw_dir}/ code and/or tests, then re-run. Go."
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
                    f"Tests failed. Fix the {fw_dir}/ code. "
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
        """Run a single pipeline step as a fresh agent (sandboxed when bwrap is available)."""
        agent_definition = load_agent(f"phase2/{step.name}")
        log_name = f"{agent_definition.log_name}_{iteration}"

        handle = sandbox.run_agent(
            f"phase2/{step.name}",
            self.event_callback,
            initial_message=initial_message,
            extra_context=extra_context,
            log_name=log_name,
            provider=self.provider,
            adapter=self.adapter,
            owner=self,
            handle_attr="_current_agent",
        )

        # Check if agent actually finished successfully (called report_to_user)
        agent_succeeded = handle.done and not self._stop_requested
        return StepResult(
            step=step.name,
            completed=agent_succeeded,
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

    def _framework_dir_name(self) -> str:
        """Return the adapter's framework directory name, falling back to ``backtest``."""
        if self.adapter is not None:
            return self.adapter.experiment.framework_dir
        return "backtest"

    def _read_review(self) -> str:
        """Read the review file from the framework directory."""
        framework_dir = self._framework_dir_name()
        review_file = "review.md"
        if self.adapter is not None:
            review_file = self.adapter.phase2_review_file
        review_relpath_obj = Path(framework_dir) / review_file
        if review_relpath_obj.is_absolute():
            logger.warning("Adapter review path must be workspace-relative: %s", review_relpath_obj)
            return f"(no {review_file} found)"

        workspace_root = Path(self.workspace).resolve()
        review_path = (workspace_root / review_relpath_obj).resolve()
        try:
            review_relpath = review_path.relative_to(workspace_root).as_posix()
        except ValueError:
            logger.warning("Adapter review path escapes workspace: %s", review_relpath_obj)
            return f"(no {review_file} found)"

        try:
            content = review_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return f"(no {review_file} found)"

        verdict = _extract_verdict(content)
        if verdict != "PASS":
            try:
                from alpha_lab.memory import remember_text
                remember_text(
                    self.workspace,
                    content,
                    tags=["phase2", "review"],
                    summary=f"{framework_dir} review found issues",
                    kind="failure",
                    phase="phase2",
                    agent="critic",
                    source_path=review_relpath,
                )
            except Exception as e:
                logger.warning("Failed to ingest review into memory: %s", e, exc_info=True)
        return content

    def _review_passes(self) -> bool:
        """Check if the framework's review file indicates PASS verdict."""
        content = self._read_review()
        verdict = _extract_verdict(content)
        return verdict == "PASS"

    def _run_tests(self) -> str:
        """Run pytest on the framework's tests/ directory and return output."""
        import subprocess
        framework_dir = self._framework_dir_name()

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
        # Search well-known locations first, then broad discovery.
        # The schema check below (country+strategy) is still time-series specific;
        # for non-time-series frameworks this function logs a warning and returns
        # False — the richer framework-agnostic discovery lives in output_generator.
        import csv as _csv
        import glob

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
                    if rows and "country" in rows[0] and "strategy" in rows[0]:
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
                    if "country" in df.columns and "strategy" in df.columns:
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
