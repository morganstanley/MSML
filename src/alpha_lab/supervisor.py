"""Supervisory agent for alpha-lab.

Monitors pipeline phases, catches problems, and can patch the domain adapter.
Each review method runs a short-lived AgentLoop with tools to inspect workspace
artifacts and optionally patch adapter files.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab.adapter import DomainAdapter
from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.supervisor")


# ---------------------------------------------------------------------------
# Supervisor system prompts
# ---------------------------------------------------------------------------

VALIDATE_ADAPTER_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing a newly generated domain adapter.

## Checks
1. **Completeness**: All 11 files present (manifest.json, 9 prompt .md files, domain_knowledge.md)
2. **Manifest validity**: Valid JSON with required fields (metric, experiment)
3. **Prompts substantive**: Each prompt .md file is >100 characters and contains domain-specific content
4. **Metric sensible**: primary_metric, direction, and extract_key are consistent
5. **Experiment structure**: required_files and entry_point are specified

## Actions
- Use `read_adapter` to read the current adapter files
- If issues found, use `patch_adapter_file` to fix them
- Call `report_to_user` with your assessment (PASS/NEEDS_FIXES + details). **Keep the review tight** — a single `read_adapter`, possibly a handful of `read_file` probes for specific issues, optional `patch_adapter_file` calls, then `report_to_user`. If you find yourself reading the adapter repeatedly or running shell probes beyond the basics, stop and finalize: the goal is a quick PASS/FAIL, not a deep audit.

Be strict but practical. Minor style issues are OK. Missing files or broken JSON are not.
"""

REVIEW_PHASE1_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing Phase 1 (exploration) output.

## Checks
1. **learnings.md** exists and contains substantive findings
2. **data_report/** directory has findings.md and/or schema.md
3. **scripts/** directory has exploration scripts
4. **plots/** directory has visualization outputs
5. No obvious errors or empty files

## Actions
- Use `read_file` to inspect key files
- Use `shell_exec` to check file sizes and directory contents
- If the adapter prompts seem misaligned with the data, use `patch_adapter_file`
- Call `report_to_user` with PASS/NEEDS_ATTENTION + details

Don't block progress — Phase 1 doesn't need to be perfect.
"""

REVIEW_PHASE2_PROMPT = """\
You are the **Alpha Lab Supervisor** reviewing Phase 2 (framework) output.

## Checks
1. **Framework directory** exists with expected files
2. **Tests exist** and pass (check test output)
3. **Review verdict** is PASS (check review.md)
4. **No obvious bugs** in framework code

## Actions
- Use `read_file` to inspect framework files and review.md
- Use `shell_exec` to run tests if needed
- If the adapter's framework config is wrong, use `patch_adapter_file`
- Call `report_to_user` with PASS/NEEDS_FIXES + details
"""

PHASE3_HEALTH_CHECK_PROMPT = """\
You are the **Alpha Lab Supervisor** checking Phase 3 experiment health.

The error rate has exceeded 40%. Your job is to diagnose the SPECIFIC
failure mode driving the rate up — not to add rules from a catalog of
common problems. Generic pattern-matching has historically misfired on
this codebase (see the constraints below).

## Mandatory diagnosis sequence — do these IN ORDER before any patch

1. **Inventory rules already in the adapter.** Call `read_adapter` and
   read the current `phase3_worker_implement.md`. Enumerate every rule
   already there (e.g. "OPENBLAS thread caps", "JOBLIB_TEMP_FOLDER",
   "native_fallback via try/except numpy", "do not set checked after
   smoke", etc.). You will NOT re-add a rule already in this list.

2. **Check how recent each rule is.** Run via `shell_exec`:
   `git log --oneline -n 30 -- adapter/phase3_worker_implement.md`
   (run from the workspace dir). The output shows when each patch was
   applied. A rule added in the last few hours has not yet had a chance
   to affect newly-implemented experiments; patches don't help in-flight
   experiments whose code was written before the rule landed.

3. **Sample failing experiments and attribute by library.** Use
   `read_board` to find recent failed/errored experiments. For at
   least 5 of them, do BOTH:
   - `read_file` their `run_experiment.py` to see which heavy libraries
     they actually import (lightgbm, torch, sklearn, scipy, xgboost,
     joblib, etc.).
   - `read_file` their `experiments/{name}/local_job*.out` (the
     subprocess stdout/stderr) and look at the last lines before the
     crash to see which library was active at the time of failure.

   Tally: of N sampled failing experiments, M used library X. The patch
   you propose MUST target the library that appears in >50% of failing
   experiments. Do NOT diagnose "joblib mmap on NFS" because the failure
   shape looks textbook — verify joblib is in the imports first.

4. **Verify the target failure is actually happening.** Before
   proposing any new rule, identify at least 3 specific failing
   experiments (by name) whose error text or subprocess log matches
   the SPECIFIC failure mode the rule targets. If you cannot find 3,
   the rule targets a phantom — do not add it.

5. **Check that existing rules don't already cover this.** If the
   failures you observe are in a library that an existing rule already
   addresses (e.g. you observe lightgbm SIGBUS but `native_fallback`
   with `try: import lightgbm except ImportError` is already in the
   prompt), the rule is taking effect on NEW implementations but the
   in-flight queue of pre-rule code is still draining through.
   Sample 3 experiments implemented AFTER that rule was added (use
   git log timestamps from step 2 and compare against
   `experiments/{name}/run_experiment.py` mtime). If post-rule
   implementations have adopted the rule, the patch is working —
   do NOT add a duplicate. Call `report_to_user` with "existing rule
   X is in effect; backlog draining; no new patch."

## Constraints
- Read the logs. `logs/` is the primary source of truth for what
  happened in the run. Use `grep_file` against `logs/`, individual
  log files, and `experiments/{name}/local_job*.out` to ground every
  claim in actual events. A wide grep across `logs/` may time out
  (the dir is large); when it does, narrow the path or filter by
  filename (`--include` style via `path=logs/dispatcher.jsonl` etc.)
  rather than abandoning log inspection.
- `read_board` complements log reading: use it for live DB state
  (current statuses, error fields, results presence). Use it
  alongside grep_file, not in place of it.
- A rule whose target failure occurred 0 times in the recent window
  must not be added, regardless of how textbook the failure mode is.
- A rule that duplicates an existing one (same target failure, same
  intervention) must not be added.

## Tools available
- `read_board` — live experiment DB state (status, errors, results)
- `read_adapter` — current adapter content (existing rules)
- `read_file`, `grep_file` — for narrow inspection
- `shell_exec` — for `git log`, file listing, and per-experiment scans
- `patch_adapter_file` — only after steps 1-5 are done
- `report_to_user` — diagnosis + what was patched (if anything)

## Required content in report_to_user
Your summary MUST state:
- The library-attribution tally from step 3 (e.g. "of 8 sampled
  failures: 5 import lightgbm and crashed during fit; 2 import torch;
  1 numpy-only — patching lightgbm import safety")
- The 3 specific experiment names from step 4 that exhibit the target
- Whether the failure overlaps an existing rule (step 5), and if so
  why a new patch is justified, or that no patch was made

Focus on fixes that target a specific, measured failure mode. Generic
rules added defensively against textbook problems that aren't happening
in THIS run have caused more harm than good in past runs.
"""


class Supervisor:
    """Meta-agent that monitors pipeline phases and patches the adapter."""

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        adapter: DomainAdapter,
        event_callback: Callable[[AgentEvent], None],
        db: Any | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.adapter = adapter
        self.event_callback = event_callback
        # ``db`` is optional at construction time because ``validate_adapter``
        # (Phase 0) runs before the experiment DB exists. Phase 1/2/3 review
        # methods need it so the ``read_board`` tool actually works inside the
        # supervisor's AgentLoop. Callers should set ``supervisor.db`` to the
        # ExperimentDB instance as soon as it's available — without this, every
        # ``read_board`` call returns ``[ERROR] Experiment database not
        # available`` and the supervisor diagnoses Phase 3 health entirely
        # from filesystem snapshots, blind to live state.
        self.db = db

    def _disabled(self, log_name: str, phase_name: str) -> bool:
        """NOOP fallback: when ``no_supervisor=True`` every public entry point
        short-circuits. Emit the normal phase-completed event so callers see
        the same event shape they would on a real review, then return True
        so the caller can ``return ""`` immediately. Mirrors no_conductor.
        Must be called at the very top of each public Supervisor method,
        BEFORE any work (some methods read adapter state before delegating
        to ``_run_review``)."""
        if not getattr(self.config.pipeline.phase3, "no_supervisor", False):
            return False
        logger.info(
            f"Supervisor {log_name} skipped — no_supervisor=True in config."
        )
        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="completed",
            detail=f"Skipped: no_supervisor=True ({log_name})",
        ))
        return True

    def _run_review(
        self,
        system_prompt: str,
        initial_message: str,
        tools: list[dict],
        log_name: str,
        phase_name: str,
    ) -> str:
        """Run a short-lived review agent and return its final report."""
        if self._disabled(log_name, phase_name):
            return ""
        context = ContextManager(
            provider=self.provider,
            model=self.config.model,
            workspace=self.workspace,
            summarization_threshold_tokens=self.config.context_summarization_threshold_tokens,
            learnings_summary_threshold_tokens=self.config.learnings_summary_threshold_tokens,
        )

        def prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
        ) -> str:
            parts = [system_prompt]
            if workspace:
                parts.append(f"\n## Workspace\n`{workspace}`")
            return "\n".join(parts)

        agent = AgentLoop(
            provider=self.provider,
            model=self.config.model,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort=self.config.reasoning_effort,
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name=log_name,
            min_report_attempts=1,
            db=self.db,
            adapter=self.adapter,
        )

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="starting",
            detail=f"Supervisor review: {log_name}",
        ))

        report = agent.run(initial_message)

        self.event_callback(PhaseEvent(
            phase=phase_name, step="supervisor", status="completed",
            detail=f"Supervisor review complete: {log_name}",
        ))

        return report or ""

    def validate_adapter(self) -> str:
        """After Phase 0: check all adapter files present and valid."""
        logger.info("Supervisor: validating adapter")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
            # So the Supervisor can flag adapter/framework issues to the
            # Conductor for steering, and acknowledge any
            # Conductor-targeted directives addressed to it.
            "note_to_conductor", "ack_directive",
        ])
        return self._run_review(
            system_prompt=VALIDATE_ADAPTER_PROMPT,
            initial_message=(
                "Review the domain adapter in the workspace. "
                "Check completeness, validity, and quality. Go."
            ),
            tools=tools,
            log_name="supervisor_validate_adapter",
            phase_name="phase0",
        )

    def review_phase1(self) -> str:
        """After Phase 1: check exploration artifacts."""
        logger.info("Supervisor: reviewing Phase 1")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
            # So the Supervisor can flag adapter/framework issues to the
            # Conductor for steering, and acknowledge any
            # Conductor-targeted directives addressed to it.
            "note_to_conductor", "ack_directive",
        ])
        return self._run_review(
            system_prompt=REVIEW_PHASE1_PROMPT,
            initial_message=(
                "Review Phase 1 exploration output. "
                "Check learnings.md, data_report/, scripts/, plots/. Go."
            ),
            tools=tools,
            log_name="supervisor_review_phase1",
            phase_name="phase1",
        )

    def review_phase2(self) -> str:
        """After Phase 2: check framework, tests, review verdict."""
        # Disable-check before reading adapter state — review_phase2
        # accesses ``self.adapter.experiment.framework_dir`` to build the
        # initial message, which would NPE if the adapter is absent or
        # waste work when the supervisor is disabled.
        if self._disabled("supervisor_review_phase2", "phase2"):
            return ""
        logger.info("Supervisor: reviewing Phase 2")
        framework_dir = "backtest"
        if self.adapter:
            framework_dir = self.adapter.experiment.framework_dir
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec",
            "read_adapter", "patch_adapter_file", "report_to_user",
            # So the Supervisor can flag adapter/framework issues to the
            # Conductor for steering, and acknowledge any
            # Conductor-targeted directives addressed to it.
            "note_to_conductor", "ack_directive",
        ])
        return self._run_review(
            system_prompt=REVIEW_PHASE2_PROMPT,
            initial_message=(
                f"Review Phase 2 framework output in {framework_dir}/. "
                f"Check files, tests, and review verdict. Go."
            ),
            tools=tools,
            log_name="supervisor_review_phase2",
            phase_name="phase2",
        )

    def phase3_health_check(self) -> str:
        """During Phase 3: diagnose high error rate."""
        logger.info("Supervisor: Phase 3 health check")
        tools = get_tool_schemas([
            "read_file", "grep_file", "shell_exec", "read_board",
            "read_adapter", "patch_adapter_file", "report_to_user",
            "note_to_conductor", "ack_directive",
        ])
        return self._run_review(
            system_prompt=PHASE3_HEALTH_CHECK_PROMPT,
            initial_message=(
                "The Phase 3 experiment error rate has exceeded 40%. "
                "Diagnose the systemic issue and patch the adapter if needed. Go."
            ),
            tools=tools,
            log_name="supervisor_health_check",
            phase_name="phase3",
        )
