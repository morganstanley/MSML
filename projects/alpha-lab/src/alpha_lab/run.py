"""Headless runner for alpha-lab.

Runs the agent to completion with plain-text logging. No web server,
no Rich, no interactivity. The primary way to run an analysis.

The web dashboard (server.py) is an optional monitoring layer on top.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from alpha_lab.agent import AgentLoop
from alpha_lab.config import load_config
from alpha_lab.context import ContextManager
from alpha_lab.events import (
    AgentEvent,
    AgentTextEvent,
    BoardSummaryEvent,
    ErrorEvent,
    ExperimentEvent,
    PhaseEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab.pipeline import Pipeline, detect_phase1_complete

logger = logging.getLogger("alpha_lab")

# Module-level JSONL event log file handle + run tag, initialized in run_main()
_event_log_file = None
_run_tag = ""


def _log_event(event: AgentEvent) -> None:
    """Event callback: human-readable summary to stderr + full JSONL to event log."""

    # --- Structured JSONL log (every event, machine-readable) ---
    if _event_log_file is not None:
        try:
            from datetime import datetime, timezone
            d = event.to_dict()
            d["datetime"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if _run_tag:
                d["run"] = _run_tag
            # Strip large fields to keep JSONL manageable
            if d.get("type") == "tool_result" and len(d.get("output", "")) > 2000:
                d["output"] = d["output"][:2000] + "... [truncated]"
            if d.get("image_base64"):
                d["image_base64"] = f"[{len(d['image_base64'])} chars]"
            _event_log_file.write(json.dumps(d, default=str) + "\n")
            _event_log_file.flush()
        except (OSError, TypeError, ValueError):
            pass

    # --- Human-readable stderr log (concise one-liners) ---
    if isinstance(event, StatusEvent):
        if event.status == "starting":
            logger.info("Agent starting")
        elif event.status == "thinking":
            logger.debug("Thinking...")
        elif event.status == "tool_executing":
            logger.debug(event.detail)
        elif event.status == "done":
            logger.info("Agent finished")
        elif event.status == "error":
            logger.error(event.detail)

    elif isinstance(event, ToolCallEvent):
        if event.name == "shell_exec":
            try:
                cmd = json.loads(event.arguments).get("command", "")
            except (json.JSONDecodeError, AttributeError):
                cmd = event.arguments
            # Collapse multi-line commands to a single log line
            oneline = cmd.replace("\n", " \\n ").strip()
            if len(oneline) > 200:
                oneline = oneline[:200] + "..."
            logger.info(f"shell_exec: {oneline}")
        else:
            logger.info(f"{event.name}")

    elif isinstance(event, ToolResultEvent):
        if event.name == "shell_exec":
            # Log first line of output for context
            first_line = event.output.split("\n")[0][:120]
            logger.debug(f"  -> {first_line}")
        elif event.name == "report_to_user":
            logger.info(f"Report: {event.output[:200]}")

    elif isinstance(event, ErrorEvent):
        logger.error(event.message)

    elif isinstance(event, PhaseEvent):
        logger.info(f"[{event.phase}] {event.step} — {event.status}: {event.detail}")

    elif isinstance(event, ExperimentEvent):
        logger.info(
            f"[experiment] {event.name}: {event.prev_status or '?'} -> {event.status}"
            f"{' — ' + event.detail if event.detail else ''}"
        )

    elif isinstance(event, BoardSummaryEvent):
        total = sum(event.counts.values())
        logger.info(f"[board] {total} experiments: {event.counts}")

    elif isinstance(event, QuestionEvent):
        # In headless mode, questions can't be answered
        logger.warning(f"Agent asked a question (unanswerable in headless mode): {event.question}")


def run_main() -> None:
    """CLI entry point for headless agent execution."""
    parser = argparse.ArgumentParser(
        prog="alpha-lab-run",
        description="Run Alpha Lab analysis headlessly",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to task config YAML file",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Workspace directory path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output (show tool outputs)",
    )
    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Load config
    config = load_config(args.config)
    workspace = os.path.abspath(args.workspace)
    Path(workspace).mkdir(parents=True, exist_ok=True)

    # Open structured JSONL event log in workspace parent (survives workspace rm -rf)
    global _event_log_file, _run_tag
    from datetime import datetime, timezone
    event_log_dir = Path(workspace).parent
    event_log_dir.mkdir(parents=True, exist_ok=True)
    _event_log_file = open(event_log_dir / "events.jsonl", "a")
    _run_tag = Path(workspace).name
    # Write run-start marker
    _event_log_file.write(json.dumps({
        "type": "run_start",
        "run": _run_tag,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "config": args.config,
        "workspace": workspace,
    }) + "\n")
    _event_log_file.flush()

    config.data_path = config.resolve_data_path(Path(workspace).parent)

    # Check for API key
    provider_name = config.provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and provider_name == "openai":
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    logger.info(f"Task: {config.description}")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Provider: {provider_name}")

    # Create provider
    from alpha_lab.client import get_provider
    provider = get_provider(provider_name, api_key=api_key)

    # Phase 0: resolve or generate domain adapter
    from alpha_lab.phase0 import run_phase0
    adapter = run_phase0(provider, config, workspace, _log_event)
    logger.info(f"Domain adapter: {adapter.domain_name} (metric: {adapter.metric.primary_metric})")

    # Create supervisor
    from alpha_lab.supervisor import Supervisor
    supervisor = Supervisor(
        provider=provider,
        config=config,
        workspace=workspace,
        adapter=adapter,
        event_callback=_log_event,
    )

    # Validate adapter (customization agent may have modified any file)
    logger.info("Validating adapter")
    supervisor.validate_adapter()

    context = ContextManager(
        provider=provider,
        model=config.model,
        workspace=workspace,
    )

    agent = AgentLoop(
        provider=provider,
        model=config.model,
        context=context,
        event_callback=_log_event,
        reasoning_effort=config.reasoning_effort,
        config=config,
        adapter=adapter,
    )

    initial_message = (
        f"Start. Workspace: {workspace}. "
        f"Data path: {config.data_path}. "
        f"Task: {config.description}"
    )
    if config.target:
        initial_message += f" Target variable: {config.target}."
    initial_message += " Go."

    # Run to completion (blocks)
    pipeline = None
    dispatcher = None
    executor = None
    cpu_executor = None
    try:
        # Phase 1: skip if already complete
        if "phase1" in config.pipeline.phases and detect_phase1_complete(workspace):
            logger.info("Phase 1 already complete — skipping")
            _log_event(PhaseEvent(
                phase="phase1", step="exploration", status="completed",
                detail="Phase 1 already complete — skipped",
            ))
        elif "phase1" in config.pipeline.phases:
            agent.run(initial_message)
        else:
            logger.info("Phase 1 not in pipeline — skipping")

        # Supervisor: review Phase 1
        if "phase1" in config.pipeline.phases:
            try:
                supervisor.review_phase1()
            except Exception as e:
                logger.warning(f"Supervisor Phase 1 review failed: {e}")

        # Phase 2: run pipeline if configured
        if "phase2" in config.pipeline.phases:
            phase1_skipped = "phase1" not in config.pipeline.phases
            if phase1_skipped and not detect_phase1_complete(workspace):
                # Phase 1 intentionally skipped (ablation) — create stub files
                # so Phase 2 can proceed without exploration context
                logger.info("Phase 1 skipped — creating stub learnings for Phase 2")
                stub_learnings = Path(workspace) / "learnings.md"
                if not stub_learnings.exists():
                    stub_learnings.write_text(
                        "# Learnings\n\n"
                        "Phase 1 exploration was skipped (ablation mode). "
                        "No prior data analysis available.\n"
                    )
                stub_report_dir = Path(workspace) / "data_report"
                stub_report_dir.mkdir(parents=True, exist_ok=True)
                stub_report = stub_report_dir / "stub.md"
                if not stub_report.exists():
                    stub_report.write_text(
                        "# Data Report\n\n"
                        "Phase 1 exploration was skipped (ablation mode).\n"
                    )

            if not detect_phase1_complete(workspace):
                logger.error("Cannot run Phase 2: Phase 1 output not found")
            else:
                logger.info("Starting Phase 2 pipeline")
                pipeline = Pipeline(
                    provider=provider,
                    config=config,
                    workspace=workspace,
                    event_callback=_log_event,
                    adapter=adapter,
                )
                pipeline.run_phase2()

        # Supervisor: review Phase 2
        if "phase2" in config.pipeline.phases:
            try:
                supervisor.review_phase2()
            except Exception as e:
                logger.warning(f"Supervisor Phase 2 review failed: {e}")

        # Phase 3: experiment orchestration
        if "phase3" in config.pipeline.phases:
            from alpha_lab.dispatcher import Dispatcher
            from alpha_lab.experiment_db import ExperimentDB

            p3 = config.pipeline.phase3
            db = ExperimentDB(os.path.join(workspace, "experiments.db"))

            # Create GPU executor (None if no GPUs configured)
            executor = None
            if p3.gpu_ids:
                if p3.executor == "local":
                    from alpha_lab.local_gpu import LocalGPUManager
                    executor = LocalGPUManager(
                        gpu_ids=p3.gpu_ids,
                        max_per_gpu=p3.max_per_gpu,
                        time_limit_seconds=p3.time_limit_seconds,
                        python_executable=p3.python_executable,
                    )
                else:
                    from alpha_lab.slurm import SlurmManager
                    executor = SlurmManager(
                        partitions=p3.slurm_partitions,
                        gpu_per_job=p3.gpu_per_job,
                        max_gpus=p3.max_concurrent_gpus,
                        time_limit=p3.slurm_time_limit,
                        python_executable=p3.python_executable,
                    )

            # Create CPU executor (always enabled for CPU-only mode,
            # optional alongside GPU executor for tree-based models)
            cpu_executor = None
            if p3.cpu_enabled or not p3.gpu_ids:
                from alpha_lab.local_cpu import LocalCPUManager
                cpu_executor = LocalCPUManager(
                    max_parallel=p3.cpu_max_parallel,
                    time_limit_seconds=p3.cpu_time_limit_seconds,
                    python_executable=p3.python_executable,
                )
                if not p3.gpu_ids:
                    logger.info("CPU-only mode: no GPUs configured, all experiments run on CPU")
                else:
                    logger.info(
                        f"CPU executor enabled: {p3.cpu_max_parallel} parallel slots "
                        f"for tree-based models"
                    )

            dispatcher = Dispatcher(
                provider=provider,
                config=config,
                workspace=workspace,
                db=db,
                executor=executor,
                event_callback=_log_event,
                worker_count=p3.worker_count,
                cpu_executor=cpu_executor,
                adapter=adapter,
                supervisor=supervisor,
            )
            dispatcher.run()

    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")
        agent.stop()
        if pipeline is not None:
            pipeline.stop()
        if dispatcher is not None:
            dispatcher.stop()
    finally:
        # Clean up executors to prevent orphaned processes
        if executor is not None:
            try:
                executor.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup GPU executor: %s", e)
        if cpu_executor is not None:
            try:
                cpu_executor.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup CPU executor: %s", e)
        if hasattr(provider, 'openai_client'):
            try:
                provider.openai_client.close()
            except Exception as e:
                logger.warning("Failed to close OpenAI client: %s", e)
        if _event_log_file is not None:
            try:
                _event_log_file.close()
            except OSError:
                pass


if __name__ == "__main__":
    run_main()
