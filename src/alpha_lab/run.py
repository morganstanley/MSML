"""Headless runner for alpha-lab.

Runs the agent to completion with plain-text logging. No web server,
no Rich, no interactivity. The primary way to run an analysis.

The web dashboard (server.py) is an optional monitoring layer on top.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import click

from alpha_lab.sandboxing import sandbox
from alpha_lab.config import TaskConfig, config_to_jsonable, load_config
from alpha_lab.constants import MemoryKind, Phase
from alpha_lab.events import (
    AgentEvent,
    BoardSummaryEvent,
    ErrorEvent,
    ExperimentEvent,
    PhaseEvent,
    QuestionEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from alpha_lab import deps, mlflow_logger
from alpha_lab.memory import remember_workspace_file
from alpha_lab.pipeline import Pipeline, detect_phase1_complete
from alpha_lab.token_metrics import init_token_metrics, shutdown_token_metrics
from alpha_lab.tracing import init_tracing, pipeline_span, resolve_run_id
from alpha_lab.utils import atomic_write

logger = logging.getLogger("alpha_lab")

# Module-level JSONL event log file handle + run tag, initialized in run_main()
_event_log_file = None
_pipeline_log_file = None  # {workspace}/logs/pipeline.jsonl — tailed by the dashboard server
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

    # Pipeline-level events (PhaseEvent) are emitted directly from run.py / pipeline.py
    # / phase0.py / supervisor.py without going through AgentLoop or Dispatcher, so they
    # never land in {workspace}/logs/*.jsonl where the dashboard's LogTailer watches.
    # Duplicate them into logs/pipeline.jsonl so the dashboard can surface phase state.
    if _pipeline_log_file is not None and isinstance(event, PhaseEvent):
        try:
            _pipeline_log_file.write(json.dumps(event.to_dict(), default=str) + "\n")
            _pipeline_log_file.flush()
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


def resolve_canonical_config(
    config_path: str | None,
    workspace: str,
    overwrite_config: bool,
) -> Path:
    """Resolve the workspace's canonical config, seeding it from ``--config`` if needed.

    The canonical config at ``{workspace}/.alpha_lab/config.json`` is the single source
    of truth (intake may edit it in place). ``config_path`` (``--config``) seeds it on
    the first run and is required only when no canonical config exists; passing it when
    one already exists is a hard conflict — resolved by exiting the process — unless it
    matches the existing config or ``overwrite_config`` is set (no silent clobber, no
    silent ignore).

    Match is compared as materialized ``TaskConfig``s (normalizes optional/defaulted
    fields), with ``data_path`` normalized to a workspace-relative form when the data
    lives inside the workspace, so the two agree regardless of the base each was
    resolved/stored against (external data stays absolute).

    Returns the canonical config path (guaranteed to exist on return).
    """
    canonical_config = Path(workspace) / ".alpha_lab" / "config.json"
    if config_path is not None:
        config_path = os.path.abspath(config_path)
    if config_path is None and overwrite_config:
        sys.exit("ERROR: --overwrite-config requires --config.")
    if canonical_config.exists() and config_path is not None and not overwrite_config:
        ws = Path(workspace).resolve()

        def _norm_data_path(cfg: TaskConfig, base: Path) -> TaskConfig:
            abs_dp = Path(cfg.resolve_data_path(base))
            try:
                cfg.data_path = str(abs_dp.relative_to(ws))
            except ValueError:
                cfg.data_path = str(abs_dp)
            return cfg

        # Incoming --config's relative data_path is anchored at that file's dir; the
        # canonical's is anchored at the workspace root (a workspace-internal data file
        # is relative to the workspace, never to .alpha_lab/).
        incoming = _norm_data_path(load_config(config_path), Path(config_path).parent)
        existing = _norm_data_path(load_config(canonical_config), ws)
        if incoming != existing:
            sys.exit(
                f"ERROR: {canonical_config} already exists and differs from --config. "
                "Omit --config to use the existing one, or pass --overwrite-config to replace it."
            )
    if not canonical_config.exists() and config_path is None:
        sys.exit(
            f"ERROR: --config is required: the workspace has no {canonical_config} to resume from."
        )
    if config_path is not None and (overwrite_config or not canonical_config.exists()):
        # Absolutize data_path against the source config's dir before seeding, so the
        # canonical copy resolves regardless of where it's later read from.
        seed_cfg = load_config(config_path)
        seed_cfg.data_path = seed_cfg.resolve_data_path(Path(config_path).parent)
        canonical_config.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(canonical_config, json.dumps(config_to_jsonable(seed_cfg), indent=2))
    return canonical_config


@click.command(name="alpha-lab-run", help="Run Alpha Lab analysis headlessly", context_settings={"show_default": True})
@click.option("--config", "config_path", type=str, default=None,
              help="Path to task config (YAML/JSON). Required only when the workspace has no "
                   ".alpha_lab/config.json to resume from.")
@click.option("--overwrite-config", is_flag=True, default=False,
              help="Allow --config to overwrite an existing workspace .alpha_lab/config.json.")
@click.option("--workspace", type=str, required=True, help="Workspace directory path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output (show tool outputs)")
@click.option("--run-id", type=str, default=None, help="Explicit run ID (overrides prefix and environment variable)")
@click.option("--run-id-prefix", type=str, default=None, help="Prefix for generated run ID (used if --run-id is not provided)")
@click.option("--enable-intake/--no-enable-intake", default=False, help="Run or skip the intake session before phase 0")
@click.option("--mlflow/--no-mlflow", "mlflow_enabled", default=True, show_default=True,
              help="Enable MLflow logging + native tracing (Run / metric / artifact logging). "
              "ON by default; pass --no-mlflow to disable (fallback for when the "
              "tracking server is down)."
              )

def run_main(
    config_path: str | None,
    overwrite_config: bool,
    workspace: str,
    verbose: bool,
    run_id: str | None,
    run_id_prefix: str | None,
    enable_intake: bool,
    mlflow_enabled: bool,
) -> None:
    """CLI entry point for headless agent execution."""
    # MLflow gate: every MLflow code path short-circuits on ALPHALAB_MLFLOW.
    # Flip it BEFORE any observability.mlflow call so is_active() returns True
    # for the rest of this process.
    if mlflow_enabled:
        if mlflow_logger.mlflow is None:
            sys.exit(
                "ERROR: MLflow is enabled but the 'mlflow' package is not "
                "installed. Install it (pip install 'mlflow==3.14.0') or pass "
                "--no-mlflow to run without MLflow."
            )
        os.environ["ALPHALAB_MLFLOW"] = "1"
    else:
        os.environ["ALPHALAB_MLFLOW"] = "0"

    mlflow_logger.resolve_experiment_name(workspace)

    if mlflow_enabled and (missing := mlflow_logger.missing_required_env()):
        sys.exit(
            "ERROR: MLflow is enabled but these required env vars are "
            "missing:\n" + "\n".join(f"  - {name}" for name in missing)
        )

    # Logging setup
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not mlflow_enabled:
        logging.getLogger("alpha_lab.run").warning(
            "MLflow disabled via --no-mlflow; no Run / metric / artifact logging "
            "or tracing will be sent to the tracking server."
        )

    workspace = os.path.abspath(workspace)
    Path(workspace).mkdir(parents=True, exist_ok=True)

    # The run's canonical config lives in the workspace and is the single source of
    # truth (intake may edit it in place). Seed it from --config on the first run; on
    # a resume the existing workspace copy is authoritative. --config is required only
    # when no workspace config exists, and passing it when one already exists is a hard
    # conflict unless --overwrite-config is given (no silent clobber, no silent ignore).
    canonical_config = resolve_canonical_config(config_path, workspace, overwrite_config)
    config = load_config(canonical_config)
    config_path = str(canonical_config)

    # Resolve gpu_ids "auto" onto the config now (single run entry point). Run deps
    # (config + executors) are built and published once below, via `with RunDeps(...)`,
    # around the dispatcher's lifecycle.
    _p3 = config.pipeline.phase3
    if _p3.gpu_ids == "auto":
        from alpha_lab.utils import detect_gpu_ids
        _p3.gpu_ids = detect_gpu_ids()

    # Open structured JSONL event log in workspace parent (survives workspace rm -rf)
    global _event_log_file, _pipeline_log_file, _run_tag
    from datetime import datetime, timezone
    event_log_dir = Path(workspace).parent
    event_log_dir.mkdir(parents=True, exist_ok=True)
    _event_log_file = open(event_log_dir / "events.jsonl", "a")
    # Pipeline-level events also go to {workspace}/logs/pipeline.jsonl so the
    # dashboard's LogTailer (which only watches {workspace}/logs/*.jsonl) sees them.
    pipeline_log_dir = Path(workspace) / "logs"
    pipeline_log_dir.mkdir(parents=True, exist_ok=True)
    _pipeline_log_file = open(pipeline_log_dir / "pipeline.jsonl", "a")
    _run_tag = Path(workspace).name
    # Write run-start marker
    _event_log_file.write(json.dumps({
        "type": "run_start",
        "run": _run_tag,
        "datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "config": config_path,
        "workspace": workspace,
    }) + "\n")
    _event_log_file.flush()

    # Backend selection. Mutually exclusive: MLflow on (the default) => MLflow
    # only (no OTel TracerProvider installed); with --no-mlflow the existing
    # Tempo gRPC path runs if OTEL_EXPORTER_OTLP_ENDPOINT is set.
    mlflow_active = mlflow_logger.is_active()
    if mlflow_active:
        mlflow_logger.configure_sdk()  # tracking URI, workspace (mTLS auth), autolog
    else:
        init_tracing()  # No-op if OTLP is not configured
    # run_id is the unique identifier for the pipeline execution
    run_id = resolve_run_id(
        run_id=run_id,
        run_id_prefix=run_id_prefix,
        workspace=workspace,
    )
    
    init_token_metrics(run_id=run_id)

    # A relative data_path in the canonical config is relative to the workspace root,
    # not to .alpha_lab/ (where the config file happens to live).
    config.data_path = config.resolve_data_path(workspace)

    # Check for API key (skip if on-prem or using bedrock)
    from alpha_lab.providers import ON_PREM_AVAILABLE
    provider_name = config.provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not ON_PREM_AVAILABLE and provider_name == "openai":
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    logger.info(f"Task: {config.description}")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Provider: {provider_name}")

    # Create provider
    from alpha_lab.providers import get_provider
    provider = get_provider(
        provider_name, api_key=api_key, model=config.model, model_tags=config.model_tags
    )

    # In MLflow mode: observability.mlflow.pipeline_run manages the MLflow Run
    # lifecycle. In Tempo (or none) mode: the existing pipeline_span opens an
    # OTel root span. Exactly one runs per invocation.
    pipeline_ctx = (
        mlflow_logger.pipeline_run(run_id, workspace, config)
        if mlflow_active
        else pipeline_span(run_id, workspace, config, config_path)
    )
    with pipeline_ctx, deps.RunDeps(config, workspace, run_id, api_key, owns_temp=True):
        # Deps scope wraps intake AND the pipeline. Init is lazy, so opening it before
        # intake is cheap (no executors built). Intake runs in-process (not sandboxed)
        # and reaches deps.memory_store / deps.temp_dir / deps.update_config directly;
        # external-`src` warm-start needs that filesystem/network access.

        # Intake: interactive user-proxy session. Runs before phase 0 so its
        # artifacts (agenda.md, proxy_state.md) seed adapter customization.
        # Independent of handoff — handoff has its own Bootstrapping fallback.
        # It may edit the run config in place via update_task_config (mutating
        # deps.config and persisting to the canonical file); the same deps.config
        # carries into the pipeline, so nothing needs reloading afterward.
        if enable_intake:
            from alpha_lab.cli import run_intake
            run_intake(provider, config, config_path, workspace, _log_event)

        # Pipeline-level params + the input config as an artifact. All
        # mlflow_logger.log_* calls are no-op in non-MLflow mode.
        mlflow_logger.log_pipeline_params({
            "task.description": getattr(config, "description", "") or "",
            "task.target": getattr(config, "target", "") or "",
            "data_path": str(getattr(config, "data_path", "")),
            "domain": getattr(config, "domain", "") or "",
            "provider": getattr(config, "provider", "") or "",
            "model": getattr(config, "model", "") or "",
            "reasoning_effort": getattr(config, "reasoning_effort", "") or "",
            "config_path": config_path,
            "workspace": workspace,
        })
        if Path(config_path).is_file():
            mlflow_logger.log_pipeline_artifact(Path(config_path), artifact_path="config.json")

        # Phase 0: resolve or generate domain adapter
        from alpha_lab.phase0 import run_phase0
        phase0_start = time.time()
        adapter = run_phase0(provider, config, workspace, _log_event)
        logger.info(f"Domain adapter: {adapter.domain_name} (metric: {adapter.metric.primary_metric})")
        adapter_dir = Path(workspace) / "adapter"
        if adapter_dir.is_dir():
            mlflow_logger.log_pipeline_artifacts_dir(adapter_dir, "phase0/adapter")
        mlflow_logger.log_pipeline_params({
            "phase0.adapter_domain": adapter.domain_name,
            "phase0.primary_metric": adapter.metric.primary_metric,
            "phase0.metric_direction": adapter.metric.direction,
        })
        mlflow_logger.log_pipeline_metrics(
            {"phase0.duration_seconds": time.time() - phase0_start}
        )

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
        try:
            # Phase 1: skip if already complete
            phase1_start = time.time()
            if Phase.PHASE1 in config.pipeline.phases and detect_phase1_complete(workspace):
                logger.info("Phase 1 already complete — skipping")
                _log_event(PhaseEvent(
                    phase=Phase.PHASE1, step="exploration", status="completed",
                    detail="Phase 1 already complete — skipped",
                ))
            elif Phase.PHASE1 in config.pipeline.phases:
                _log_event(PhaseEvent(
                    phase=Phase.PHASE1, step="exploration", status="starting",
                    detail="Phase 1: exploring data",
                ))
                sandbox.run_agent(
                    "phase1/explorer",
                    _log_event,
                    initial_message=initial_message,
                    provider=provider,
                    adapter=adapter,
                )
                _log_event(PhaseEvent(
                    phase=Phase.PHASE1, step="exploration", status="completed",
                    detail="Phase 1 complete",
                ))
            else:
                logger.info("Phase 1 not in pipeline — skipping")

            # Supervisor: review Phase 1
            if Phase.PHASE1 in config.pipeline.phases:
                try:
                    supervisor.review_phase1()
                except Exception as e:
                    logger.warning(f"Supervisor Phase 1 review failed: {e}")
                # Phase 1 artifacts (no-op in non-MLflow mode)
                ws_path = Path(workspace)
                if (ws_path / "learnings.md").is_file():
                    mlflow_logger.log_pipeline_artifact(
                        ws_path / "learnings.md", "phase1/learnings.md",
                    )
                    # Ingest learnings.md into memory on any path that produced it
                    # (fresh run or resume where Phase 1 was already complete); the
                    # dedup in remember_workspace_file keeps repeat calls idempotent.
                    # Best-effort: a failure must not abort the run.
                    try:
                        remember_workspace_file(
                            "learnings.md",
                            tags=["phase1", "learnings"],
                            summary="Phase 1 learnings",
                            kind=MemoryKind.PATTERN,
                            agent="phase1",
                        )
                    except Exception as e:
                        logger.warning("Failed to ingest learnings into memory: %s", e, exc_info=True)
                for sub in ("data_report", "plots", "scripts"):
                    if (ws_path / sub).is_dir():
                        mlflow_logger.log_pipeline_artifacts_dir(
                            ws_path / sub, f"phase1/{sub}",
                        )
                mlflow_logger.log_pipeline_metrics(
                    {"phase1.duration_seconds": time.time() - phase1_start}
                )

            # Phase 2: run pipeline if configured
            if Phase.PHASE2 in config.pipeline.phases:
                phase2_completed = False
                phase1_skipped = Phase.PHASE1 not in config.pipeline.phases
                if phase1_skipped and not detect_phase1_complete(workspace):
                    # Phase 1 intentionally skipped (ablation) — create stub files
                    # so Phase 2 can proceed without exploration context
                    logger.info("Phase 1 skipped — creating stub learnings for Phase 2")
                    stub_learnings = Path(workspace) / "learnings.md"
                    if not stub_learnings.exists():
                        atomic_write(
                            stub_learnings,
                            "# Learnings\n\n"
                            "Phase 1 exploration was skipped (ablation mode). "
                            "No prior data analysis available.\n"
                        )
                    stub_report_dir = Path(workspace) / "data_report"
                    stub_report_dir.mkdir(parents=True, exist_ok=True)
                    stub_report = stub_report_dir / "stub.md"
                    if not stub_report.exists():
                        atomic_write(
                            stub_report,
                            "# Data Report\n\n"
                            "Phase 1 exploration was skipped (ablation mode).\n"
                        )

                if not detect_phase1_complete(workspace):
                    logger.error("Cannot run Phase 2: Phase 1 output not found")
                else:
                    logger.info("Starting Phase 2 pipeline")
                    phase2_start = time.time()
                    pipeline = Pipeline(
                        provider=provider,
                        config=config,
                        workspace=workspace,
                        event_callback=_log_event,
                        adapter=adapter,
                    )
                    phase2_completed = pipeline.run_phase2()
                    mlflow_logger.log_pipeline_metrics(
                        {"phase2.duration_seconds": time.time() - phase2_start}
                    )

            # Supervisor: review Phase 2
            if Phase.PHASE2 in config.pipeline.phases:
                try:
                    supervisor.review_phase2()
                except Exception as e:
                    logger.warning(f"Supervisor Phase 2 review failed: {e}")
                # Phase 2 artifacts
                ws_path = Path(workspace)
                fw_dir_name = (
                    adapter.experiment.framework_dir
                    if adapter is not None and getattr(adapter, "experiment", None)
                    else "backtest"
                )
                fw_dir = ws_path / fw_dir_name
                if fw_dir.is_dir():
                    mlflow_logger.log_pipeline_artifacts_dir(
                        fw_dir, f"phase2/{fw_dir_name}",
                    )
                for fname in ("framework_review.md", "framework_critique.md"):
                    p = ws_path / fname
                    if p.is_file():
                        mlflow_logger.log_pipeline_artifact(p, f"phase2/{fname}")

                if phase2_completed is not True:
                    raise RuntimeError(
                        "Phase 2 did not complete successfully"
                    )

            # Phase 3: experiment orchestration
            if Phase.PHASE3 in config.pipeline.phases:
                from alpha_lab.dispatcher import Dispatcher
                from alpha_lab.experiment_db import ExperimentDB

                db = ExperimentDB(os.path.join(workspace, ".alpha_lab", "experiments.db"))

                dispatcher = Dispatcher(
                    provider=provider,
                    db=db,
                    event_callback=_log_event,
                    adapter=adapter,
                    supervisor=supervisor,
                )
                phase3_start = time.time()
                try:
                    dispatcher.run()
                finally:
                    # Stop/join the worker + strategist threads before RunDeps tears the executors
                    # down, so late threads can't hit cleared deps or a half-torn-down executor.
                    dispatcher.stop()

                # Phase 3 artifacts (no-op in non-MLflow mode)
                ws_path = Path(workspace)
                for fname in ("leaderboard.md", "playbook.md", "final_report.md"):
                    p = ws_path / fname
                    if p.is_file():
                        mlflow_logger.log_pipeline_artifact(p, f"phase3/{fname}")
                if (ws_path / "reports").is_dir():
                    mlflow_logger.log_pipeline_artifacts_dir(
                        ws_path / "reports", "phase3/reports",
                    )
                mlflow_logger.log_pipeline_metrics(
                    {"phase3.duration_seconds": time.time() - phase3_start}
                )
                if dispatcher.crashed:
                    raise RuntimeError("Phase 3 dispatcher crashed unexpectedly")

        except KeyboardInterrupt:
            logger.info("Interrupted, stopping")
            if pipeline is not None:
                pipeline.stop()
        finally:
            # Flush any token-usage metrics accumulated since the last periodic
            # export before the process exits (no-op if metrics weren't enabled).
            shutdown_token_metrics()
            if hasattr(provider, 'openai_client'):
                try:
                    provider.openai_client.close()
                except Exception as e:
                    logger.warning("Failed to close OpenAI client: %s", e)
            if _event_log_file is not None:
                try:
                    _event_log_file.flush()
                    _event_log_file.close()
                except OSError:
                    pass
            if _pipeline_log_file is not None:
                try:
                    _pipeline_log_file.flush()
                    _pipeline_log_file.close()
                except OSError:
                    pass


if __name__ == "__main__":
    run_main()
