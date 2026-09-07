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
import shutil
import sys
import time
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
_pipeline_log_file = None  # {workspace}/logs/pipeline.jsonl — tailed by the dashboard server
_run_tag = ""
# Directory for content too large to inline in the JSONL (full tool outputs,
# images). The event keeps a preview plus a pointer, so the log stays readable
# while the original bytes remain on disk.
_overflow_dir = None
_overflow_seq = 0

# Inline preview length for tool output. Anything longer is written whole to
# the overflow directory and referenced from the event.
TOOL_OUTPUT_INLINE_CHARS = 2000


def _spill(content: str, kind: str, suffix: str) -> str:
    """Write oversized event content to its own file; return the path.

    Returns "" on failure -- logging must never interrupt a run.
    """
    global _overflow_seq
    if _overflow_dir is None:
        return ""
    try:
        _overflow_seq += 1
        path = _overflow_dir / f"{_overflow_seq:06d}_{kind}{suffix}"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
        return str(path)
    except OSError:
        return ""


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
            # Oversized content is moved out of the JSONL, never discarded:
            # the event carries a preview, the full length, and the path to
            # the complete original. A truncated tool result used to be the
            # end of that output -- the rest was gone from every log.
            out = d.get("output", "")
            if d.get("type") == "tool_result" and len(out) > TOOL_OUTPUT_INLINE_CHARS:
                d["output_full_chars"] = len(out)
                d["output_path"] = _spill(out, "tool_output", ".txt")
                d["output"] = out[:TOOL_OUTPUT_INLINE_CHARS] + "... [full output at output_path]"
            if d.get("image_base64"):
                b64 = d["image_base64"]
                d["image_full_chars"] = len(b64)
                d["image_path"] = _spill(b64, "image", ".b64")
                d["image_base64"] = f"[{len(b64)} chars, full data at image_path]"
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
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID (for tracing / MLflow run resumption)",
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default=None,
        help="Prefix for auto-generated run ID",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        dest="mlflow_flag",
        help=(
            "Enable MLflow integration (Run / metric / artifact logging + "
            "MLflow native tracing). Requires MLFLOW_TRACKING_URI and "
            "MLFLOW_EXPERIMENT_NAME (or MLFLOW_EXPERIMENT_ID)."
        ),
    )
    args = parser.parse_args()

    # MLflow gate: flip ALPHALAB_MLFLOW=1 BEFORE any mlflow_logger.is_active()
    # call so the rest of the process sees it. Validate required env vars here
    # so the failure is immediate and clear rather than deep inside the pipeline.
    if args.mlflow_flag:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            sys.exit("ERROR: --mlflow requires MLFLOW_TRACKING_URI to be set.")
        if not (
            os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or os.environ.get("MLFLOW_EXPERIMENT_ID")
        ):
            sys.exit(
                "ERROR: --mlflow requires MLFLOW_EXPERIMENT_NAME "
                "(or MLFLOW_EXPERIMENT_ID) to be set."
            )
        os.environ["ALPHALAB_MLFLOW"] = "1"

    # Install SIGTERM/SIGINT/SIGHUP handlers that kill any live shell_exec
    # subprocess groups before the parent exits. Without this, sending the
    # parent SIGTERM leaves long-running script subprocesses orphaned (still
    # holding GPUs, still writing to artifacts/), which then fight the next
    # ``run.py`` invocation when the pipeline is restarted.
    from alpha_lab.tools import install_termination_handlers
    install_termination_handlers()

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
    global _event_log_file, _pipeline_log_file, _run_tag, _overflow_dir
    from datetime import datetime, timezone
    event_log_dir = Path(workspace).parent
    event_log_dir.mkdir(parents=True, exist_ok=True)
    _event_log_file = open(event_log_dir / "events.jsonl", "a")
    _overflow_dir = event_log_dir / "event_overflow"
    _overflow_dir.mkdir(parents=True, exist_ok=True)
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
        "config": args.config,
        "workspace": workspace,
    }) + "\n")
    _event_log_file.flush()

    config.data_path = config.resolve_data_path(Path(workspace).parent)

    # Check for API key (skip if on-prem or using bedrock)
    from alpha_lab.client import ON_PREM_AVAILABLE
    provider_name = config.provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not ON_PREM_AVAILABLE and provider_name == "openai":
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    logger.info(f"Task: {config.description}")
    logger.info(f"Data: {config.data_path}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Provider: {provider_name}")

    # Backend selection. Mutually exclusive: --mlflow ⇒ MLflow only;
    # otherwise the existing Tempo gRPC path runs if OTEL_EXPORTER_OTLP_ENDPOINT is set.
    from alpha_lab import mlflow_logger
    from alpha_lab.tracing import init_tracing, resolve_run_id, pipeline_span
    mlflow_active = mlflow_logger.is_active()
    if mlflow_active:
        mlflow_logger.configure_sdk()
    else:
        init_tracing()
    run_id = resolve_run_id(
        run_id=getattr(args, "run_id", None),
        run_id_prefix=getattr(args, "run_id_prefix", None),
        workspace=workspace,
    )

    # Create provider
    from alpha_lab.client import get_provider
    provider = get_provider(provider_name, api_key=api_key)

    # Phase rewind marker handling must be available BEFORE Phase 0 runs
    # so a phase0-target marker from a prior run actually wipes the
    # adapter dir before run_phase0 inspects it. Defined as closures
    # here; the conductor used by _run_conductor_steer is bound below
    # and captured by name.
    from alpha_lab.experiment_db import ExperimentDB
    from alpha_lab.phase0 import run_phase0
    from alpha_lab.supervisor import Supervisor
    from alpha_lab.conductor import build_conductor
    from alpha_lab.conductor_tools import PHASE_REWIND_MARKER
    from alpha_lab.meta_layout import meta_dir
    _db_path = os.path.join(workspace, "experiments.db")
    _rewind_marker_path = meta_dir(workspace) / PHASE_REWIND_MARKER

    # These are rebound each rewind iteration of the main loop below.
    adapter = None
    supervisor = None
    conductor = None

    def _run_conductor_steer(method_name: str, label: str) -> bool:
        """Best-effort call into the Conductor at a phase boundary.
        Never raises — a failed steer must not block the pipeline.

        Returns True if the turn completed with a real ``report_to_user``
        summary, False if it crashed or got stuck without reporting
        (empty summary). Caller uses the return to decide whether to
        ``_record_steered`` — recording a no-report turn would cause
        future restarts to skip the audit even though no audit
        actually happened."""
        if conductor is None:
            return False
        try:
            _log_event(PhaseEvent(
                phase="conductor", step=label, status="starting",
                detail=f"Conductor steer ({label})",
            ))
            summary = getattr(conductor, method_name)()
            success = bool(summary)
            detail = (
                f"Conductor steer ({label}) complete"
                if success else
                f"Conductor steer ({label}) halted without report_to_user "
                "(cap or stop). Not recorded as audited."
            )
            _log_event(PhaseEvent(
                phase="conductor", step=label,
                status="completed" if success else "error",
                detail=detail,
            ))
            return success
        except Exception as e:
            logger.warning(f"Conductor {label} steer failed: {e}")
            _log_event(PhaseEvent(
                phase="conductor", step=label, status="error",
                detail=f"Conductor steer ({label}) failed: {e}",
            ))
            return False

    from contextlib import contextmanager
    import threading as _threading

    @contextmanager
    def _conductor_phase_sidecar(phase_name: str):
        """While a long-running phase (Phase 1 or Phase 2) is in flight,
        fire ``conductor.steer_timer`` every ``conductor_interval``
        seconds so a stuck or off-task agent gets observed without
        having to wait for ``phaseN_done`` (which may never come if
        the agent is looping). Single-thread daemon — exits when the
        wrapped block returns. Timer turns are NOT recorded in
        ``steer_state.json`` because they're continuous monitoring,
        not boundary audits."""
        stop = _threading.Event()
        thread: _threading.Thread | None = None
        if conductor is not None:
            def _tick() -> None:
                interval = max(60, int(config.pipeline.phase3.conductor_interval))
                while not stop.wait(interval):
                    try:
                        _run_conductor_steer("steer_timer", f"{phase_name}_timer")
                    except Exception as e:
                        logger.warning(f"{phase_name} conductor sidecar tick failed: {e}")
            thread = _threading.Thread(target=_tick, daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=5)

    def _consume_phase_rewind_marker() -> str | None:
        """Read meta/phase_rewind_pending.json if present, move the
        target phase's live artifacts into ``meta/backups/pre_rewind_*/``
        so ``detect_phaseN_complete`` returns False, and rotate the
        marker to a ``phase_rewind_consumed_*.json``.

        Returns the target phase string (``phase0``/``phase1``/``phase2``)
        or ``None`` if no marker was present / parseable / valid.

        Phase 0 rewinds: the adapter dir is moved aside AND phase1/2
        artifacts are wiped, because phase1/2 outputs are downstream of
        the adapter. Phase 1 rewinds wipe phase1 artifacts AND
        framework dirs (phase 2 outputs). Phase 2 rewinds wipe only the
        framework dirs.
        """
        m = _rewind_marker_path
        if not m.exists():
            return None
        try:
            payload = json.loads(m.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"phase rewind marker unreadable: {e}; rotating it aside")
            try:
                m.rename(m.with_name(f"phase_rewind_consumed_unreadable_{int(time.time())}.json"))
            except OSError as rotate_error:
                raise RuntimeError(
                    f"could not rotate unreadable phase rewind marker: {rotate_error}"
                ) from rotate_error
            return None
        target = payload.get("target_phase", "")
        if target not in ("phase0", "phase1", "phase2"):
            logger.warning(f"phase rewind marker has invalid target_phase={target!r}; rotating it aside")
            try:
                m.rename(m.with_name(f"phase_rewind_consumed_invalid_{int(time.time())}.json"))
            except OSError as rotate_error:
                raise RuntimeError(
                    f"could not rotate invalid phase rewind marker: {rotate_error}"
                ) from rotate_error
            return None
        ts = int(time.time())
        pre_dir = meta_dir(workspace) / "backups" / f"pre_rewind_{target}_{ts}"
        pre_dir.mkdir(parents=True, exist_ok=True)
        ws_path = Path(workspace)

        # Adapter (phase 0 only).
        if target == "phase0":
            src = ws_path / "adapter"
            if src.exists():
                try:
                    shutil.move(str(src), str(pre_dir / "adapter"))
                except OSError as e:
                    logger.warning(f"could not move adapter aside: {e}")

        # Phase 1 outputs (also wiped on phase 0 rewind — downstream).
        if target in ("phase0", "phase1"):
            for name in ("learnings.md", "data_report", "scripts", "plots", "notes"):
                src = ws_path / name
                if not src.exists():
                    continue
                try:
                    shutil.move(str(src), str(pre_dir / name))
                except OSError as e:
                    logger.warning(f"could not move {name} aside: {e}")

        # Phase 2 outputs (framework dir — its name is adapter-declared,
        # so resolve from the loaded adapter rather than guessing common
        # values). Always wiped on phase 0/1/2 rewinds because those
        # downstream phases regenerate framework artifacts.
        fw_dir = None
        if adapter is not None:
            fw_dir = getattr(getattr(adapter, "experiment", None), "framework_dir", None)
        if fw_dir:
            src = ws_path / fw_dir
            if src.exists() and src.is_dir():
                try:
                    shutil.move(str(src), str(pre_dir / fw_dir))
                except OSError as e:
                    logger.warning(f"could not move {fw_dir} aside: {e}")

        # Rotate the marker so it doesn't fire repeatedly.
        try:
            m.rename(m.with_name(f"phase_rewind_consumed_{ts}.json"))
        except OSError as e:
            raise RuntimeError(f"could not rotate phase rewind marker: {e}") from e

        _log_event(PhaseEvent(
            phase="conductor", step=f"rewind_to_{target}", status="completed",
            detail=(
                f"Phase rewind executed: target={target}; prior artifacts → "
                f"{pre_dir} (Conductor's own pre-rewind backup is at "
                f"{payload.get('backup_dir','?')})"
            ),
        ))
        logger.warning(
            f"Phase rewind to {target}: live artifacts moved to {pre_dir}. "
            f"Next pass will re-run from {target}."
        )
        return target

    initial_message_template = (
        f"Start. Workspace: {workspace}. "
        f"Data path: {config.data_path}. "
        f"Task: {config.description}"
    )
    if config.target:
        initial_message_template += f" Target variable: {config.target}."
    initial_message_template += " Go."

    # Backend observability context: MLflow pipeline_run (when --mlflow) or
    # OTel pipeline_span (when OTEL_EXPORTER_OTLP_ENDPOINT is set). Both are
    # no-ops when their respective backends aren't configured.
    pipeline_ctx = (
        mlflow_logger.pipeline_run(run_id, workspace, config)
        if mlflow_active
        else pipeline_span(run_id, workspace, config, args.config)
    )
    # Run to completion (blocks)
    pipeline = None
    dispatcher = None
    executor = None
    cpu_executor = None
    agent = None
    import contextlib as _contextlib
    _pipeline_stack = _contextlib.ExitStack()
    try:
        # Enter the observability context (MLflow pipeline_run or OTel pipeline_span).
        # ExitStack ensures __exit__ is called in the finally block regardless of
        # how the pipeline exits (normal, KeyboardInterrupt, or uncaught exception).
        _pipeline_stack.enter_context(pipeline_ctx)
        # Log pipeline-level params + input config artifact (no-ops when MLflow is off)
        mlflow_logger.log_pipeline_params({
            "task.description": getattr(config, "description", "") or "",
            "task.target": getattr(config, "target", "") or "",
            "data_path": str(getattr(config, "data_path", "")),
            "domain": getattr(config, "domain", "") or "",
            "provider": getattr(config, "provider", "") or "",
            "model": getattr(config, "model", "") or "",
            "reasoning_effort": getattr(config, "reasoning_effort", "") or "",
            "config_path": args.config,
            "workspace": workspace,
        })
        if Path(args.config).is_file():
            mlflow_logger.log_pipeline_artifact(args.config, artifact_path="config.json")

        # Phase 0/1/2/3 are wrapped in a rewind-aware loop. Each iteration
        # consumes any pending phase_rewind_pending.json marker at the top
        # (wiping the affected phase's artifacts so it re-runs), then
        # walks Phase 0 → Phase 3 normally. The Conductor's request_phase_rewind
        # tool writes the marker; the dispatcher exits early when it sees
        # one mid-Phase-3 (without rotating it) so this loop picks it up.
        # Termination of the loop relies on the Conductor's prompt (it
        # says rewinds are rare and require Python-verified evidence)
        # rather than a hard rewind-count cap; a runaway conductor that
        # keeps requesting rewinds would loop indefinitely here until
        # the user kills the process.
        # Track whether each phase's Conductor steer turn has already
        # fired in this process. After a rewind we only re-fire the
        # turns whose phase actually re-ran — without this guard a
        # phase2-target rewind would needlessly re-fire steer_phase0
        # (15+ minutes of opus at reasoning_effort=max for zero
        # information gain because Phase 0 didn't change).
        steered: dict[str, bool] = {
            "phase0": False, "phase1": False, "phase2": False,
        }
        last_rewind_target: str | None = None

        def _phase_artifact_hash(phase: str) -> str:
            """Return a stable content hash of the artifacts a Conductor
            steer turn would audit for the given phase. Used to skip
            no-op turns across process restarts: if the user killed and
            relaunched the pipeline and nothing has changed since the
            last successful audit, there is nothing for the Conductor
            to find that it didn't find before, and we'd otherwise burn
            ~$10 and 5-15 min of wall-clock per restart."""
            import hashlib
            h = hashlib.sha1()
            ws = Path(workspace)
            paths: list[Path] = []
            if phase == "phase0":
                ad = ws / "adapter"
                if ad.is_dir():
                    paths = sorted(p for p in ad.rglob("*") if p.is_file())
            elif phase == "phase1":
                for name in ("learnings.md", "data_report"):
                    p = ws / name
                    if p.is_file():
                        paths.append(p)
                    elif p.is_dir():
                        paths.extend(sorted(q for q in p.rglob("*") if q.is_file()))
            elif phase == "phase2":
                # Framework dir varies by adapter — use whatever the
                # adapter declares. Adapter is the source of truth; if
                # it isn't loaded yet the hash is empty and the audit
                # turn runs (safer than silently mismatching).
                fw_dir_name = None
                if adapter is not None:
                    fw_dir_name = getattr(getattr(adapter, "experiment", None), "framework_dir", None)
                if fw_dir_name:
                    d = ws / fw_dir_name
                    if d.is_dir():
                        paths = sorted(q for q in d.rglob("*") if q.is_file())
            for p in paths:
                try:
                    h.update(str(p.relative_to(ws)).encode())
                    h.update(b"\0")
                    h.update(p.read_bytes())
                    h.update(b"\0")
                except OSError:
                    continue
            return h.hexdigest()

        # Maps a state-file key → the phase whose artifacts to hash. Both
        # the conductor's steer_phaseN and the supervisor's
        # validate/review on the same phase audit the same artifacts —
        # so we hash once and key the state independently per audit
        # action (skip them independently across restarts).
        _HASH_PHASE_FOR_KEY = {
            "phase0": "phase0",
            "phase1": "phase1",
            "phase2": "phase2",
            "supervisor_validate_adapter": "phase0",
            "supervisor_review_phase1": "phase1",
            "supervisor_review_phase2": "phase2",
        }

        def _should_skip_steer(state_key: str) -> bool:
            """Check the persistent steer-state file: if a successful
            audit exists with the same artifact hash for ``state_key``,
            skip the turn.

            Falsey return means "go ahead and audit". Tolerant of any
            FS / JSON error — defaults to running the audit rather than
            silently skipping based on a corrupt state file."""
            hash_phase = _HASH_PHASE_FOR_KEY.get(state_key, state_key)
            try:
                from alpha_lab.meta_layout import steer_state_path
                p = steer_state_path(workspace)
                if not p.exists():
                    return False
                state = json.loads(p.read_text())
                entry = state.get(state_key) if isinstance(state, dict) else None
                if not isinstance(entry, dict):
                    return False
                stored = entry.get("hash", "")
                if not stored:
                    return False
                current = _phase_artifact_hash(hash_phase)
                return bool(stored) and stored == current
            except (OSError, ValueError) as e:
                logger.debug(f"_should_skip_steer({state_key}) read failed: {e}")
                return False

        def _record_steered(state_key: str) -> None:
            """After a successful audit (conductor steer or supervisor
            review), store the phase's artifact hash so the next process
            can skip if unchanged."""
            hash_phase = _HASH_PHASE_FOR_KEY.get(state_key, state_key)
            try:
                from alpha_lab.meta_layout import steer_state_path, ensure_meta_layout
                ensure_meta_layout(workspace)
                p = steer_state_path(workspace)
                state: dict = {}
                if p.exists():
                    try:
                        state = json.loads(p.read_text()) or {}
                    except ValueError:
                        state = {}
                state[state_key] = {
                    "hash": _phase_artifact_hash(hash_phase),
                    "ts": time.time(),
                }
                p.write_text(json.dumps(state, indent=2) + "\n")
            except OSError as e:
                logger.warning(f"_record_steered({state_key}) write failed: {e}")
        while True:
            # Consume any pending marker BEFORE Phase 0 runs. This is what
            # makes phase0-target rewinds actually replay Phase 0 in the
            # same process — by wiping the adapter dir before run_phase0
            # inspects it. Markers from prior runs (left if the user killed
            # mid-turn) are also handled here on first iteration.
            last_rewind_target = _consume_phase_rewind_marker()

            # Phase 0: resolve or generate domain adapter
            _t_phase0 = time.time()
            adapter = run_phase0(provider, config, workspace, _log_event)
            logger.info(
                f"Domain adapter: {adapter.domain_name} "
                f"(metric: {adapter.metric.primary_metric})"
            )
            mlflow_logger.log_pipeline_params({
                "phase0.adapter_domain": adapter.domain_name,
                "phase0.metric": adapter.metric.primary_metric,
                "phase0.metric_direction": adapter.metric.direction,
            })
            mlflow_logger.log_pipeline_metrics({"phase0.duration_s": round(time.time() - _t_phase0, 1)})
            _p0_adapter_dir = Path(workspace) / "adapter"
            if _p0_adapter_dir.is_dir():
                mlflow_logger.log_pipeline_artifacts_dir(
                    _p0_adapter_dir, artifact_path_prefix="phase0/adapter"
                )

            # (Re)build supervisor + conductor + agent with the iteration's
            # adapter. Cheap relative to the LLM calls below; safer than
            # threading adapter mutations through long-lived instances.
            supervisor = Supervisor(
                provider=provider, config=config, workspace=workspace,
                adapter=adapter, event_callback=_log_event,
            )
            # Skip the adapter-validation supervisor turn on restart if
            # the adapter is bit-identical to its last validated state.
            # On a fresh process startup the supervisor would otherwise
            # do another ~20-second LLM call just to PASS again, costing
            # tokens and wall-clock for no new information.
            if _should_skip_steer("supervisor_validate_adapter"):
                logger.info(
                    "Supervisor validate_adapter skipped — adapter unchanged "
                    "since last successful validation."
                )
                _log_event(PhaseEvent(
                    phase="phase0", step="supervisor", status="completed",
                    detail="Skipped: adapter unchanged since prior validation",
                ))
            else:
                logger.info("Validating adapter")
                supervisor.validate_adapter()
                _record_steered("supervisor_validate_adapter")

            conductor_db = ExperimentDB(_db_path)
            # Attach the DB to the supervisor so its ``read_board`` tool
            # actually returns live state instead of [ERROR] Experiment
            # database not available. The supervisor was constructed above
            # before the DB existed (so it could run validate_adapter, which
            # doesn't need the DB); now that the DB is open, attach it for
            # the remaining Phase 1/2/3 review methods.
            supervisor.db = conductor_db
            # Attach the TaskConfig to the DB so tools that need config
            # values (e.g. propose_variant's max_variants_per_base cap)
            # can read them without threading config through every dispatch
            # signature. Defensive: the dispatch handler reads via getattr
            # and falls back to defaults if the attribute is missing.
            conductor_db._task_config = config  # type: ignore[attr-defined]
            conductor = build_conductor(
                main_provider=provider, config=config, workspace=workspace,
                db=conductor_db, adapter=adapter, event_callback=_log_event,
            )

            context = ContextManager(
                provider=provider, model=config.model, workspace=workspace,
                summarization_threshold_tokens=config.context_summarization_threshold_tokens,
                learnings_summary_threshold_tokens=config.learnings_summary_threshold_tokens,
            )
            agent = AgentLoop(
                provider=provider, model=config.model, context=context,
                event_callback=_log_event,
                reasoning_effort=config.reasoning_effort,
                config=config, adapter=adapter,
            )
            initial_message = initial_message_template

            # Conductor: first turn after Phase 0. Only fire when Phase 0
            # actually re-ran — that's true on the very first iteration
            # (steered["phase0"] is False) or when this iteration was
            # triggered by a phase0-target rewind (which wipes the adapter
            # dir and forces re-customization). Also skip if the persistent
            # steer-state file says the current adapter is bit-identical
            # to a prior successful audit — avoids burning $10 and 5-15
            # min per process restart on a no-op audit.
            if not steered["phase0"] or last_rewind_target == "phase0":
                if _should_skip_steer("phase0"):
                    logger.info(
                        "Phase 0 conductor steer skipped — adapter unchanged "
                        "since last successful audit (meta/steer_state.json)."
                    )
                    _log_event(PhaseEvent(
                        phase="conductor", step="phase0_done", status="completed",
                        detail="Skipped: adapter unchanged since prior audit",
                    ))
                    steered["phase0"] = True
                else:
                    audited = _run_conductor_steer("steer_phase0", "phase0_done")
                    steered["phase0"] = True
                    if audited:
                        _record_steered("phase0")
                    if _rewind_marker_path.exists():
                        continue  # restart from Phase 0

            # Phase 1: skip if already complete
            if "phase1" in config.pipeline.phases and detect_phase1_complete(workspace):
                logger.info("Phase 1 already complete — skipping")
                _log_event(PhaseEvent(
                    phase="phase1", step="exploration", status="completed",
                    detail="Phase 1 already complete — skipped",
                ))
            elif "phase1" in config.pipeline.phases:
                _log_event(PhaseEvent(
                    phase="phase1", step="exploration", status="starting",
                    detail="Phase 1: exploring data",
                ))
                _t_phase1 = time.time()
                with _conductor_phase_sidecar("phase1"):
                    agent.run(initial_message)
                mlflow_logger.log_pipeline_metrics({"phase1.duration_s": round(time.time() - _t_phase1, 1)})
                for _p1_name in ("learnings.md",):
                    _p1_path = Path(workspace) / _p1_name
                    if _p1_path.is_file():
                        mlflow_logger.log_pipeline_artifact(_p1_path, artifact_path=f"phase1/{_p1_name}")
                for _p1_dir in ("data_report", "plots", "scripts"):
                    _p1_dir_path = Path(workspace) / _p1_dir
                    if _p1_dir_path.is_dir():
                        mlflow_logger.log_pipeline_artifacts_dir(
                            _p1_dir_path, artifact_path_prefix=f"phase1/{_p1_dir}"
                        )
                _log_event(PhaseEvent(
                    phase="phase1", step="exploration", status="completed",
                    detail="Phase 1 complete",
                ))
            else:
                logger.info("Phase 1 not in pipeline — skipping")

            # Supervisor: review Phase 1
            if "phase1" in config.pipeline.phases:
                if _should_skip_steer("supervisor_review_phase1"):
                    logger.info(
                        "Supervisor review_phase1 skipped — phase1 artifacts "
                        "unchanged since last successful review."
                    )
                    _log_event(PhaseEvent(
                        phase="phase1", step="supervisor", status="completed",
                        detail="Skipped: phase1 artifacts unchanged since prior review",
                    ))
                else:
                    try:
                        supervisor.review_phase1()
                        _record_steered("supervisor_review_phase1")
                    except Exception as e:
                        logger.warning(f"Supervisor Phase 1 review failed: {e}")

                # Conductor turn after Phase 1. Only fire when Phase 1
                # actually re-ran this iteration — i.e. first time
                # through OR after a phase0/phase1-target rewind
                # (phase0 wipes phase1 outputs; phase1 wipes its own).
                if (
                    not steered["phase1"]
                    or last_rewind_target in ("phase0", "phase1")
                ):
                    if _should_skip_steer("phase1"):
                        logger.info(
                            "Phase 1 conductor steer skipped — learnings/data_report "
                            "unchanged since last successful audit."
                        )
                        _log_event(PhaseEvent(
                            phase="conductor", step="phase1_done", status="completed",
                            detail="Skipped: phase1 artifacts unchanged since prior audit",
                        ))
                        steered["phase1"] = True
                    else:
                        audited = _run_conductor_steer("steer_phase1", "phase1_done")
                        steered["phase1"] = True
                        if audited:
                            _record_steered("phase1")
                        # If the Conductor's steer_phase1 turn requested a rewind,
                        # honor it before starting Phase 2.
                        if _rewind_marker_path.exists():
                            continue  # restart from Phase 1

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
                    _t_phase2 = time.time()
                    with _conductor_phase_sidecar("phase2"):
                        pipeline.run_phase2()
                    mlflow_logger.log_pipeline_metrics({"phase2.duration_s": round(time.time() - _t_phase2, 1)})
                    _p2_fw_dir = getattr(getattr(adapter, "experiment", None), "framework_dir", None) if adapter else None
                    if _p2_fw_dir:
                        _p2_fw_path = Path(workspace) / _p2_fw_dir
                        if _p2_fw_path.is_dir():
                            mlflow_logger.log_pipeline_artifacts_dir(
                                _p2_fw_path, artifact_path_prefix="phase2/framework"
                            )

            # Supervisor: review Phase 2
            if "phase2" in config.pipeline.phases:
                if _should_skip_steer("supervisor_review_phase2"):
                    logger.info(
                        "Supervisor review_phase2 skipped — framework dir "
                        "unchanged since last successful review."
                    )
                    _log_event(PhaseEvent(
                        phase="phase2", step="supervisor", status="completed",
                        detail="Skipped: phase2 artifacts unchanged since prior review",
                    ))
                else:
                    try:
                        supervisor.review_phase2()
                        _record_steered("supervisor_review_phase2")
                    except Exception as e:
                        logger.warning(f"Supervisor Phase 2 review failed: {e}")

                # Conductor turn after Phase 2. Fire on first iteration
                # or after any upstream rewind (phase0/1/2) that
                # re-runs Phase 2. Only check for a new rewind marker
                # if we actually fired the turn (no turn → no new
                # marker, no need to consume).
                if (
                    not steered["phase2"]
                    or last_rewind_target in ("phase0", "phase1", "phase2")
                ):
                    if _should_skip_steer("phase2"):
                        logger.info(
                            "Phase 2 conductor steer skipped — framework dir "
                            "unchanged since last successful audit."
                        )
                        _log_event(PhaseEvent(
                            phase="conductor", step="phase2_done", status="completed",
                            detail="Skipped: phase2 artifacts unchanged since prior audit",
                        ))
                        steered["phase2"] = True
                    else:
                        audited = _run_conductor_steer("steer_phase2", "phase2_done")
                        steered["phase2"] = True
                        if audited:
                            _record_steered("phase2")
                        if _rewind_marker_path.exists():
                            continue  # restart from Phase 1

            # Phase 3: experiment orchestration — inside the rewind loop so a
            # mid-Phase-3 Conductor rewind request actually replays upstream
            # phases. The dispatcher exits cleanly when it sees the marker
            # (see Dispatcher._consume_phase_rewind_marker which sets
            # _stop_requested but leaves the marker for us to read).
            if "phase3" in config.pipeline.phases:
                from alpha_lab.dispatcher import Dispatcher

                p3 = config.pipeline.phase3
                # Reuse the ExperimentDB instance the Conductor already
                # holds — the underlying SQLite file is the same workspace
                # path. Avoids opening two handles.
                db = conductor_db

                # Create GPU executor based on config (rebuilt each iteration
                # because Dispatcher takes ownership; previous iteration's
                # executor was cleaned up at the bottom of the loop).
                if p3.executor == "local":
                    from alpha_lab.local_gpu import LocalGPUManager
                    executor = LocalGPUManager(
                        gpu_ids=p3.gpu_ids,
                        max_per_gpu=p3.max_per_gpu,
                        time_limit_seconds=p3.time_limit_seconds,
                        python_executable=p3.python_executable,
                        data_path=config.data_path or "",
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

                # Create optional CPU executor for tree-based models
                cpu_executor = None
                if p3.cpu_enabled:
                    from alpha_lab.local_cpu import LocalCPUManager
                    cpu_executor = LocalCPUManager(
                        max_parallel=p3.cpu_max_parallel,
                        time_limit_seconds=p3.cpu_time_limit_seconds,
                        python_executable=p3.python_executable,
                        data_path=config.data_path or "",
                    )
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
                    conductor=conductor,
                )
                dispatcher.run()

                # MLflow: Phase 3 summary artifacts
                for _p3_name in ("leaderboard.md", "playbook.md", "final_report.md"):
                    _p3_path = Path(workspace) / _p3_name
                    if _p3_path.is_file():
                        mlflow_logger.log_pipeline_artifact(_p3_path, artifact_path=f"phase3/{_p3_name}")
                _p3_reports = Path(workspace) / "reports"
                if _p3_reports.is_dir():
                    mlflow_logger.log_pipeline_artifacts_dir(
                        _p3_reports, artifact_path_prefix="phase3/reports"
                    )

                # If Phase 3 exited because the Conductor requested a phase
                # rewind, the marker is still on disk (dispatcher does NOT
                # rotate it any more). Leave it for the loop-top consumer.
                if _rewind_marker_path.exists():
                    # Gracefully shut down this iteration's dispatcher
                    # (joins worker/strategist/conductor threads, cleans
                    # up executors). Without this, leftover threads from
                    # a half-finished Phase 3 would race the next
                    # iteration's dispatcher on experiments.db.
                    try:
                        dispatcher.stop()
                    except Exception as e:
                        logger.warning("dispatcher.stop() on rewind failed: %s", e)
                    executor = None
                    cpu_executor = None
                    dispatcher = None
                    continue

            break  # success — no more rewinds, exit the loop

    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")
        if agent is not None:
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
        # Exit the observability context (MLflow pipeline_run or OTel pipeline_span).
        try:
            _pipeline_stack.close()
        except Exception as e:
            logger.warning("Pipeline observability stack close failed: %s", e)
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
