"""MLflow integration for alpha-lab.

Self-contained MLflow Run + tracing wiring. Pure MLflow SDK — no
OpenTelemetry imports. Strictly opt-in: every public function is a no-op
unless ``ALPHALAB_MLFLOW=1`` is set in the environment (the ``--mlflow``
CLI flag on ``run.py`` flips this) **and** ``MLFLOW_TRACKING_URI`` is set.

The OTel/Tempo path in :mod:`alpha_lab.tracing` is wholly separate; the two
backends are mutually exclusive at runtime. When MLflow is active,
``init_tracing()`` in ``tracing.py`` won't install an OTel TracerProvider
(no ``OTEL_EXPORTER_OTLP_ENDPOINT`` is consulted), so the OTel SDK stays
dormant and the MLflow SDK handles all span transport itself.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("alpha_lab.mlflow_logger")

ALPHALAB_MLFLOW_ENV = "ALPHALAB_MLFLOW"
TRACE_INFO_FILENAME = "trace_info.json"

# Import the MLflow SDK once at module load. If it's missing (e.g. a deploy
# that hasn't pulled in the optional dep yet), every public API becomes a
# no-op via the ``_MLFLOW_AVAILABLE`` short-circuit in ``is_active()`` and
# the ``mlflow`` symbol is left as ``None`` (never referenced because
# ``is_active()`` gates all SDK calls upstream).
try:
    import mlflow
    from mlflow.tracing.constant import TraceMetadataKey
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    TraceMetadataKey = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def is_active() -> bool:
    """True when MLflow is installed, ``--mlflow`` is on, and the URI is set."""
    return (
        _MLFLOW_AVAILABLE
        and os.getenv(ALPHALAB_MLFLOW_ENV, "").lower() in ("1", "true", "yes")
        and bool(os.getenv("MLFLOW_TRACKING_URI"))
    )


# Alias for readability at call sites.
mlflow_enabled = is_active


def _user() -> str | None:
    """Resolve the username for ADC ``remote-user`` (and for ``set_workspace``)."""
    return os.getenv("USER") or None


def _user_headers() -> dict[str, str]:
    """ADC ``remote-user`` header sent on every MLflow HTTP call."""
    user = _user()
    return {"remote-user": user} if user else {}


# ---------------------------------------------------------------------------
# Module state — populated only while ``pipeline_run`` is active.
# ---------------------------------------------------------------------------


_run_state: dict[str, Any] = {
    "mlflow_run_uuid": None,
    "mlflow_artifact_uri": None,
    "experiment_id": None,
    "run_id": None,          # alpha-lab run identifier; also the MLflow Run name
    "user": None,
    "workspace": None,
}
_run_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SDK setup — ADC headers + auto-instrumentation
# ---------------------------------------------------------------------------


class _AlphaLabHeaderProvider:
    """MLflow ``RequestHeaderProvider`` that adds ADC + workspace headers.

    Some MLflow tracking servers use a ``remote-user`` trust-the-header auth
    scheme. The MLflow SDK doesn't know about that header by default; we
    register this provider with MLflow's
    ``_request_header_provider_registry`` so every outgoing SDK HTTP request
    gets the right headers.
    """

    def in_context(self) -> bool:
        return is_active()

    def request_headers(self) -> dict[str, str]:
        return _user_headers()


def configure_sdk() -> None:
    """One-time SDK setup: tracking URI, workspace, ADC headers, autolog.

    Idempotent — safe to call multiple times. Bails early if MLflow is off.
    """
    if not is_active():
        return

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_workspace(_user())

    # Register the ADC header provider for every SDK HTTP call. 
    try:
        from mlflow.tracking.request_header.registry import (
            _request_header_provider_registry,
        )
        _request_header_provider_registry.register(_AlphaLabHeaderProvider)
    except Exception as e:
        logger.debug("RequestHeaderProvider registration failed: %s", e)

    # Auto-instrument the provider SDKs alpha-lab talks to. Wrapped per-call
    # so an autolog-side version mismatch doesn't crash the whole --mlflow path.
    for mod_name in ("openai",):
        try:
            getattr(mlflow, mod_name).autolog()
        except Exception as e:
            logger.warning("mlflow.%s.autolog() failed: %s", mod_name, e)
    logger.info("MLflow SDK configured")


# ---------------------------------------------------------------------------
# Pipeline Run lifecycle
# ---------------------------------------------------------------------------


def _resolve_experiment_id() -> str | None:
    """Get the MLflow experiment ID from env (creating by name if needed)."""
    if experiment_id := os.getenv("MLFLOW_EXPERIMENT_ID"):
        return experiment_id
    name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not name:
        return None
    try:
        exp = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        return mlflow.create_experiment(name)
    except Exception as e:
        logger.error("Could not resolve MLflow experiment %r: %s", name, e)
        return None


def _get_or_create_run(experiment_id: str, run_name: str) -> tuple[str, str] | None:
    """Idempotent get-or-create by ``run_name``. Returns ``(uuid, artifact_uri)``."""
    client = mlflow.MlflowClient()
    # MLflow's search filter uses single-quoted string literals; escape any
    # single-quotes in the caller-supplied run_name to avoid producing a
    # malformed filter (which MLflow rejects with an opaque error).
    safe_run_name = run_name.replace("'", "\\'")
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"attributes.run_name = '{safe_run_name}'",
            max_results=1,
        )
        if runs:
            info = runs[0].info
            return info.run_id, info.artifact_uri
        run = client.create_run(experiment_id=experiment_id, run_name=run_name)
        return run.info.run_id, run.info.artifact_uri
    except Exception as e:
        logger.warning("get_or_create_run(%r) failed: %s", run_name, e)
        return None


def _write_trace_info(workspace: str, run_id: str) -> None:
    """Write ``workspace/trace_info.json`` so ``MLflowRunner`` can find this Run."""
    info = {
        "run_id": run_id,
        "user": os.getenv("USER", "unknown"),
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = Path(workspace) / TRACE_INFO_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2) + "\n")
    except OSError as e:
        logger.warning("Failed to write %s: %s", path, e)


@contextmanager
def pipeline_run(run_id: str, workspace: str, config: Any):
    """Lifecycle wrapper for the pipeline-level MLflow Run.

    On entry:
      * Resolves the experiment by ``MLFLOW_EXPERIMENT_ID`` / ``_NAME``.
      * Get-or-creates the Run named after ``run_id`` (so restarts in the
        same workspace reuse the same Run).
      * Resets status to ``RUNNING`` (so a resumed Run doesn't show stale
        ``FAILED`` in the UI during attempt N).
      * Tags ``alpha_lab.run_kind = "pipeline"``.
      * Enters ``mlflow.start_run(run_id=...)`` so spans created via
        :func:`agent_trace` (and autolog) land in this Run.
      * Writes ``trace_info.json`` so ``MLflowRunner`` can find this Run by
        name post-hoc (and re-parent it under a Suite Run).

    On exit:
      * Terminates the Run with ``FINISHED`` / ``KILLED`` / ``FAILED`` based
        on how the with-block exited.
      * Clears module state.

    No-op when MLflow isn't enabled.
    """
    if not is_active():
        yield None
        return

    client = mlflow.MlflowClient()

    experiment_id = _resolve_experiment_id()
    if experiment_id is None:
        logger.error(
            "MLflow enabled but neither MLFLOW_EXPERIMENT_ID nor "
            "MLFLOW_EXPERIMENT_NAME is set; pipeline_run is a no-op."
        )
        yield None
        return

    info = _get_or_create_run(experiment_id, run_id)
    if info is None:
        yield None
        return
    mlflow_run_uuid, mlflow_artifact_uri = info

    # Resumed Run: re-open so the UI doesn't show stale FAILED.
    try:
        client.update_run(mlflow_run_uuid, status="RUNNING")
    except Exception as e:
        logger.debug("Failed to set RUNNING on %s: %s", mlflow_run_uuid, e)

    # UI filterability.
    try:
        client.set_tag(mlflow_run_uuid, "alpha_lab.run_kind", "pipeline")
    except Exception as e:
        logger.debug("Failed to set alpha_lab.run_kind tag: %s", e)

    with _run_state_lock:
        _run_state.update({
            "mlflow_run_uuid": mlflow_run_uuid,
            "mlflow_artifact_uri": mlflow_artifact_uri,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "user": _user(),
            "workspace": workspace,
        })

    _write_trace_info(workspace, run_id)

    logger.info(
        "MLflow pipeline run: %s (workspace: %s, mlflow_run_uuid: %s, "
        "artifact_uri: %s)",
        run_id, workspace, mlflow_run_uuid, mlflow_artifact_uri,
    )

    status = "FAILED"
    try:
        # Active Run context so mlflow.start_span / autolog see the right Run.
        with mlflow.start_run(run_id=mlflow_run_uuid):
            try:
                yield mlflow_run_uuid
                status = "FINISHED"
            except KeyboardInterrupt:
                status = "KILLED"
                raise
    finally:
        try:
            client.set_terminated(mlflow_run_uuid, status=status)
        except Exception as e:
            logger.warning("Terminate run %s failed: %s", mlflow_run_uuid, e)
        with _run_state_lock:
            for k in _run_state:
                _run_state[k] = None


# ---------------------------------------------------------------------------
# Per-agent traces
# ---------------------------------------------------------------------------


class SpanType:
    """MLflow span type constants (mirror :class:`mlflow.entities.SpanType`)."""

    AGENT = "AGENT"
    LLM = "LLM"
    CHAT_MODEL = "CHAT_MODEL"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    EMBEDDING = "EMBEDDING"
    RETRIEVER = "RETRIEVER"
    PARSER = "PARSER"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"


@contextmanager
def agent_trace(
    name: str,
    *,
    target_run_uuid: str | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Open a per-agent MLflow Span scoped to a Run.

    ``target_run_uuid``:
      - ``None`` → use the active pipeline Run (set by :func:`pipeline_run`)
      - explicit UUID → switch to that Run (Phase 3 worker case: experiment
        sub-run). Useful in worker threads where the pipeline Run isn't
        automatically inherited from the main thread.

    Yields the MLflow Span (or ``None`` when MLflow is off).
    """
    if not is_active():
        yield None
        return

    target = target_run_uuid
    if target is None:
        with _run_state_lock:
            target = _run_state.get("mlflow_run_uuid")
    if target is None:
        yield None
        return

    # Link the trace to ``target`` purely via mlflow.tracing.context, which
    # pins the trace's mlflow.sourceRun through a contextvar (thread-safe).
    # Without it MLflow stamps source_run using _get_latest_active_run(),
    # which scans every thread's run stack and picks whichever run was
    # started most recently 
    with mlflow.tracing.context(metadata={TraceMetadataKey.SOURCE_RUN: target}):
        with mlflow.start_span(name=name, span_type=SpanType.AGENT) as span:
            if attributes and span is not None:
                for k, v in attributes.items():
                    try:
                        span.set_attribute(k, v)
                    except Exception:
                        pass
            yield span


@contextmanager
def child_span(name: str, *, span_type: str = SpanType.UNKNOWN):
    """Start a child Span under the current trace.

    For ``chat`` / ``execute_tool`` spans inside an :func:`agent_trace`.
    No-op when MLflow is off.
    """
    if not is_active():
        yield None
        return
    with mlflow.start_span(name=name, span_type=span_type) as span:
        yield span


def set_inputs(span: Any, value: Any) -> None:
    """Set inputs on an MLflow Span. No-op when ``span`` is ``None``."""
    if span is None:
        return
    try:
        span.set_inputs(value)
    except Exception as e:
        logger.debug("span.set_inputs failed: %s", e)


def set_outputs(span: Any, value: Any) -> None:
    """Set outputs on an MLflow Span. No-op when ``span`` is ``None``."""
    if span is None:
        return
    try:
        span.set_outputs(value)
    except Exception as e:
        logger.debug("span.set_outputs failed: %s", e)


# ---------------------------------------------------------------------------
# Sub-Run lifecycle (Phase 3 experiments)
# ---------------------------------------------------------------------------


def create_experiment_run(
    name: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[str, str] | None:
    """Create a sub-run nested under the active pipeline Run.

    Returns ``(run_uuid, artifact_uri)`` or ``None`` when MLflow isn't
    configured / no active pipeline Run.

    Tags set on the sub-run:
      * ``mlflow.parentRunId``        — standard MLflow nesting
      * ``alpha_lab.parent_run_id``    — dot-free key for UI filtering
      * ``alpha_lab.parent_run_name``  — human-readable parent name
      * ``alpha_lab.run_kind = "experiment"``
      * ``mlflow.user``
    """
    if not is_active():
        return None
    with _run_state_lock:
        parent_uuid = _run_state.get("mlflow_run_uuid")
        parent_name = _run_state.get("run_id")
        experiment_id = _run_state.get("experiment_id")
    if not parent_uuid or not experiment_id:
        return None

    client = mlflow.MlflowClient()

    tags = {
        "mlflow.parentRunId": parent_uuid,
        "alpha_lab.parent_run_id": parent_uuid,
        "alpha_lab.run_kind": "experiment",
    }
    if parent_name:
        tags["alpha_lab.parent_run_name"] = parent_name
    if user := _user():
        tags["mlflow.user"] = user

    try:
        run = client.create_run(
            experiment_id=experiment_id, run_name=name, tags=tags,
        )
    except Exception as e:
        logger.warning("create_experiment_run(%r) failed: %s", name, e)
        return None

    if params:
        for k, v in params.items():
            try:
                client.log_param(run.info.run_id, k, _stringify(v))
            except Exception as e:
                logger.debug("log_param %s on %s failed: %s", k, run.info.run_id, e)

    return run.info.run_id, run.info.artifact_uri


def terminate_run(run_uuid: str | None, status: str = "FINISHED") -> None:
    """Mark an MLflow Run terminated."""
    if not run_uuid or not is_active():
        return
    try:
        mlflow.MlflowClient().set_terminated(run_uuid, status=status)
    except Exception as e:
        logger.warning("terminate_run %s failed: %s", run_uuid, e)


# ---------------------------------------------------------------------------
# Logging API — metrics, params, artifacts (per-Run and per-pipeline-Run wrappers)
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    """Render a value safely for MLflow param logging (which requires strings)."""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            s = repr(value)
    return s[:5990] + "…[trunc]" if len(s) > 6000 else s


def log_run_metrics(run_uuid: str | None, metrics: dict[str, Any]) -> None:
    """Log numeric metrics to a specific MLflow Run."""
    if not run_uuid or not is_active():
        return
    client = mlflow.MlflowClient()
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                client.log_metric(run_uuid, k, float(v))
            except Exception as e:
                logger.debug("log_metric %s=%s failed: %s", k, v, e)


def log_run_params(run_uuid: str | None, params: dict[str, Any]) -> None:
    """Log string-valued params to a specific MLflow Run."""
    if not run_uuid or not is_active():
        return
    client = mlflow.MlflowClient()
    for k, v in params.items():
        try:
            client.log_param(run_uuid, k, _stringify(v))
        except Exception as e:
            # Params are immutable — duplicate writes on restart will fail
            # silently. log_debug instead of log_warning to avoid noise.
            logger.debug("log_param %s failed: %s", k, e)


def log_run_artifact(
    run_uuid: str | None,
    local_path: str | Path,
    artifact_path: str = "",
) -> bool:
    """Upload a single file as an artifact to ``run_uuid``.

    ``artifact_path`` is the destination path relative to the Run's
    artifact root, including the filename (e.g. ``"phase1/learnings.md"``).
    When empty, the file is uploaded to the Run's artifact root using its
    local basename.
    """
    if not run_uuid or not is_active():
        return False
    p = Path(local_path)
    if artifact_path:
        dest = Path(artifact_path)
        dest_dir = str(dest.parent) if str(dest.parent) != "." else None
    else:
        # Empty artifact_path → upload to the Run's artifact root.
        dest_dir = None
    try:
        mlflow.MlflowClient().log_artifact(run_uuid, str(p), artifact_path=dest_dir)
        return True
    except Exception as e:
        logger.warning(
            "log_artifact failed (run=%s local=%s dest=%s): %s",
            run_uuid, p, artifact_path, e,
        )
        return False


def log_run_artifacts_dir(
    run_uuid: str | None,
    local_dir: str | Path,
    artifact_path_prefix: str = "",
    max_file_bytes: int = 100 * 1024 * 1024,
    skip_globs: tuple[str, ...] = ("__pycache__", "*.pyc", ".git", "*.lock"),
) -> int:
    """Recursively upload every regular file under ``local_dir``."""
    if not run_uuid or not is_active():
        return 0
    local = Path(local_dir)
    if not local.is_dir():
        return 0
    # Resolve once so we can verify each candidate file doesn't escape ``local`` via a symlink.
    local_resolved = local.resolve()
    count = 0
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        try:
            real_path = path.resolve()
        except OSError:
            # Broken symlink, permission error, or recursive symlink.
            continue
        if not real_path.is_relative_to(local_resolved):
            logger.warning(
                "Skipping artifact %s: resolved path %s escapes %s",
                path, real_path, local_resolved,
            )
            continue
        rel_parts = path.relative_to(local).parts
        if any(any(fnmatch.fnmatch(p, pat) for pat in skip_globs) for p in rel_parts):
            continue
        try:
            if real_path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        rel = "/".join(rel_parts)
        dest = f"{artifact_path_prefix}/{rel}" if artifact_path_prefix else rel
        if log_run_artifact(run_uuid, path, dest):
            count += 1
    return count


# Pipeline-Run wrappers: route to the active pipeline Run by reading module state.


def log_pipeline_metrics(metrics: dict[str, Any]) -> None:
    with _run_state_lock:
        run_uuid = _run_state.get("mlflow_run_uuid")
    log_run_metrics(run_uuid, metrics)


def log_pipeline_params(params: dict[str, Any]) -> None:
    with _run_state_lock:
        run_uuid = _run_state.get("mlflow_run_uuid")
    log_run_params(run_uuid, params)


def log_pipeline_artifact(local_path: str | Path, artifact_path: str = "") -> bool:
    with _run_state_lock:
        run_uuid = _run_state.get("mlflow_run_uuid")
    return log_run_artifact(run_uuid, local_path, artifact_path)


def log_pipeline_artifacts_dir(local_dir: str | Path, artifact_path_prefix: str = "") -> int:
    with _run_state_lock:
        run_uuid = _run_state.get("mlflow_run_uuid")
    return log_run_artifacts_dir(run_uuid, local_dir, artifact_path_prefix)


