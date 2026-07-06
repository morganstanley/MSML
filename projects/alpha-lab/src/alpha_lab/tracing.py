"""OpenTelemetry tracing for alpha-lab.

Encapsulates all OTel concerns in one module so callers interact with
simple helpers rather than the OTel API directly.

When no OTel SDK/exporter is configured, everything is a no-op (the
opentelemetry-api package stubs out all operations with no overhead).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate
from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, TraceFlags
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from alpha_lab import __version__

# ---------------------------------------------------------------------------
# GenAI semantic convention re-exports (Decision #1)
#
# Imported from opentelemetry-semantic-conventions' _incubating namespace.
# Re-exported here so callers (agent.py, dispatcher.py) import from this
# module only — isolating them from upstream path changes.
# ---------------------------------------------------------------------------
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (  # noqa: E402
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
)

__all__ = [
    # SDK init
    "init_tracing",
    "get_tracer",
    # Pipeline
    "resolve_run_id",
    "pipeline_span",
    # Context propagation
    "copy_context",
    "attach_context",
    "detach_context",
    "inject_context",
    "context_from_carrier",
    # GenAI semconv constants
    "GEN_AI_AGENT_NAME",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_SYSTEM",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GenAiOperationNameValues",
    # Helpers
    "provider_to_system",
    "TRACE_INFO_FILENAME",
]

# Provider name mapping (Decision #2): keeps OTel concerns out of provider.py
_PROVIDER_SYSTEM: dict[str, str] = {
    "openai": "openai",
    "bedrock": "aws.bedrock",
}


def provider_to_system(provider: str) -> str:
    """Map config provider name to gen_ai.system spec value."""
    return _PROVIDER_SYSTEM.get(provider, provider)

logger = logging.getLogger("alpha_lab.tracing")

TRACE_INFO_FILENAME = "trace_info.json"


def get_tracer() -> trace.Tracer:
    """Return a tracer from the current TracerProvider.

    Called at span-creation time rather than stored at module level because
    init_tracing() registers the real TracerProvider after import.  A
    module-level ``tracer = trace.get_tracer(...)`` would capture the no-op
    provider and never pick up the real one.
    """
    return trace.get_tracer("alpha_lab", __version__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_tracing() -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def resolve_run_id(
    run_id: str | None,
    run_id_prefix: str | None,
    workspace: str,
) -> str:
    """Resolve run ID from explicit value, prefix, env var, or fallback.

    Resolution order:
      1. run_id (--run-id) → use as-is
      2. run_id_prefix (--run-id-prefix) → {prefix}-{timestamp}-{short_uuid}
      3. ALPHALAB_RUN_ID env var → use as-is
      4. fallback → {basename}-{timestamp}-{short_uuid}
         where basename is Path(workspace).name
    """
    if run_id:
        return run_id
    if run_id_prefix:
        return f"{run_id_prefix}-{_timestamp()}-{_short_uuid()}"
    env_id = os.getenv("ALPHALAB_RUN_ID")
    if env_id:
        return env_id
    basename = Path(workspace).name
    return f"{basename}-{_timestamp()}-{_short_uuid()}"


@contextmanager
def pipeline_span(
    run_id: str,
    workspace: str,
    config: Any,
    config_path: str = "",
):
    """Root span for the entire pipeline run.

    Handles restart linking (span links to previous trace via trace_info.json)
    and writes trace_info.json for later querying.

    Usage in run_main():
        run_id = resolve_run_id(args.run_id, args.run_id_prefix, workspace)
        with pipeline_span(run_id, workspace, config, args.config):
            ...
    """
    previous = _read_trace_info(workspace)
    links: list[Link] = []
    if previous and previous.get("span_context"):
        links = [Link(previous["span_context"])]
    attempt = (previous["attempt"] + 1) if previous else 1

    attributes = {
        "user.id": os.getenv("USER", "unknown"),
        "run.id": run_id,
        "run.attempt": attempt,
        "run.workspace": workspace,
        "run.config": config_path,
        "run.domain": getattr(config, "domain", ""),
        "run.provider": getattr(config, "provider", ""),
        "run.model": getattr(config, "model", ""),
    }

    with get_tracer().start_as_current_span(
        "alpha_lab.run", links=links, attributes=attributes
    ) as span:
        # Extract trace ID and span ID for persistence
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")

        _write_trace_info(
            workspace=workspace,
            run_id=run_id,
            trace_id=trace_id,
            span_id=span_id,
            attempt=attempt,
            previous=previous,
            config_path=config_path,
        )

        logger.info(
            "Trace ID: %s (run: %s, attempt: %d)", trace_id, run_id, attempt
        )

        yield span


def copy_context() -> otel_context.Context:
    """Snapshot current OTel context for passing to a new thread.

    Use this before spawning a thread, then call attach_context() inside
    the thread to propagate the trace hierarchy.
    """
    return otel_context.get_current()


def attach_context(ctx: otel_context.Context) -> object:
    """Attach a snapshotted context in a child thread.

    Returns a token that must be passed to detach_context() when done.
    """
    return otel_context.attach(ctx)


def detach_context(token: object) -> None:
    """Detach a previously attached context."""
    otel_context.detach(token)


def inject_context() -> dict[str, str]:
    """Serialize the current OTel context into a carrier dict for cross-process
    propagation. Empty when no span is active / OTel is unconfigured.
    """
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def context_from_carrier(carrier: dict[str, str]) -> otel_context.Context:
    """Rebuild an OTel context from a carrier produced by :func:`inject_context`.

    An empty/missing carrier yields the root context (the child starts a new trace).
    """
    return propagate.extract(carrier or {})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Short timestamp for run ID generation: YYYYMMDD-HHMM."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")


def _short_uuid() -> str:
    """Short random suffix for run ID uniqueness."""
    return uuid.uuid4().hex[:8]


def _read_trace_info(workspace: str) -> dict[str, Any] | None:
    """Read existing trace_info.json for restart linking.

    Returns None on first run or if the file is unreadable.
    If a previous trace exists, the returned dict includes a 'span_context'
    key with a SpanContext object suitable for use in a Link.
    """
    path = Path(workspace) / TRACE_INFO_FILENAME
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    # Reconstruct a SpanContext from the stored IDs so we can link to it.
    # Both trace_id and span_id must be present and non-zero; a zero span_id
    # makes SpanContext.is_valid False and exporters silently drop the link.
    trace_id_str = data.get("trace_id", "")
    span_id_str = data.get("span_id", "")
    if trace_id_str and span_id_str:
        try:
            trace_id_int = int(trace_id_str, 16)
            span_id_int = int(span_id_str, 16)
            if trace_id_int == 0 or span_id_int == 0:
                logger.warning(
                    "Invalid span context in %s: trace_id or span_id is zero", path
                )
                data["span_context"] = None
            else:
                span_context = SpanContext(
                    trace_id=trace_id_int,
                    span_id=span_id_int,
                    is_remote=True,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
                data["span_context"] = span_context
        except (ValueError, TypeError) as e:
            logger.warning("Invalid span context IDs in %s: %s", path, e)
            data["span_context"] = None
    else:
        data["span_context"] = None

    return data


def _write_trace_info(
    workspace: str,
    run_id: str,
    trace_id: str,
    span_id: str,
    attempt: int,
    previous: dict[str, Any] | None,
    config_path: str = "",
) -> None:
    """Write trace_info.json to workspace for later querying."""
    info: dict[str, Any] = {
        "run_id": run_id,
        "attempt": attempt,
        "trace_id": trace_id,
        "span_id": span_id,
        "user": os.getenv("USER", "unknown"),
        "started_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "config": config_path,
    }

    if previous and previous.get("trace_id"):
        info["previous_trace_id"] = previous["trace_id"]

    path = Path(workspace) / TRACE_INFO_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2) + "\n")
    except OSError as e:
        logger.warning("Failed to write %s: %s", path, e)
