"""Always-on LLM token-usage metrics for alpha-lab.

Emit a single aggregatable :class:`Counter`
  (``alpha_lab_token_usage``) to the firm Cortex/Mimir store via
  **Prometheus remote_write**

Mechanism: the OpenTelemetry SDK ``MeterProvider`` +
``PeriodicExportingMetricReader`` + the Prometheus remote_write exporter
(``opentelemetry-exporter-prometheus-remote-write``). 

Token-type breakdown (``gen_ai.token.type`` label):

* ``input`` / ``output``
* ``cache_read`` / ``cache_write`` — a *subset* of the input tokens categorized
  by prompt-cache status.
"""

from __future__ import annotations

import atexit
import logging
import os
import uuid
from collections.abc import Callable
from typing import Any

from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from alpha_lab import __version__

from alpha_lab.tracing import GEN_AI_REQUEST_MODEL, GEN_AI_SYSTEM

__all__ = [
    "apply",
    "init_token_metrics",
    "record",
    "record_chat_usage",
    "record_embedding_usage",
    "set_forwarder",
    "shutdown_token_metrics",
]

logger = logging.getLogger("alpha_lab.token_metrics")

# --- Configuration ----------------------------------------------------------
# Token metrics are ON BY DEFAULT (see init_token_metrics). These env vars
# OVERRIDE the built-in defaults; ALPHALAB_TOKEN_METRICS_DISABLED turns the
# signal off entirely.
_DISABLED_ENV = "ALPHALAB_TOKEN_METRICS_DISABLED"   # any non-empty value -> off
_ENDPOINT_ENV = "ALPHALAB_TOKEN_METRICS_ENDPOINT"   # Cortex remote_write URL (/api/v1/push)
_TENANT_ENV = "ALPHALAB_TOKEN_METRICS_TENANT"       # Cortex tenant -> X-Scope-OrgID header
_CINIT_CC_ENV = "CINITCCNAME"                       # mTLS client cert dir (cert.pem/key.pem)

_DEFAULT_ENDPOINT = "https://metrics.example.com/api/v1/push"
_DEFAULT_TENANT = "tenant-dev"
_SERVICE_NAME = "alpha-lab"
_SCHEMA_VERSION = "v3"

# Periodic export cadence: the background reader pushes the cumulative counter to
# Cortex every N seconds during the run (plus a final flush on shutdown).
_INTERVAL_ENV = "ALPHALAB_TOKEN_METRICS_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 30.0

_COUNTER_NAME = "alpha_lab_token_usage"

# Metric attribute (Prometheus label) for the token bucket. Standard semconv
# values are "input"/"output"; we extend with "cache_read"/"cache_write".
GEN_AI_TOKEN_TYPE = "gen_ai.token.type"

# Module-level state, populated by init_token_metrics().
_provider: MeterProvider | None = None
_counter: Counter | None = None
_base_attrs: dict[str, str] = {}
# In a sandboxed child, record() forwards each delta here instead of exporting to
# Cortex, so the parent's counter stays the single exporter (see set_forwarder).
_forward: Callable[[str, str, int, int, int, int], None] | None = None


def init_token_metrics(run_id: str | None = None) -> None:
    """Set up the token-usage MeterProvider. Idempotent; on by default.

    Call once at process startup (in run.py), unconditionally. Token tracking is
    meant to run on every real run, so it is **enabled by default** using the
    built-in Cortex endpoint/tenant. It becomes a no-op when:

    * ``ALPHALAB_TOKEN_METRICS_DISABLED`` is set (explicit off-switch), or
    * no mTLS client cert is available (``$CINITCCNAME``/cert.pem missing) — e.g.
      dev laptops / CI that can't reach Cortex anyway, or
    * the endpoint override (``ALPHALAB_TOKEN_METRICS_ENDPOINT``) is blanked.

    Endpoint / tenant can be overridden via the ``ALPHALAB_TOKEN_METRICS_*``
    env vars (e.g. to target the prod tenant).
    """
    global _provider, _counter, _base_attrs

    if _counter is not None:
        return  # already initialized
    if os.getenv(_DISABLED_ENV):
        return  # explicit off-switch

    endpoint = os.getenv(_ENDPOINT_ENV, _DEFAULT_ENDPOINT)
    if not endpoint:
        return  # endpoint override blanked -> disabled

    # mTLS client cert is required to reach Cortex. Without it (dev laptops, CI)
    # stay a no-op rather than logging repeated export failures.
    cert_dir = os.getenv(_CINIT_CC_ENV, "")
    cert_file = os.path.join(cert_dir, "cert.pem") if cert_dir else ""
    key_file = os.path.join(cert_dir, "key.pem") if cert_dir else ""
    ca_file = os.path.join(cert_dir, "ca.pem") if cert_dir else ""
    if not cert_file or not os.path.exists(cert_file) or not key_file or not os.path.exists(key_file):
        logger.debug("token metrics disabled: no mTLS cert/key at $%s", _CINIT_CC_ENV)
        return

    # Imported lazily so the no-op path never requires the remote_write exporter.
    try:
        from opentelemetry.exporter.prometheus_remote_write import (
            PrometheusRemoteWriteMetricsExporter,
        )
    except Exception: 
        logger.debug("Token metrics disabled: prometheus remote_write exporter unavailable", exc_info=True)
        return

    tenant = os.getenv(_TENANT_ENV, _DEFAULT_TENANT)
    headers = {"X-Scope-OrgID": tenant} if tenant else {}
    tls_config: dict[str, str] = {
        "ca_file": ca_file,
        "cert_file": cert_file,
        "key_file": key_file,
    }
    try: 
        exporter = PrometheusRemoteWriteMetricsExporter(
            endpoint=endpoint,
            headers=headers,
            tls_config=tls_config,
            resources_as_labels=False,
        )
    except Exception: 
        logger.debug("Token metrics disabled: exporter initialization failed", exc_info=True)
        return
    try:
        interval_s = float(os.getenv(_INTERVAL_ENV, _DEFAULT_INTERVAL_SECONDS))
    except ValueError:
        interval_s = _DEFAULT_INTERVAL_SECONDS
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=int(interval_s * 1000)
    )

    _provider = MeterProvider(metric_readers=[reader])

    run_id = run_id or uuid.uuid4().hex
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME")
    _base_attrs = {
        "service.name": _SERVICE_NAME,
        "user.id": os.getenv("USER", "unknown"),
        "run_id": run_id,
        "schema": _SCHEMA_VERSION,
        "experiment": experiment if experiment else "none",
    }

    meter = _provider.get_meter("alpha_lab", __version__)
    _counter = meter.create_counter(
        _COUNTER_NAME,
        unit="",
        description="LLM tokens consumed, broken down by gen_ai.token.type.",
    )

    atexit.register(shutdown_token_metrics)
    logger.info(
        "Token metrics enabled (endpoint=%s tenant=%s user=%s run_id=%s "
        "schema=%s experiment=%s)",
        endpoint, tenant, _base_attrs["user.id"], _base_attrs["run_id"],
        _SCHEMA_VERSION, _base_attrs["experiment"],
    )


def set_forwarder(fn: Callable[[str, str, int, int, int, int], None] | None) -> None:
    """Route record() to *fn* instead of a local counter (sandboxed-child mode). """
    global _forward
    _forward = fn


def _add_to_counter(
    model: str,
    system: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
) -> None:
    """Add one API call's token buckets to the process-wide counter.

    Shared by the local recording path (:func:`record`) and the parent-side apply of
    a forwarded child delta (:func:`apply`). No-op / never raises when disabled.
    """
    if _counter is None:
        return
    try:
        common = {**_base_attrs, GEN_AI_REQUEST_MODEL: model, GEN_AI_SYSTEM: system}
        for token_type, amount in (
            ("input", input_tokens),
            ("output", output_tokens),
            ("cache_read", cache_read),
            ("cache_write", cache_write),
        ):
            if amount:
                _counter.add(amount, {**common, GEN_AI_TOKEN_TYPE: token_type})
    except Exception:  # metrics must never break an LLM call
        logger.debug("token metrics counter add failed", exc_info=True)


def apply(
    model: str,
    system: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> None:
    """Apply a token delta forwarded from a sandboxed child to the single counter.

    Called only in the orchestrator, from ``sandbox._relay``. In the real pipeline the
    orchestrator always initializes the counter before any agent runs; if a delta
    arrives here with no counter (some future entrypoint spawned a sandboxed agent
    without init), it is dropped with a debug log rather than vanishing silently.
    """
    if _counter is None:
        logger.debug(
            "token metrics apply() dropped a forwarded delta: no counter in this process"
        )
        return
    _add_to_counter(model, system, input_tokens, output_tokens, cache_read, cache_write)


def record(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
    *,
    model: str,
    system: str,
) -> None:
    """Emit one counter increment per non-zero token bucket.

    No-op if init_token_metrics() did not configure an exporter, so callers can
    invoke this unconditionally. ``input_tokens`` is the FULL input count;
    ``cache_read``/``cache_write`` are a breakdown of it (see module docstring).

    In a sandboxed child (a forwarder is installed via :func:`set_forwarder`), the
    delta is forwarded to the parent instead of exported locally; the forwarder takes
    precedence over the counter so a child never becomes its own Cortex exporter.
    """
    if _forward is not None:
        try:
            _forward(model, system, input_tokens, output_tokens, cache_read, cache_write)
        except Exception:  # metrics must never break an LLM call
            logger.debug("token metrics forward() failed", exc_info=True)
        return
    _add_to_counter(model, system, input_tokens, output_tokens, cache_read, cache_write)


def record_chat_usage(response: Any, *, model: str, system: str) -> None:
    """Record token usage from an OpenAI **Chat Completions** response object.

    Shared by the ``complete()`` paths (OpenAI summarization and the Bedrock->GPT
    proxy), whose usage shape differs from the streaming paths:
    ``usage.prompt_tokens`` / ``usage.completion_tokens`` and, for cache reads,
    ``usage.prompt_tokens_details.cached_tokens``. No-op when disabled; never
    raises.
    """
    # Skip the parse only when there's nowhere for the delta to go. In a sandboxed
    # child _counter is None but a forwarder is set, so we must still proceed there.
    if _counter is None and _forward is None:
        return
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cache_read = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        record(input_tokens, output_tokens, cache_read=cache_read,
               model=model, system=system)
    except Exception:
        logger.debug("token metrics record_chat_usage() failed", exc_info=True)


def record_embedding_usage(response: Any, *, model: str, system: str) -> None:
    """Record token usage from an OpenAI **embeddings** response object.

    Embeddings usage carries only ``usage.prompt_tokens`` / ``usage.total_tokens``. 
    So this records ``prompt_tokens`` as ``input`` and nothing else. 
    No-op when disabled; never raises.
    """
    # child _counter is None but a forwarder is set, so we must still proceed there.
    if _counter is None and _forward is None:
        return
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        record(input_tokens, 0, model=model, system=system)
    except Exception:
        logger.debug("token metrics record_embedding_usage() failed", exc_info=True)


def shutdown_token_metrics() -> None:
    """Flush and shut down the MeterProvider. Safe to call more than once."""
    global _provider, _counter
    if _provider is None:
        return
    try:
        _provider.force_flush()
        _provider.shutdown()
    except Exception:  # best-effort flush on exit
        logger.debug("token metrics shutdown error", exc_info=True)
    finally:
        _provider = None
        _counter = None
