"""Tests for the token-usage metrics module.

Covers the four public entry points:

* ``init_token_metrics`` — the enable/no-op decision and MeterProvider wiring
* ``record`` — per-bucket counter increments and label wiring
* ``record_chat_usage`` — parsing OpenAI Chat Completions usage shapes
* ``shutdown_token_metrics`` — flush + teardown idempotency

The network exporter (Prometheus remote_write to Cortex) is never exercised:
``record``-path tests wire a real :class:`MeterProvider` to an
``InMemoryMetricReader`` so we assert on the actual SDK data points, and
``init`` tests stub the exporter class so no socket is ever opened.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricExporter

from alpha_lab import __version__, token_metrics
from alpha_lab.token_metrics import (
    GEN_AI_TOKEN_TYPE,
    _COUNTER_NAME,
    init_token_metrics,
    record,
    record_chat_usage,
    record_embedding_usage,
    shutdown_token_metrics,
)
from alpha_lab.tracing import GEN_AI_REQUEST_MODEL, GEN_AI_SYSTEM


@pytest.fixture(autouse=True)
def _reset_module_state() -> Any:
    """Reset token_metrics' module globals around every test.

    The module keeps process-wide singletons (_provider/_counter/_base_attrs).
    Reset before and after each test so ordering never leaks state, and so the
    conftest-wide ALPHALAB_TOKEN_METRICS_DISABLED=1 default can't make an
    already-initialized counter linger.
    """
    token_metrics._provider = None
    token_metrics._counter = None
    token_metrics._base_attrs = {}
    token_metrics._forward = None
    yield
    if token_metrics._provider is not None:
        try:
            token_metrics._provider.shutdown()
        except Exception:
            pass
    token_metrics._provider = None
    token_metrics._counter = None
    token_metrics._base_attrs = {}
    token_metrics._forward = None


@pytest.fixture()
def in_memory_counter() -> Any:
    """Wire token_metrics to a real MeterProvider + InMemoryMetricReader.

    Yields the reader so tests can collect data points via
    ``reader.get_metrics_data()``. This exercises the genuine OTel counter and
    attribute plumbing without any network exporter.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("alpha_lab", __version__)
    token_metrics._provider = provider
    token_metrics._counter = meter.create_counter(_COUNTER_NAME, unit="")
    token_metrics._base_attrs = {
        "service.name": "alpha-lab",
        "user.id": "tester",
        "run_id": "run-abc",
        "schema": token_metrics._SCHEMA_VERSION,
        "experiment": "none",
    }
    return reader


def _points(reader: InMemoryMetricReader) -> list[Any]:
    """Flatten the counter's number data points from a reader collection."""
    data = reader.get_metrics_data()
    points: list[Any] = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == _COUNTER_NAME:
                    points.extend(metric.data.data_points)
    return points


def _by_type(reader: InMemoryMetricReader) -> dict[str, int]:
    """Map gen_ai.token.type -> summed value across data points."""
    out: dict[str, int] = {}
    for p in _points(reader):
        ttype = p.attributes[GEN_AI_TOKEN_TYPE]
        out[ttype] = out.get(ttype, 0) + p.value
    return out


# --- record() ---------------------------------------------------------------
class TestRecord:
    """record() emits one counter increment per non-zero token bucket."""

    def test_noop_when_uninitialized(self) -> None:
        """With no counter configured, record() is a silent no-op."""
        # _reset_module_state guarantees _counter is None here.
        record(10, 20, model="gpt-5.2", system="openai")  # must not raise

    def test_records_all_four_buckets(self, in_memory_counter: Any) -> None:
        record(100, 50, cache_read=30, cache_write=10,
               model="gpt-5.2", system="openai")
        assert _by_type(in_memory_counter) == {
            "input": 100, "output": 50, "cache_read": 30, "cache_write": 10,
        }

    def test_labels_include_model_system_and_base_attrs(
        self, in_memory_counter: Any
    ) -> None:
        record(5, 5, model="claude-opus-4-6-v1", system="aws.bedrock")
        point = _points(in_memory_counter)[0]
        attrs = point.attributes
        assert attrs[GEN_AI_REQUEST_MODEL] == "claude-opus-4-6-v1"
        assert attrs[GEN_AI_SYSTEM] == "aws.bedrock"
        # base attributes are merged onto every point
        assert attrs["service.name"] == "alpha-lab"
        assert attrs["user.id"] == "tester"
        assert attrs["run_id"] == "run-abc"
        assert attrs["schema"] == token_metrics._SCHEMA_VERSION
        assert attrs["experiment"] == "none"

    def test_counter_is_cumulative(self, in_memory_counter: Any) -> None:
        """Two calls with the same labels accumulate into one summed point."""
        record(10, 0, model="m", system="openai")
        record(15, 0, model="m", system="openai")
        assert _by_type(in_memory_counter) == {"input": 25}

    def test_distinct_models_produce_distinct_points(
        self, in_memory_counter: Any
    ) -> None:
        record(10, 0, model="model-a", system="openai")
        record(20, 0, model="model-b", system="openai")
        inputs = {
            p.attributes[GEN_AI_REQUEST_MODEL]: p.value
            for p in _points(in_memory_counter)
        }
        assert inputs == {"model-a": 10, "model-b": 20}

    def test_never_raises_on_counter_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure inside the counter is swallowed (metrics must not break calls)."""
        class Boom:
            def add(self, *a: Any, **k: Any) -> None:
                raise RuntimeError("exporter blew up")

        monkeypatch.setattr(token_metrics, "_counter", Boom())
        # Should log-and-continue rather than propagate.
        record(10, 20, model="m", system="openai")


# --- forwarder / apply (sandboxed-child -> parent) --------------------------
class TestForwarderAndApply:
    """A sandboxed child forwards deltas; the parent applies them to its counter."""

    def test_record_forwards_and_does_not_touch_counter(
        self, in_memory_counter: Any
    ) -> None:
        """With a forwarder installed, record() ships the delta and adds nothing local."""
        seen: list[tuple] = []
        token_metrics.set_forwarder(
            lambda *args: seen.append(args)
        )
        record(100, 50, cache_read=30, cache_write=10, model="gpt-5.2", system="openai")
        assert seen == [("gpt-5.2", "openai", 100, 50, 30, 10)]
        assert _points(in_memory_counter) == []  # counter untouched in child mode

    def test_forwarder_takes_precedence_over_counter(
        self, in_memory_counter: Any
    ) -> None:
        """The forwarder wins even when a counter also happens to be configured."""
        token_metrics.set_forwarder(lambda *args: None)
        record(10, 0, model="m", system="openai")
        assert _by_type(in_memory_counter) == {}

    def test_forwarder_failure_is_swallowed(self) -> None:
        def boom(*args: Any) -> None:
            raise RuntimeError("pipe closed")

        token_metrics.set_forwarder(boom)
        record(10, 20, model="m", system="openai")  # must not raise

    def test_apply_adds_forwarded_delta_to_counter(
        self, in_memory_counter: Any
    ) -> None:
        token_metrics.apply("gpt-5.2", "openai", 100, 50, 30, 10)
        assert _by_type(in_memory_counter) == {
            "input": 100, "output": 50, "cache_read": 30, "cache_write": 10,
        }

    def test_apply_stamps_parent_base_attrs(self, in_memory_counter: Any) -> None:
        """apply() labels with the PARENT's run_id/user/etc., not the child's."""
        token_metrics.apply("model-x", "anthropic", 5, 5)
        attrs = _points(in_memory_counter)[0].attributes
        assert attrs["run_id"] == "run-abc"
        assert attrs[GEN_AI_REQUEST_MODEL] == "model-x"

    def test_apply_is_noop_without_counter(self) -> None:
        """A forwarded delta with no counter in this process is dropped, not fatal."""
        # _reset_module_state guarantees _counter is None here.
        token_metrics.apply("m", "openai", 10, 20)  # must not raise


# --- record_chat_usage() ----------------------------------------------------
class _Details:
    def __init__(self, cached_tokens: int) -> None:
        self.cached_tokens = cached_tokens


class _Usage:
    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cached is not None:
            self.prompt_tokens_details = _Details(cached)


class _Response:
    def __init__(self, usage: Any) -> None:
        self.usage = usage


class TestRecordChatUsage:
    """record_chat_usage() maps a Chat Completions usage object onto record()."""

    def test_noop_when_uninitialized(self) -> None:
        resp = _Response(_Usage(prompt_tokens=10, completion_tokens=5))
        record_chat_usage(resp, model="m", system="openai")  # must not raise

    def test_maps_prompt_and_completion_tokens(
        self, in_memory_counter: Any
    ) -> None:
        resp = _Response(_Usage(prompt_tokens=120, completion_tokens=60))
        record_chat_usage(resp, model="gpt-5.2", system="openai")
        assert _by_type(in_memory_counter) == {"input": 120, "output": 60}

    def test_maps_cached_tokens_to_cache_read(
        self, in_memory_counter: Any
    ) -> None:
        resp = _Response(_Usage(prompt_tokens=200, completion_tokens=40, cached=150))
        record_chat_usage(resp, model="gpt-5.2", system="openai")
        by_type = _by_type(in_memory_counter)
        assert by_type["input"] == 200
        assert by_type["output"] == 40
        assert by_type["cache_read"] == 150
        assert "cache_write" not in by_type  # chat path never sets cache_write

    def test_missing_prompt_tokens_details(self, in_memory_counter: Any) -> None:
        """No prompt_tokens_details attribute -> cache_read stays 0."""
        resp = _Response(_Usage(prompt_tokens=90, completion_tokens=30))
        record_chat_usage(resp, model="m", system="openai")
        assert _by_type(in_memory_counter) == {"input": 90, "output": 30}

    def test_none_usage_is_noop(self, in_memory_counter: Any) -> None:
        """A response with usage=None records nothing but does not raise."""
        record_chat_usage(_Response(None), model="m", system="openai")
        assert _points(in_memory_counter) == []

    def test_forwards_in_child_mode_without_counter(self) -> None:
        """Regression: in a sandboxed child (_counter is None, forwarder set) the chat
        path must still forward — not early-return on the missing counter."""
        seen: list[tuple] = []
        token_metrics.set_forwarder(lambda *args: seen.append(args))
        resp = _Response(_Usage(prompt_tokens=120, completion_tokens=60, cached=40))
        record_chat_usage(resp, model="gpt-5.2", system="openai")
        assert seen == [("gpt-5.2", "openai", 120, 60, 40, 0)]


# --- record_embedding_usage() -----------------------------------------------
class _EmbUsage:
    """OpenAI embeddings usage shape: prompt_tokens + total_tokens only."""

    def __init__(self, prompt_tokens: int = 0, total_tokens: int | None = None) -> None:
        self.prompt_tokens = prompt_tokens
        self.total_tokens = prompt_tokens if total_tokens is None else total_tokens


class TestRecordEmbeddingUsage:
    """record_embedding_usage() maps an embeddings usage object onto record()."""

    def test_noop_when_uninitialized(self) -> None:
        record_embedding_usage(_Response(_EmbUsage(prompt_tokens=10)),
                               model="text-embedding-3-large", system="openai")  # no raise

    def test_records_prompt_tokens_as_input_only(self, in_memory_counter: Any) -> None:
        """Only the input bucket is emitted — no output, no cache, total ignored."""
        record_embedding_usage(_Response(_EmbUsage(prompt_tokens=42, total_tokens=42)),
                               model="text-embedding-3-large", system="openai")
        assert _by_type(in_memory_counter) == {"input": 42}

    def test_labels_are_model_and_openai_system(self, in_memory_counter: Any) -> None:
        record_embedding_usage(_Response(_EmbUsage(prompt_tokens=7)),
                               model="text-embedding-3-large", system="openai")
        attrs = _points(in_memory_counter)[0].attributes
        assert attrs[GEN_AI_REQUEST_MODEL] == "text-embedding-3-large"
        assert attrs[GEN_AI_SYSTEM] == "openai"
        assert attrs[GEN_AI_TOKEN_TYPE] == "input"

    def test_zero_prompt_tokens_is_noop(self, in_memory_counter: Any) -> None:
        record_embedding_usage(_Response(_EmbUsage(prompt_tokens=0)),
                               model="m", system="openai")
        assert _points(in_memory_counter) == []

    def test_forwards_in_child_mode_without_counter(self) -> None:
        """Regression: in a sandboxed child (_counter is None, forwarder set) the
        embedding path must forward — not early-return on the missing counter."""
        seen: list[tuple] = []
        token_metrics.set_forwarder(lambda *args: seen.append(args))
        record_embedding_usage(_Response(_EmbUsage(prompt_tokens=42)),
                               model="text-embedding-3-large", system="openai")
        assert seen == [("text-embedding-3-large", "openai", 42, 0, 0, 0)]


# --- init_token_metrics() ---------------------------------------------------
class TestInitTokenMetrics:
    """The enable/no-op decision and MeterProvider wiring."""

    def _stub_exporter(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Replace the remote_write exporter with a capture stub. Returns the
        list that captures each constructor kwargs dict."""
        captured: list[dict[str, Any]] = []
        import opentelemetry.exporter.prometheus_remote_write as rw

        class FakeExporter(MetricExporter):
            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)
                # Base init wires _preferred_temporality/_aggregation, which
                # PeriodicExportingMetricReader reads on construction.
                super().__init__()

            def export(self, *a: Any, **k: Any) -> Any:
                return None

            def shutdown(self, *a: Any, **k: Any) -> None:
                return None

            def force_flush(self, *a: Any, **k: Any) -> bool:
                return True

        monkeypatch.setattr(rw, "PrometheusRemoteWriteMetricsExporter", FakeExporter)
        return captured

    def _valid_cert_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Point CINITCCNAME at a dir with real cert.pem/key.pem files."""
        (tmp_path / "cert.pem").write_text("x")
        (tmp_path / "key.pem").write_text("x")
        monkeypatch.setenv("CINITCCNAME", str(tmp_path))

    def test_noop_when_disabled_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALPHALAB_TOKEN_METRICS_DISABLED", "1")
        init_token_metrics()
        assert token_metrics._counter is None

    def test_noop_when_endpoint_blanked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.setenv("ALPHALAB_TOKEN_METRICS_ENDPOINT", "")
        init_token_metrics()
        assert token_metrics._counter is None

    def test_noop_when_cert_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.delenv("CINITCCNAME", raising=False)
        init_token_metrics()
        assert token_metrics._counter is None

    def test_noop_when_cert_dir_has_no_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """CINITCCNAME points at a dir but cert.pem/key.pem don't exist."""
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.setenv("CINITCCNAME", str(tmp_path))
        init_token_metrics()
        assert token_metrics._counter is None

    def test_enabled_path_creates_counter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)
        init_token_metrics()
        assert token_metrics._counter is not None
        assert token_metrics._provider is not None

    def test_enabled_path_passes_tls_and_headers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.setenv("ALPHALAB_TOKEN_METRICS_ENDPOINT", "https://cortex.test/api/v1/push")
        monkeypatch.setenv("ALPHALAB_TOKEN_METRICS_TENANT", "12345-prod")
        self._valid_cert_env(monkeypatch, tmp_path)
        captured = self._stub_exporter(monkeypatch)
        init_token_metrics()
        assert len(captured) == 1
        kwargs = captured[0]
        assert kwargs["endpoint"] == "https://cortex.test/api/v1/push"
        assert kwargs["headers"] == {"X-Scope-OrgID": "12345-prod"}
        assert kwargs["tls_config"]["cert_file"] == str(tmp_path / "cert.pem")
        assert kwargs["tls_config"]["key_file"] == str(tmp_path / "key.pem")

    def test_base_attrs_populated_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-exp")
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)
        init_token_metrics()
        assert token_metrics._base_attrs["user.id"] == "alice"
        assert token_metrics._base_attrs["experiment"] == "my-exp"
        assert token_metrics._base_attrs["service.name"] == "alpha-lab"
        assert token_metrics._base_attrs["schema"] == token_metrics._SCHEMA_VERSION

    def test_experiment_defaults_to_none_string(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)
        init_token_metrics()
        assert token_metrics._base_attrs["experiment"] == "none"

    def test_run_id_uses_passed_pipeline_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The stable pipeline run_id is used verbatim as the series label."""
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)

        init_token_metrics(run_id="pipeline-2026-07-17-abc")
        assert token_metrics._base_attrs["run_id"] == "pipeline-2026-07-17-abc"

    def test_run_id_falls_back_to_unique_uuid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """With no run_id passed, each process mints a distinct uuid so two concurrent
        pipelines never collide on one series."""
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)

        init_token_metrics()
        first = token_metrics._base_attrs["run_id"]
        assert first
        # A second (concurrent) process: reset the module singletons, re-init.
        token_metrics._provider = None
        token_metrics._counter = None
        token_metrics._base_attrs = {}
        init_token_metrics()
        second = token_metrics._base_attrs["run_id"]

        assert second and second != first  # distinct identity -> no series collision

    def test_idempotent_second_call_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A second init while already initialized doesn't rebuild the exporter."""
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        self._valid_cert_env(monkeypatch, tmp_path)
        captured = self._stub_exporter(monkeypatch)
        init_token_metrics()
        first_counter = token_metrics._counter
        init_token_metrics()
        assert token_metrics._counter is first_counter
        assert len(captured) == 1  # exporter constructed exactly once

    def test_bad_interval_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A non-numeric interval env var doesn't crash init."""
        monkeypatch.delenv("ALPHALAB_TOKEN_METRICS_DISABLED", raising=False)
        monkeypatch.setenv("ALPHALAB_TOKEN_METRICS_INTERVAL_SECONDS", "not-a-number")
        self._valid_cert_env(monkeypatch, tmp_path)
        self._stub_exporter(monkeypatch)
        init_token_metrics()
        assert token_metrics._counter is not None


# --- shutdown_token_metrics() -----------------------------------------------
class TestShutdown:
    """Flush + teardown behavior."""

    def test_noop_when_never_initialized(self) -> None:
        shutdown_token_metrics()  # must not raise
        assert token_metrics._provider is None

    def test_flushes_and_clears_state(self) -> None:
        calls: list[str] = []

        class FakeProvider:
            def force_flush(self) -> None:
                calls.append("flush")

            def shutdown(self) -> None:
                calls.append("shutdown")

        token_metrics._provider = FakeProvider() 
        token_metrics._counter = object() 
        shutdown_token_metrics()
        assert calls == ["flush", "shutdown"]
        assert token_metrics._provider is None
        assert token_metrics._counter is None