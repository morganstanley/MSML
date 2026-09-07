"""Responses API stream integrity and usage telemetry tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from alpha_lab.providers.grok import GrokProvider
from alpha_lab.providers.openai import OpenAIProvider
from alpha_lab.providers.types import serialize_usage


class _Stream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self) -> None:
        self.closed = True


PROVIDER_CASES = (
    ("openai", "OpenAI", "alpha_lab.providers.openai.token_metrics.record"),
    ("grok", "Grok", "alpha_lab.providers.grok.token_metrics.record"),
)


class _LegacyUsage:
    def dict(self) -> dict[str, object]:
        return {
            "input_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
        }


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            SimpleNamespace(
                input_tokens=7,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
            {
                "input_tokens": 7,
                "input_tokens_details": {"cached_tokens": 3},
            },
        ),
        (
            _LegacyUsage(),
            {
                "input_tokens": 7,
                "input_tokens_details": {"cached_tokens": 3},
            },
        ),
        (
            {"input_tokens": 7, "details": SimpleNamespace(cached_tokens=3)},
            {"input_tokens": 7, "details": {"cached_tokens": 3}},
        ),
    ],
)
def test_serialize_usage_supports_sdk_and_mapping_shapes(
    usage: object, expected: dict[str, object],
) -> None:
    assert serialize_usage(usage) == expected


def _provider(name: str, client: MagicMock):
    if name == "openai":
        return OpenAIProvider(client)
    return GrokProvider(client, MagicMock())


@pytest.mark.parametrize(("name", "label", "_metrics_target"), PROVIDER_CASES)
def test_stream_raises_when_terminal_event_is_missing(
    name: str, label: str, _metrics_target: str,
) -> None:
    stream = _Stream([
        SimpleNamespace(type="response.output_text.delta", delta="partial"),
    ])
    client = MagicMock()
    client.responses.create.return_value = stream

    with pytest.raises(
        RuntimeError, match=rf"{label} response stream ended before response\.completed"
    ):
        list(_provider(name, client).stream_response(
            model="model",
            system="system",
            history=[],
            tools=[],
            reasoning_effort="high",
        ))

    assert stream.closed is True


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(error="upstream failure"),
            ),
            "response stream failed: upstream failure",
        ),
        (
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(incomplete_details="max output tokens"),
            ),
            "response stream was incomplete: max output tokens",
        ),
        (
            SimpleNamespace(
                type="error", code="server_error", message="connection closed",
            ),
            "response stream error: server_error: connection closed",
        ),
    ],
)
@pytest.mark.parametrize(("name", "_label", "_metrics_target"), PROVIDER_CASES)
def test_stream_surfaces_terminal_failure_details(
    name: str,
    _label: str,
    _metrics_target: str,
    event: SimpleNamespace,
    message: str,
) -> None:
    stream = _Stream([event])
    client = MagicMock()
    client.responses.create.return_value = stream

    with pytest.raises(RuntimeError, match=message):
        list(_provider(name, client).stream_response(
            model="model",
            system="system",
            history=[],
            tools=[],
            reasoning_effort="high",
        ))

    assert stream.closed is True


@pytest.mark.parametrize(("name", "_label", "metrics_target"), PROVIDER_CASES)
def test_stream_preserves_complete_usage_payload(
    name: str, _label: str, metrics_target: str,
) -> None:
    usage_payload = {
        "input_tokens": 90,
        "output_tokens": 30,
        "input_tokens_details": {"cached_tokens": 60},
        "output_tokens_details": {"reasoning_tokens": 20},
    }
    usage = SimpleNamespace(
        input_tokens=90,
        output_tokens=30,
        input_tokens_details=SimpleNamespace(cached_tokens=60),
        output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        model_dump=lambda: usage_payload,
    )
    raw_response = SimpleNamespace(id="resp-usage", output=[], usage=usage)
    client = MagicMock()
    client.responses.create.return_value = _Stream([
        SimpleNamespace(type="response.completed", response=raw_response),
    ])

    with patch(metrics_target):
        events = list(_provider(name, client).stream_response(
            model="model",
            system="system",
            history=[],
            tools=[],
            reasoning_effort="high",
        ))

    response = events[-1].response
    assert response is not None
    assert response.cache_read_input_tokens == 60
    assert response.reasoning_tokens == 20
    assert response.usage_raw == usage_payload


def test_openai_stream_falls_back_to_emitted_text_deltas() -> None:
    raw_response = SimpleNamespace(id="resp-text", usage=None)
    client = MagicMock()
    client.responses.create.return_value = _Stream([
        SimpleNamespace(type="response.output_text.delta", delta="visible text"),
        SimpleNamespace(type="response.completed", response=raw_response),
    ])

    with patch("alpha_lab.providers.openai.token_metrics.record"):
        events = list(OpenAIProvider(client).stream_response(
            model="model",
            system="system",
            history=[],
            tools=[],
            reasoning_effort="high",
        ))

    response = events[-1].response
    assert response is not None
    assert response.text == "visible text"
