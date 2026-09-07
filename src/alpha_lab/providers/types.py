"""Normalized wire records and the ``Provider`` interface.

Holds the leaf dataclasses every backend produces, plus the structural
``Provider`` protocol they satisfy (no inheritance required).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _plain_usage_value(value: Any) -> Any:
    """Convert usage-model values to JSON-compatible Python containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_usage_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_usage_value(item) for item in value]
    try:
        attributes = vars(value)
    except TypeError:
        return str(value)
    return {
        str(key): _plain_usage_value(item)
        for key, item in attributes.items()
        if not key.startswith("_")
    }


def serialize_usage(usage: Any) -> dict[str, Any]:
    """Serialize SDK v2, SDK v1, mapping, and simple usage objects safely."""
    for method_name in ("model_dump", "dict", "to_dict"):
        serializer = getattr(usage, method_name, None)
        if not callable(serializer):
            continue
        try:
            payload = serializer()
        except (AttributeError, TypeError, ValueError):
            continue
        if isinstance(payload, Mapping):
            return {
                str(key): _plain_usage_value(value)
                for key, value in payload.items()
            }
    if isinstance(usage, Mapping):
        return {
            str(key): _plain_usage_value(value) for key, value in usage.items()
        }
    try:
        attributes = vars(usage)
    except TypeError:
        return {}
    return {
        str(key): _plain_usage_value(value)
        for key, value in attributes.items()
        if not key.startswith("_")
    }


# ---------------------------------------------------------------------------
# Normalized types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    call_id: str
    name: str
    arguments: str  # JSON string


@dataclass
class StreamEvent:
    """A single event from a streaming response.

    type is one of:
      - "text_delta": partial text output (delta field set)
      - "done": stream finished (response field set)
    """

    type: str  # "text_delta" | "done"
    delta: str = ""
    response: Response | None = None


@dataclass
class Response:
    """A completed model response in normalized form."""

    id: str
    text: str
    tool_calls: list[ToolCall]
    has_web_search: bool
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    # Reasoning tokens, when the provider reports them separately.
    reasoning_tokens: int = 0
    # Complete provider usage payload for durable telemetry.
    usage_raw: dict[str, Any] = field(default_factory=dict)
    # Provider-native format items for history tracking.
    raw_output_items: list[dict[str, Any]] = field(default_factory=list)
    # The model that actually served this response. Useful when the request model
    # differs from what ran — e.g. LocalProvider selects a concrete deployment per
    # request from a tag pool. Empty → caller falls back to the request model.
    model: str = ""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """Abstract interface for LLM providers (OpenAI, Bedrock, etc.)."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str,
    ) -> Iterator[StreamEvent]:
        """Stream a model response, yielding StreamEvents.

        Parameters
        ----------
        model : str
            Model identifier (e.g. "gpt-5.4" or "claude-opus-4-6-v1").
        system : str
            System instructions / prompt.
        history : list[dict]
            Conversation history in provider-native format.
        tools : list[dict]
            Tool schemas in the OpenAI Responses API format (the provider
            translates them as needed).
        reasoning_effort : str
            One of "none", "low", "medium", "high".
        """
        ...

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str:
        """Simple non-streaming completion (used for summarization).

        Returns the text content of the response.
        """
        ...

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build provider-native history items for a user message."""
        ...

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build provider-native history items for tool results.

        Parameters
        ----------
        results : list[dict]
            Each dict has "call_id", "output", and optionally "name".
        images : list of (base64_data, media_type) tuples
            Images to inject alongside tool results.
        """
        ...

    def append_response_to_history(
        self,
        history: list[dict[str, Any]],
        response: Response,
    ) -> None:
        """Append a completed response to the conversation history in-place."""
        ...
