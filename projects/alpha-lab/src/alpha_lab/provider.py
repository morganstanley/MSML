"""Provider protocol and normalized types for alpha-lab.

Defines a common interface that both OpenAI and Anthropic providers implement,
plus the normalized dataclasses they produce.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
    raw_output_items: list[dict[str, Any]] = field(default_factory=list)
    """Provider-native format items for history tracking."""


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """Abstract interface for LLM providers (OpenAI, Anthropic, etc.)."""

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
            Model identifier (e.g. "gpt-5.2" or "us.anthropic.claude-sonnet-4-20250514").
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
