"""Shared helpers for providers speaking the OpenAI Responses stream dialect.

Used by the OpenAI and grok providers, which receive identical stream-event
and usage shapes. Lives in ``utils`` so neither provider imports generic
logic from the other; the only dependency is ``providers.types``, which is
itself a stdlib-only data module, so the leaf property of this subpackage
holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

from alpha_lab.providers.types import serialize_usage


@dataclass(frozen=True)
class ResponsesUsage:
    """Token accounting for one Responses-API call.

    All zeros / empty when the API reported no usage. Cache *writes* are
    never surfaced by this API shape, so no field for them here — callers
    supply that value themselves.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    usage_raw: dict[str, Any] = field(default_factory=dict)


def parse_responses_usage(usage: Any) -> ResponsesUsage:
    """Normalize a Responses-API usage payload into :class:`ResponsesUsage`."""
    if not usage:
        return ResponsesUsage()
    cache_read_input_tokens = 0
    if usage.input_tokens_details:
        cache_read_input_tokens = usage.input_tokens_details.cached_tokens
    reasoning_tokens = 0
    output_details = getattr(usage, "output_tokens_details", None)
    if output_details is not None:
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0
    return ResponsesUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_raw=serialize_usage(usage),
    )


def raise_for_stream_failure(event: Any, provider_label: str) -> None:
    """Raise for a terminal failure stream event; no-op for everything else.

    ``response.failed`` / ``response.incomplete`` carry their detail on
    ``event.response``; a bare ``error`` event carries ``code`` / ``message``
    on the event itself.
    """
    if event.type == "response.failed":
        detail = getattr(event.response, "error", None)
        raise RuntimeError(
            f"{provider_label} response stream failed: "
            f"{detail or 'no details supplied'}"
        )
    if event.type == "response.incomplete":
        detail = getattr(event.response, "incomplete_details", None)
        raise RuntimeError(
            f"{provider_label} response stream was incomplete: "
            f"{detail or 'no details supplied'}"
        )
    if event.type == "error":
        code = getattr(event, "code", None)
        message = getattr(event, "message", None)
        detail = ": ".join(
            str(value) for value in (code, message) if value
        )
        raise RuntimeError(
            f"{provider_label} response stream error: "
            f"{detail or 'no details supplied'}"
        )


def raise_stream_ended_early(provider_label: str) -> NoReturn:
    """The stream closed without ``response.completed`` — always an error."""
    raise RuntimeError(
        f"{provider_label} response stream ended before response.completed"
    )
