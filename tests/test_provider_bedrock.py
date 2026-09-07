"""Tests for BedrockProvider.stream_response maxTokens + thinking wiring."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from alpha_lab.providers.bedrock import (
    ADAPTIVE_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    OUTPUT_HEADROOM_TOKENS,
    THINKING_BUDGETS,
    BedrockProvider,
)


def _empty_stream_response() -> dict:
    """converse_stream returns a dict; an empty ``stream`` iterable finishes
    the generator without yielding any text/tool events."""
    return {"stream": iter([])}


def _make_provider() -> BedrockProvider:
    bedrock = MagicMock()
    bedrock.converse_stream.return_value = _empty_stream_response()
    return BedrockProvider(bedrock_client=bedrock, openai_client=MagicMock())


def _capture_kwargs(
    provider: BedrockProvider, *, model: str, effort: str
) -> dict:
    """Call stream_response with a realistic Bedrock-format history, drain
    the generator, and return the kwargs that were passed to converse_stream.

    Building history via ``provider.build_user_items`` keeps the test input
    shape identical to what the agent loop produces at runtime, so the
    captured request kwargs reflect actual usage."""
    history = provider.build_user_items("hi")
    list(
        provider.stream_response(
            model=model,
            system="sys",
            history=history,
            tools=[],
            reasoning_effort=effort,
        )
    )
    call = provider._bedrock.converse_stream.call_args
    assert call is not None, "converse_stream was never called"
    return call.kwargs


# ---------------------------------------------------------------------------
# maxTokens scaling
# ---------------------------------------------------------------------------


def test_opus_47_uses_full_ceiling_regardless_of_effort() -> None:
    for effort in ("low", "medium", "high", "xhigh", "max", ""):
        provider = _make_provider()
        kwargs = _capture_kwargs(
            provider, model="us.anthropic.claude-opus-4-7-v1", effort=effort
        )
        assert kwargs["inferenceConfig"]["maxTokens"] == ADAPTIVE_MAX_TOKENS


def test_non_opus_47_high_effort_scales_max_tokens_above_budget() -> None:
    """Regression: the old fixed 8_192 default was less than
    THINKING_BUDGETS['high'] == 32_000, so every high-effort call failed
    with 'max_tokens must be greater than thinking.budget_tokens'."""
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort="high"
    )
    max_tokens = kwargs["inferenceConfig"]["maxTokens"]
    assert max_tokens == THINKING_BUDGETS["high"] + OUTPUT_HEADROOM_TOKENS
    assert max_tokens > THINKING_BUDGETS["high"]


def test_non_opus_47_medium_effort_scales_max_tokens_above_budget() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort="medium"
    )
    max_tokens = kwargs["inferenceConfig"]["maxTokens"]
    assert max_tokens == THINKING_BUDGETS["medium"] + OUTPUT_HEADROOM_TOKENS
    assert max_tokens > THINKING_BUDGETS["medium"]


def test_non_opus_47_low_effort_scales_max_tokens_above_budget() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort="low"
    )
    max_tokens = kwargs["inferenceConfig"]["maxTokens"]
    assert max_tokens == THINKING_BUDGETS["low"] + OUTPUT_HEADROOM_TOKENS


def test_non_opus_47_no_effort_uses_default_max_tokens() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort=""
    )
    assert kwargs["inferenceConfig"]["maxTokens"] == DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Thinking block wiring
# ---------------------------------------------------------------------------


def test_non_opus_47_high_effort_emits_budget_tokens_thinking_block() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort="high"
    )
    thinking = kwargs["additionalModelRequestFields"]["thinking"]
    assert thinking == {
        "type": "enabled",
        "budget_tokens": THINKING_BUDGETS["high"],
    }


def test_opus_47_high_effort_emits_adaptive_thinking_block() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="us.anthropic.claude-opus-4-7-v1", effort="high"
    )
    extras = kwargs["additionalModelRequestFields"]
    assert extras["thinking"] == {"type": "adaptive"}
    assert extras["output_config"] == {"effort": "high"}


def test_non_opus_47_xhigh_effort_skips_thinking_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: 'xhigh'/'max' aren't in THINKING_BUDGETS, so the old code
    fell through both branches and silently sent no thinking block at all.
    The fix must at least log a warning so operators notice the misconfig."""
    provider = _make_provider()
    with caplog.at_level(logging.WARNING, logger="alpha_lab.provider_bedrock"):
        kwargs = _capture_kwargs(
            provider, model="anthropic.claude-opus-4-6-v1", effort="xhigh"
        )
    assert "additionalModelRequestFields" not in kwargs
    assert any(
        "not supported on legacy (< 4.7)" in rec.message
        for rec in caplog.records
    ), f"expected a warning log; got {[r.message for r in caplog.records]}"


def test_opus_47_unknown_effort_skips_thinking_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirror of the xhigh warning path on the adaptive branch: an effort
    outside ADAPTIVE_EFFORTS (e.g. a typo) must log a warning and drop the
    thinking block rather than sending a malformed additionalModelRequestFields
    payload that Bedrock would reject."""
    provider = _make_provider()
    with caplog.at_level(logging.WARNING, logger="alpha_lab.provider_bedrock"):
        kwargs = _capture_kwargs(
            provider, model="us.anthropic.claude-opus-4-7-v1", effort="ultra"
        )
    assert "additionalModelRequestFields" not in kwargs
    assert any(
        "Unknown reasoning_effort" in rec.message and "adaptive Claude" in rec.message
        for rec in caplog.records
    ), f"expected a warning log; got {[r.message for r in caplog.records]}"


def test_empty_effort_sends_no_thinking_block() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort=""
    )
    assert "additionalModelRequestFields" not in kwargs


def test_effort_none_string_sends_no_thinking_block() -> None:
    provider = _make_provider()
    kwargs = _capture_kwargs(
        provider, model="anthropic.claude-opus-4-6-v1", effort="none"
    )
    assert "additionalModelRequestFields" not in kwargs
