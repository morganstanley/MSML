"""AWS Bedrock Converse API provider for alpha-lab.

Implements the Provider protocol using the Bedrock Converse / ConverseStream
API through the MS AI Gateway with Bearer token auth.

Key differences from OpenAI:
  - Tool schemas use toolSpec format (not function/parameters)
  - Content is block-based: [{"text": ...}, {"toolUse": ...}]
  - History requires strict role alternation (merged if consecutive same-role)
  - Images are raw bytes, not base64 data URIs
  - Extended thinking via additionalModelRequestFields
  - No built-in web search (proxied through GPT via web_search tool)
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab import token_metrics
from alpha_lab.providers.types import Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.clients import get_bedrock_client, get_openai_client

logger = logging.getLogger("alpha_lab.provider_bedrock")

# Reasoning effort -> budget_tokens mapping (used for legacy Claudes, < 4.7,
# where adaptive thinking isn't supported). Claude >= 4.7 takes the effort
# string directly via "thinking": {"type": "adaptive"}.
THINKING_BUDGETS = {
    "low": 5000,
    "medium": 16000,
    "high": 32000,
}

# Valid reasoning_effort values for adaptive thinking (Claude >= 4.7).
ADAPTIVE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# Baseline maxTokens when thinking is disabled or unavailable. Older Bedrock
# Claude revisions (4.6 and below) reject larger values with a validation
# error; 8_192 is conservative and accepted by all current Bedrock Claudes.
DEFAULT_MAX_TOKENS = 8_192

# maxTokens for adaptive-thinking Claudes (Claude >= 4.7, model-supported ceiling).
ADAPTIVE_MAX_TOKENS = 64_000

# Visible-output headroom added on top of any explicit thinking budget.
# Bedrock requires maxTokens > budget_tokens, and maxTokens caps thinking +
# output combined — so we scale maxTokens with the budget to preserve room
# for the actual response.
OUTPUT_HEADROOM_TOKENS = 16_384

# Web search tool schema in OpenAI format (matched in _translate_tools)
WEB_SEARCH_OPENAI_TYPE = "web_search"

# Claude family version (major, minor) at which Bedrock switched from the legacy
# ``thinking.budget_tokens`` path to adaptive thinking + ``output_config.effort``.
_CLAUDE_ADAPTIVE_MIN_VERSION = (4, 7)

# Extracts the Claude family version from a Bedrock model id, e.g.
# ``anthropic.claude-opus-4-6-v1``, ``us.anthropic.claude-opus-4-7-v1``,
# ``claude-opus-4-8`` -> ("4", "6"/"7"/"8").
_CLAUDE_VERSION_RE = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)-(\d+)")


def _supports_adaptive(model: str) -> bool:
    """Return True if ``model`` uses adaptive thinking rather than legacy budgets.

    Claude on Bedrock switched thinking APIs at 4.7: ``>= 4.7`` takes
    ``thinking={"type": "adaptive"}`` + ``output_config.effort``, while older
    revisions require the legacy ``thinking={"type": "enabled", "budget_tokens"}``
    path (which Opus 4.8 rejects with a 400). Raises ``ValueError`` on any id
    without a parseable Claude version so a misrouted model fails loudly rather
    than guessing the wrong thinking args.
    """
    match = _CLAUDE_VERSION_RE.search(model)
    if match is None:
        raise ValueError(
            f"Cannot determine Claude version from Bedrock model id {model!r}; "
            "expected a 'claude-<opus|sonnet|haiku>-<major>-<minor>' id."
        )
    version = (int(match.group(1)), int(match.group(2)))
    return version >= _CLAUDE_ADAPTIVE_MIN_VERSION


class BedrockProvider:
    """Provider backed by the AWS Bedrock Converse API."""

    @property
    def openai_client(self) -> OpenAI:
        """The OpenAI client used for the web_search proxy / summarization."""
        return self._openai_client

    @classmethod
    def from_config(
        cls, api_key: str | None = None, model: str = ""
    ) -> BedrockProvider:
        """Build a BedrockProvider from run config (proxy client backs web_search)."""
        return cls(bedrock_client=get_bedrock_client(), openai_client=get_openai_client(api_key))

    def __init__(self, bedrock_client: Any, openai_client: OpenAI) -> None:
        self._bedrock = bedrock_client
        self._openai_client = openai_client

    # ------------------------------------------------------------------
    # stream_response
    # ------------------------------------------------------------------

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str,
    ) -> Iterator[StreamEvent]:
        """Stream a response via Bedrock ConverseStream."""
        # Translate tool schemas from OpenAI format to Bedrock format
        tool_config = self._translate_tools(tools)

        # Normalize history (merge consecutive same-role messages)
        messages = self._normalize_history(history)

        # Build request kwargs.
        #
        # Bedrock Converse prefix-caching requires explicit ``cachePoint``
        # markers at the end of the static prefix (system prompt and tool
        # list). Without these, every request re-ingests the full prefix.
        # See https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        #
        # Normalize effort up-front so maxTokens and the thinking block stay
        # in sync. Bedrock rejects effort="none"/""; treat those as disabled.
        # Lower-case so "High"/"MED"/etc. don't leak through to Bedrock.
        effort_norm = reasoning_effort.strip().lower() if reasoning_effort else ""

        # maxTokens: adaptive-thinking Claudes (>= 4.7) support up to 64_000
        # output tokens; older Bedrock Claude revisions (4.6 and below) cap much
        # lower and reject 64_000 with a validation error. When extended thinking
        # is enabled on a legacy model, Bedrock also requires
        # maxTokens > budget_tokens — otherwise ConverseStream fails with
        # "max_tokens must be greater than thinking.budget_tokens". The old fixed
        # 8_192 was less than budget_tokens for medium (16_000) and high (32_000),
        # so every medium/high-effort call was failing. Scale maxTokens with the
        # requested budget instead, leaving fixed headroom for visible output.
        is_adaptive = _supports_adaptive(model)
        if is_adaptive:
            max_tokens = ADAPTIVE_MAX_TOKENS
        elif effort_norm in THINKING_BUDGETS:
            max_tokens = THINKING_BUDGETS[effort_norm] + OUTPUT_HEADROOM_TOKENS
        else:
            max_tokens = DEFAULT_MAX_TOKENS
        kwargs: dict[str, Any] = {
            "modelId": model,
            "system": [
                {"text": system},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ],
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if tool_config:
            cached_tools = list(tool_config.get("tools", []))
            cached_tools.append({"cachePoint": {"type": "default", "ttl": "1h"}})
            kwargs["toolConfig"] = {**tool_config, "tools": cached_tools}

        # Extended thinking
        if effort_norm and effort_norm != "none":
            if is_adaptive:
                if effort_norm in ADAPTIVE_EFFORTS:
                    kwargs["additionalModelRequestFields"] = {
                        "thinking": {"type": "adaptive"},
                        "output_config": {"effort": effort_norm},
                    }
                else:
                    logger.warning(
                        "Unknown reasoning_effort %r for adaptive Claude; skipping "
                        "thinking (expected one of %s)",
                        reasoning_effort, sorted(ADAPTIVE_EFFORTS),
                    )
            elif effort_norm in THINKING_BUDGETS:
                kwargs["additionalModelRequestFields"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": THINKING_BUDGETS[effort_norm],
                    }
                }
            else:
                # e.g., reasoning_effort="xhigh"/"max" against a legacy (< 4.7)
                # Claude — THINKING_BUDGETS has no mapping for those tiers.
                # Warn instead of silently dropping thinking.
                logger.warning(
                    "reasoning_effort %r not supported on legacy (< 4.7) Claude; "
                    "skipping thinking (expected one of %s)",
                    reasoning_effort, sorted(THINKING_BUDGETS),
                )

        response = self._bedrock.converse_stream(**kwargs)

        # Parse the stream
        text_chunks: list[str] = []
        current_tool_use_id: str | None = None
        current_tool_name: str | None = None
        tool_input_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_assistant_content: list[dict[str, Any]] = []
        stop_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_write_input_tokens = 0

        # Track content blocks for raw output
        current_block_type: str | None = None

        for event in response.get("stream"):
            if "messageStart" in event:
                pass

            elif "contentBlockStart" in event:
                start = event["contentBlockStart"]
                start_data = start.get("start", {})
                if "toolUse" in start_data:
                    current_tool_use_id = start_data["toolUse"]["toolUseId"]
                    current_tool_name = start_data["toolUse"]["name"]
                    tool_input_chunks = []
                    current_block_type = "toolUse"
                else:
                    current_block_type = "text"

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    text_chunks.append(delta["text"])
                    yield StreamEvent(type="text_delta", delta=delta["text"])
                elif "toolUse" in delta:
                    tool_input_chunks.append(delta["toolUse"]["input"])
                elif "reasoningContent" in delta:
                    # Extended thinking content — track but don't stream
                    pass

            elif "contentBlockStop" in event:
                if current_block_type == "toolUse" and current_tool_use_id:
                    # Parse accumulated tool input
                    full_input_str = "".join(tool_input_chunks)
                    try:
                        tool_input = json.loads(full_input_str)
                    except json.JSONDecodeError:
                        tool_input = {}

                    tool_calls.append(ToolCall(
                        call_id=current_tool_use_id,
                        name=current_tool_name or "",
                        arguments=json.dumps(tool_input),
                    ))
                    raw_assistant_content.append({
                        "toolUse": {
                            "toolUseId": current_tool_use_id,
                            "name": current_tool_name or "",
                            "input": tool_input,
                        }
                    })
                    current_tool_use_id = None
                    current_tool_name = None
                    tool_input_chunks = []

                elif current_block_type == "text" and text_chunks:
                    # Accumulate text block into raw content
                    # (text_chunks may have content from previous text blocks too,
                    # but we only add the text block on stop)
                    pass

                current_block_type = None

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
                cache_read_input_tokens = usage.get("cacheReadInputTokens", 0)
                cache_write_input_tokens = usage.get("cacheWriteInputTokens", 0)

        # Build final text
        full_text = "".join(text_chunks)
        if full_text:
            raw_assistant_content.insert(0, {"text": full_text})

        response_obj = Response(
            id=f"bedrock-{id(response)}",
            text=full_text,
            tool_calls=tool_calls,
            has_web_search=False,  # Bedrock doesn't have built-in web search
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            raw_output_items=raw_assistant_content,
        )

        # Token-usage metric
        token_metrics.record(
            input_tokens,
            output_tokens,
            cache_read=cache_read_input_tokens,
            cache_write=cache_write_input_tokens,
            model=model,
            system="aws.bedrock",
        )

        yield StreamEvent(type="done", response=response_obj)

    # ------------------------------------------------------------------
    # complete (non-streaming, for summarization)
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str:
        """Non-streaming completion.

        Routes to Bedrock ``converse()`` when the model looks like a
        Bedrock model ID, otherwise falls back to the OpenAI client.
        """
        if any(model.startswith(p) for p in ("anthropic.", "us.", "claude")):
            return self._complete_bedrock(model, system, messages, max_tokens)
        return self._complete_openai(model, system, messages, max_tokens)

    def _complete_openai(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Completion via the OpenAI client (for GPT models)."""
        chat_messages = [{"role": "system", "content": system}] + messages
        response = self._openai_client.chat.completions.create(
            model=model,
            messages=chat_messages,
            max_tokens=max_tokens,
        )
        token_metrics.record_chat_usage(response, model=model, system="openai")
        return response.choices[0].message.content or ""

    def _complete_bedrock(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Completion via Bedrock ``converse()`` (for Claude models)."""
        # Translate messages to Bedrock format
        bedrock_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                bedrock_messages.append({"role": role, "content": [{"text": content}]})
            else:
                bedrock_messages.append({"role": role, "content": content})

        response = self._bedrock.converse(
            modelId=model,
            system=[
                {"text": system},
                {"cachePoint": {"type": "default", "ttl": "1h"}},
            ],
            messages=bedrock_messages,
            inferenceConfig={"maxTokens": max_tokens},
        )
        usage = response.get("usage", {})
        token_metrics.record(
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            cache_read=usage.get("cacheReadInputTokens", 0),
            cache_write=usage.get("cacheWriteInputTokens", 0),
            model=model,
            system="aws.bedrock",
        )
        # Extract text from response
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        text_parts = [b["text"] for b in content_blocks if "text" in b]
        return "".join(text_parts)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build Bedrock-format user message."""
        return [{"role": "user", "content": [{"text": message}]}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build Bedrock-format tool result items.

        All tool results are packed into a single user-role message
        (Bedrock requires strict role alternation).
        """
        content_blocks: list[dict[str, Any]] = []

        for r in results:
            content_blocks.append({
                "toolResult": {
                    "toolUseId": r["call_id"],
                    "content": [{"text": r["output"]}],
                }
            })

        # Inject images
        if images:
            for b64_data, media_type in images:
                # Bedrock expects raw bytes, not base64
                image_bytes = base64.b64decode(b64_data)
                # Extract format from media type (e.g. "image/png" -> "png")
                fmt = media_type.split("/")[-1]
                if fmt == "jpeg":
                    fmt = "jpeg"
                content_blocks.append({
                    "image": {
                        "format": fmt,
                        "source": {"bytes": image_bytes},
                    }
                })
            content_blocks.append({
                "text": "Here is the image you requested. Analyze it carefully.",
            })

        return [{"role": "user", "content": content_blocks}]

    def append_response_to_history(
        self,
        history: list[dict[str, Any]],
        response: Response,
    ) -> None:
        """Append the assistant response to conversation history."""
        if response.raw_output_items:
            history.append({
                "role": "assistant",
                "content": response.raw_output_items,
            })

    # ------------------------------------------------------------------
    # Tool schema translation
    # ------------------------------------------------------------------

    def _translate_tools(
        self, tools: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Translate OpenAI-format tool schemas to Bedrock toolConfig.

        OpenAI format:
            {"type": "function", "name": "...", "parameters": {...}}
            {"type": "web_search"}

        Bedrock format:
            {"tools": [{"toolSpec": {"name": "...", "inputSchema": {"json": {...}}}}]}
        """
        bedrock_tools: list[dict[str, Any]] = []

        for tool in tools:
            tool_type = tool.get("type", "")

            if tool_type == WEB_SEARCH_OPENAI_TYPE:
                # Replace built-in web search with a custom function tool
                bedrock_tools.append({
                    "toolSpec": {
                        "name": "web_search",
                        "description": (
                            "Search the web for current information. "
                            "Returns search results as text."
                        ),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query.",
                                    },
                                },
                                "required": ["query"],
                            }
                        },
                    }
                })
            elif tool_type == "function":
                bedrock_tools.append({
                    "toolSpec": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "inputSchema": {
                            "json": tool.get("parameters", {}),
                        },
                    }
                })

        if not bedrock_tools:
            return None

        return {"tools": bedrock_tools}

    # ------------------------------------------------------------------
    # History normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_history(
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages (Bedrock requires strict alternation).

        Also ensures all messages have the Bedrock content block format.
        """
        if not history:
            return []

        normalized: list[dict[str, Any]] = []

        for item in history:
            role = item.get("role", "")
            content = item.get("content", [])

            # Ensure content is in block format
            if isinstance(content, str):
                content = [{"text": content}]
            elif isinstance(content, list):
                # Already block format — validate
                normalized_content = []
                for block in content:
                    if isinstance(block, str):
                        normalized_content.append({"text": block})
                    else:
                        normalized_content.append(block)
                content = normalized_content

            if normalized and normalized[-1].get("role") == role:
                # Merge into previous message
                prev_content = normalized[-1].get("content", [])
                if isinstance(prev_content, str):
                    prev_content = [{"text": prev_content}]
                prev_content.extend(content)
                normalized[-1]["content"] = prev_content
            else:
                normalized.append({"role": role, "content": content})

        return normalized
