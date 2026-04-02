"""Anthropic Claude provider for alpha-lab.

Implements the Provider protocol using the Anthropic Messages API.

Key differences from OpenAI:
  - Tool schemas use Anthropic's input_schema format (not function/parameters)
  - Content is block-based: [{"type": "text", "text": ...}, {"type": "tool_use", ...}]
  - History requires strict role alternation (merged if consecutive same-role)
  - Images are base64 with media_type
  - Extended thinking via thinking parameter
  - No built-in web search (proxied through GPT via web_search tool)
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab.provider import Provider, Response, StreamEvent, ToolCall

logger = logging.getLogger("alpha_lab.provider_anthropic")

# Reasoning effort -> budget_tokens mapping
THINKING_BUDGETS = {
    "low": 5000,
    "medium": 16000,
    "high": 32000,
}

# Web search tool schema in OpenAI format (matched in _translate_tools)
WEB_SEARCH_OPENAI_TYPE = "web_search_preview"


class AnthropicProvider:
    """Provider backed by the Anthropic Messages API."""

    def __init__(self, anthropic_client: Any, openai_client: OpenAI) -> None:
        self._anthropic = anthropic_client
        self._openai = openai_client

    @property
    def openai_client(self) -> OpenAI:
        """Expose the OpenAI client (used for web search proxy and summarization)."""
        return self._openai

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
        """Stream a response via Anthropic Messages API."""
        # Translate tool schemas from OpenAI format to Anthropic format
        anthropic_tools = self._translate_tools(tools)

        # Normalize history (merge consecutive same-role messages)
        messages = self._normalize_history(history)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": 16384,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # Extended thinking
        if reasoning_effort in THINKING_BUDGETS:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": THINKING_BUDGETS[reasoning_effort],
            }
            # With thinking enabled, need higher max_tokens
            kwargs["max_tokens"] = THINKING_BUDGETS[reasoning_effort] + 16384

        # Stream the response
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_assistant_content: list[dict[str, Any]] = []
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        tool_input_chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0

        with self._anthropic.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        tool_input_chunks = []

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_chunks.append(delta.text)
                        yield StreamEvent(type="text_delta", delta=delta.text)
                    elif delta.type == "input_json_delta":
                        tool_input_chunks.append(delta.partial_json)
                    elif delta.type == "thinking_delta":
                        pass  # Extended thinking — track but don't stream

                elif event.type == "content_block_stop":
                    if current_tool_id is not None:
                        full_input_str = "".join(tool_input_chunks)
                        try:
                            tool_input = json.loads(full_input_str)
                        except json.JSONDecodeError:
                            tool_input = {}

                        tool_calls.append(ToolCall(
                            call_id=current_tool_id,
                            name=current_tool_name or "",
                            arguments=json.dumps(tool_input),
                        ))
                        raw_assistant_content.append({
                            "type": "tool_use",
                            "id": current_tool_id,
                            "name": current_tool_name or "",
                            "input": tool_input,
                        })
                        current_tool_id = None
                        current_tool_name = None
                        tool_input_chunks = []

                elif event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = getattr(event.message.usage, "input_tokens", 0)

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = getattr(event.usage, "output_tokens", 0)

        # Build final text
        full_text = "".join(text_chunks)
        if full_text:
            raw_assistant_content.insert(0, {"type": "text", "text": full_text})

        response_obj = Response(
            id=f"anthropic-{id(stream)}",
            text=full_text,
            tool_calls=tool_calls,
            has_web_search=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_output_items=raw_assistant_content,
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

        Uses the OpenAI client for summarization (cheap, GPT-based) to avoid
        consuming Claude quota on utility tasks.
        """
        chat_messages = [{"role": "system", "content": system}] + messages
        response = self._openai.chat.completions.create(
            model=model,
            messages=chat_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build Anthropic-format user message."""
        return [{"role": "user", "content": [{"type": "text", "text": message}]}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build Anthropic-format tool result items.

        All tool results are packed into a single user-role message
        (Anthropic requires strict role alternation).
        """
        content_blocks: list[dict[str, Any]] = []

        for r in results:
            content_blocks.append({
                "type": "tool_result",
                "tool_use_id": r["call_id"],
                "content": r["output"],
            })

        # Inject images
        if images:
            for b64_data, media_type in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            content_blocks.append({
                "type": "text",
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
    ) -> list[dict[str, Any]]:
        """Translate OpenAI-format tool schemas to Anthropic format.

        OpenAI format:
            {"type": "function", "name": "...", "parameters": {...}}
            {"type": "web_search_preview"}

        Anthropic format:
            {"name": "...", "description": "...", "input_schema": {...}}
        """
        anthropic_tools: list[dict[str, Any]] = []

        for tool in tools:
            tool_type = tool.get("type", "")

            if tool_type == WEB_SEARCH_OPENAI_TYPE:
                # Replace built-in web search with a custom function tool
                anthropic_tools.append({
                    "name": "web_search",
                    "description": (
                        "Search the web for current information. "
                        "Returns search results as text."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            },
                        },
                        "required": ["query"],
                    },
                })
            elif tool_type == "function":
                anthropic_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                })

        return anthropic_tools

    # ------------------------------------------------------------------
    # History normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_history(
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages (Anthropic requires strict alternation).

        Also ensures all messages have the Anthropic content block format.
        """
        if not history:
            return []

        normalized: list[dict[str, Any]] = []

        for item in history:
            role = item.get("role", "")
            content = item.get("content", [])

            # Ensure content is in block format
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_content = []
                for block in content:
                    if isinstance(block, str):
                        normalized_content.append({"type": "text", "text": block})
                    else:
                        normalized_content.append(block)
                content = normalized_content

            if normalized and normalized[-1].get("role") == role:
                # Merge into previous message
                prev_content = normalized[-1].get("content", [])
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                prev_content.extend(content)
                normalized[-1]["content"] = prev_content
            else:
                normalized.append({"role": role, "content": content})

        return normalized
