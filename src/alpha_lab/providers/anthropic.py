"""Native Anthropic Messages API provider for alpha-lab.

Implements the Provider protocol against the Anthropic Messages API served by
the MS AI Gateway (``.../anthropic/v1/messages``), rather than through Bedrock
Converse. This is the correct path for current-generation Claude models: every
Claude the gateway serves natively (Opus 4.6/4.7/4.8, Sonnet 4.6) uses
``thinking: {"type": "adaptive"}`` + ``output_config.effort``, so there is no
per-model thinking-param routing to get wrong — the Bedrock backend's legacy
``thinking.type.enabled`` / ``budget_tokens`` path (which Opus 4.8 rejects with a
400) simply doesn't apply here.

Key points, mirroring ``bedrock`` where it makes sense:
  - History is kept in native Anthropic Messages format: ``{"role", "content": [blocks]}``.
  - Tool schemas are translated from the OpenAI Responses format the rest of the
    system emits into Anthropic ``tools`` (flat ``name``/``description``/``input_schema``).
  - No built-in web search — ``web_search`` is proxied through the OpenAI
    client via a ``web_search`` function tool (same pattern as Bedrock).
  - Extended thinking is adaptive: ``reasoning_effort`` maps straight to
    ``output_config.effort``; ``none``/empty disables thinking.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab import token_metrics
from alpha_lab.providers.types import Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.clients import get_anthropic_client, get_openai_client

logger = logging.getLogger("alpha_lab.provider_anthropic")

# Streaming output ceiling. Native Claude models support up to 128k output, but
# 64k leaves ample room for adaptive thinking + response while staying well
# under the cap; streaming means there's no HTTP-timeout concern.
DEFAULT_MAX_TOKENS = 64_000

# reasoning_effort values accepted by output_config.effort (adaptive thinking).
ADAPTIVE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

# Web search tool schema in the OpenAI Responses format (matched in _translate_tools)
WEB_SEARCH_OPENAI_TYPE = "web_search"

# Name the proxied search tool travels under on the wire. The gateway rejects
# a Messages-API tool called "web_search" outright -- that name is reserved for
# Anthropic's own hosted tool, and only caller-defined tools are permitted:
#   "Server/hosted Anthropic tool name 'web_search' is not currently allowed
#    for messages API calls."
# It is sent under this name and mapped back before the agent sees it, so tool
# dispatch is unchanged.
PROXIED_WEB_SEARCH_NAME = "search_web"


# 1h TTL cache breakpoint, applied to the system prompt, the last tool, and a
# moving breakpoint on the last history block (see _mark_history_cache_point).
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


def _prompt_caching_enabled() -> bool:
    """Prompt caching is on by default; ``ANTHROPIC_PROMPT_CACHING=0`` disables it.

    The off switch exists for strict zero-retention deployments: a
    ``cache_control`` marker asks the service to retain the cached prompt
    prefix server-side until the TTL expires. Caching is a billing mechanism,
    not conversation storage — history is tracked and resent locally on every
    call either way — but a policy that forbids any server-side retention can
    set the variable to send no cache markers at all.
    """
    return os.environ.get("ANTHROPIC_PROMPT_CACHING", "").lower() not in (
        "0", "false", "no",
    )


_CACHEABLE_BLOCK_TYPES = ("text", "image", "tool_use", "tool_result", "document")


def _mark_history_cache_point(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add a moving cache breakpoint to the last eligible block of history.

    Placing the breakpoint on the growing end of history means each call
    only pays the write rate for its new suffix; the API matches the
    unchanged prefix against the previous call's write and reads it at the
    cheaper cache-read rate. Uses 1 of the API's 4 breakpoints (system and
    the last tool use the other 2).

    Returns a new list; never mutates ``messages`` or its blocks in place —
    the caller reuses the same history list across turns, and a marker left
    on an old block would accumulate and exceed the breakpoint limit. Scans
    backward for the nearest block whose type supports ``cache_control``
    (``thinking``/``redacted_thinking`` don't).
    """
    if not messages or not messages[-1].get("content"):
        return messages

    last = messages[-1]
    content = last["content"]
    for i, block in reversed(tuple(enumerate(content))):
        if isinstance(block, dict) and block.get("type") in _CACHEABLE_BLOCK_TYPES:
            last = {**last, "content": [*content[:i], {**block, "cache_control": CACHE_CONTROL}, *content[i + 1:]]}
            break

    return [*messages[:-1], last]


def _strip_orphan_tool_results(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop tool_result blocks whose tool_use is no longer in the history.

    A tool_result must sit in the message directly after the assistant turn
    that requested it. History trimming and summarization can drop that
    assistant turn while keeping the result, and the API rejects the whole
    request with "unexpected tool_use_id found in tool_result blocks" -- one
    such rejection ended a Phase 1 agent mid-run. Orphans are dropped here so
    a trimmed history stays sendable.
    """
    cleaned: list[dict[str, Any]] = []
    for i, msg in enumerate(history):
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            cleaned.append(msg)
            continue
        prev = cleaned[-1] if cleaned else None
        allowed: set[str] = set()
        if prev is not None and prev.get("role") == "assistant":
            allowed = {b["id"] for b in prev.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "tool_use"
                       and isinstance(b.get("id"), str)}
        kept = [b for b in content
                if not (isinstance(b, dict) and b.get("type") == "tool_result"
                        and b.get("tool_use_id") not in allowed)]
        if len(kept) == len(content):
            # Nothing dropped — keep the original message object rather than
            # allocating a copy per request.
            cleaned.append(msg)
            continue
        logger.warning(
            "Dropped %d orphaned tool_result block(s) at message %d "
            "(their tool_use turn is no longer in history)",
            len(content) - len(kept), i,
        )
        if kept:
            cleaned.append({**msg, "content": kept})
    return cleaned


class AnthropicProvider:
    """Provider backed by the native Anthropic Messages API (via the gateway)."""

    @property
    def openai_client(self) -> OpenAI:
        """The OpenAI client used for the web_search proxy / summarization."""
        return self._openai_client

    @classmethod
    def from_config(
        cls, api_key: str | None = None, model: str = ""
    ) -> AnthropicProvider:
        """Build an AnthropicProvider from run config.

        ``api_key`` is the Anthropic key; the web_search proxy always uses
        OpenAI credentials, so its client is built from OpenAI env/token.
        """
        return cls(client=get_anthropic_client(api_key), openai_client=get_openai_client())

    def __init__(self, client: Any, openai_client: OpenAI) -> None:
        # ``client`` is an ``anthropic.Anthropic`` instance (typed ``Any`` so this
        # module imports cleanly even when the anthropic SDK isn't installed —
        # it's constructed lazily in get_anthropic_client()).
        self._client = client
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
        """Stream a response via the Anthropic Messages API."""
        history = _strip_orphan_tool_results(history)
        caching = _prompt_caching_enabled()
        kwargs: dict[str, Any] = {
            "model": model,
            "system": (
                [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}]
                if caching
                else system
            ),
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": _mark_history_cache_point(history) if caching else history,
        }

        anthropic_tools = self._translate_tools(tools)
        if anthropic_tools:
            if caching:
                anthropic_tools[-1] = {**anthropic_tools[-1], "cache_control": CACHE_CONTROL}
            kwargs["tools"] = anthropic_tools

        # Adaptive thinking. Every natively-served Claude uses adaptive + effort;
        # "none"/empty disables thinking. Unknown tiers warn rather than 400.
        effort_norm = reasoning_effort.strip().lower() if reasoning_effort else ""
        if effort_norm and effort_norm != "none":
            if effort_norm in ADAPTIVE_EFFORTS:
                # ``output_config`` postdates the installed SDK's typed
                # signature, which rejects it as an unexpected keyword; the
                # passthrough puts it on the wire regardless of SDK version.
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["extra_body"] = {"output_config": {"effort": effort_norm}}
            else:
                logger.warning(
                    "Unknown reasoning_effort %r; skipping thinking "
                    "(expected one of %s)",
                    reasoning_effort, sorted(ADAPTIVE_EFFORTS),
                )

        text_chunks: list[str] = []
        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_delta" and getattr(
                    event.delta, "type", None
                ) == "text_delta":
                    text_chunks.append(event.delta.text)
                    yield StreamEvent(type="text_delta", delta=event.delta.text)
            final = stream.get_final_message()

        # Rebuild tool calls and native-format history items from the final message.
        tool_calls: list[ToolCall] = []
        raw_output_items: list[dict[str, Any]] = []
        for block in final.content:
            raw_output_items.append(self._block_to_param(block))
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    call_id=block.id,
                    name=(WEB_SEARCH_OPENAI_TYPE
                          if block.name == PROXIED_WEB_SEARCH_NAME
                          else block.name),
                    arguments=json.dumps(block.input),
                ))

        full_text = "".join(text_chunks)
        usage = final.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        token_metrics.record(
            input_tokens,
            output_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
            model=model,
            system="anthropic",
        )

        yield StreamEvent(
            type="done",
            response=Response(
                id=getattr(final, "id", "") or "",
                text=full_text,
                tool_calls=tool_calls,
                has_web_search=False,  # web search is proxied through OpenAI
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read,
                cache_write_input_tokens=cache_write,
                raw_output_items=raw_output_items,
            ),
        )

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

        Routes Claude model ids to the Anthropic Messages API; anything else
        (e.g. a GPT summarizer) falls back to the OpenAI client, matching the
        Bedrock backend's behavior.
        """
        if model.startswith(("claude-", "anthropic.", "us.anthropic.")):
            return self._complete_anthropic(model, system, messages, max_tokens)
        return self._complete_openai(model, system, messages, max_tokens)

    def _complete_anthropic(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Completion via the Anthropic Messages API (for Claude models)."""
        response = self._client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            messages=messages,
        )
        usage = response.usage
        token_metrics.record(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=model,
            system="anthropic",
        )
        return "".join(b.text for b in response.content if b.type == "text")

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

    # ------------------------------------------------------------------
    # History helpers — native Anthropic Messages format
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build an Anthropic-format user message."""
        return [{"role": "user", "content": [{"type": "text", "text": message}]}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build Anthropic-format tool-result items.

        All results are packed into a single ``user`` message (Anthropic pairs
        ``tool_result`` blocks with the preceding assistant ``tool_use`` turn).
        """
        content: list[dict[str, Any]] = []
        for r in results:
            content.append({
                "type": "tool_result",
                "tool_use_id": r["call_id"],
                "content": r["output"],
            })

        if images:
            for b64_data, media_type in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
            content.append({
                "type": "text",
                "text": "Here is the image you requested. Analyze it carefully.",
            })

        return [{"role": "user", "content": content}]

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_to_param(block: Any) -> dict[str, Any]:
        """Convert a response content block into an input-param dict for history.

        Reconstructs the common block types explicitly (so replay to the same
        model is faithful — thinking blocks keep their signature); falls back to
        ``model_dump()`` for anything unrecognized.
        """
        btype = block.type
        if btype == "text":
            return {"type": "text", "text": block.text}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        if btype == "thinking":
            return {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
        if btype == "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.data}
        return block.model_dump()

    def _translate_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Translate OpenAI Responses-format tool schemas to Anthropic ``tools``.

        OpenAI format:
            {"type": "function", "name": "...", "description": "...", "parameters": {...}}
            {"type": "web_search"}

        Anthropic format:
            {"name": "...", "description": "...", "input_schema": {...}}

        The built-in ``web_search`` tool is replaced with a function tool named
        ``search_web`` on the wire (``PROXIED_WEB_SEARCH_NAME`` — the gateway
        reserves the literal name ``web_search``); the tool_use is mapped back
        to ``web_search`` before the agent sees it, and the dispatcher routes
        it through OpenAI (native web search isn't used).
        """
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == WEB_SEARCH_OPENAI_TYPE:
                anthropic_tools.append({
                    "name": PROXIED_WEB_SEARCH_NAME,
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

        return anthropic_tools or None
