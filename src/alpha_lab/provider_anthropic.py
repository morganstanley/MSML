"""Provider backed by the native Anthropic Messages API.

Claude reached through ``/anthropic/v1/messages`` on the gateway rather than
through the Bedrock Converse route. Two practical advantages over the Bedrock
path:

* **No schema translation.** Tool definitions, tool results, thinking blocks
  and usage all stay in Anthropic's own format, so nothing is reshaped on the
  way in or out. The Bedrock path needed per-model thinking routing (Opus 4.8
  rejects the legacy ``thinking.type=enabled`` form, Opus 5 only accepts
  ``adaptive`` + ``output_config.effort``); natively, every Claude uses
  adaptive thinking and the effort string passes straight through.
* **Complete usage.** The response's usage object is recorded verbatim in
  ``Response.usage_raw`` — including ``cache_read_input_tokens`` and
  ``cache_creation_input_tokens``. The Bedrock provider dropped those fields
  for months, which made cache behaviour unknowable from run logs.

Model ids are plain (``claude-opus-4-8``, ``claude-opus-5``): no version
parsing, so a new model needs no code change.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab.provider import Response, StreamEvent, ToolCall

logger = logging.getLogger("alpha_lab.provider_anthropic")

# Streaming output ceiling. Native Claude supports more, but 64k leaves room
# for adaptive thinking plus the answer; streaming avoids HTTP-timeout risk.
DEFAULT_MAX_TOKENS = 64_000

# Effort tiers accepted by output_config.effort under adaptive thinking.
ADAPTIVE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

WEB_SEARCH_OPENAI_TYPE = "web_search"

# Name the proxied search tool travels under on the wire. The gateway rejects
# a Messages-API tool called "web_search" outright -- that name is reserved for
# Anthropic's own hosted tool, and only caller-defined tools are permitted:
#   "Server/hosted Anthropic tool name 'web_search' is not currently allowed
#    for messages API calls."
# It is sent under this name and mapped back before the agent sees it, so tool
# dispatch is unchanged.
PROXIED_WEB_SEARCH_NAME = "search_web"

# Prefix caching. The Bedrock path marks the static prefix -- system prompt
# and tool list -- with a 1-hour cache point; without the equivalent marker
# here, every turn would re-ingest that prefix at full price. Placed on the
# system block and on the LAST tool so the cached prefix spans both.
CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}



_CACHEABLE_BLOCK_TYPES = ("text", "image", "tool_use", "tool_result", "document")


def _mark_history_cache_point(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Moving cache breakpoint on the last history block.

    Without this, only the system prompt and tool list are cached and the
    ENTIRE conversation history re-bills as fresh input on every turn —
    measured on d2_o5_cond (2026-07-29): a 70-turn worker session read the
    flat 16,761-token system+tools prefix from cache each call while fresh
    input climbed 2.6K -> 94K per call, 3.26M fresh tokens total for ~95K of
    unique content. Session cost grows quadratically with depth.

    Marking the last content block of the last message makes each call write
    only its new suffix to the 1h cache (2x write rate on the delta) and read
    the whole prior history at the 0.1x cache-read rate — the next call's
    prefix extends this one, so the breakpoint always hits. Breakpoints used:
    system (1) + tools (1) + history (1) = 3 of the 4 the API allows.

    Never mutates the caller's history: the agent reuses its local history
    list across turns, and stale markers accumulating there would exceed the
    4-breakpoint limit. The last message and its content list are copied.
    """
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        new_content: Any = [{"type": "text", "text": content,
                             "cache_control": CACHE_CONTROL}]
    elif isinstance(content, list) and content:
        idx = None
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if (isinstance(block, dict)
                    and block.get("type") in _CACHEABLE_BLOCK_TYPES):
                idx = i
                break
        if idx is None:
            return messages
        new_content = list(content)
        new_content[idx] = {**new_content[idx], "cache_control": CACHE_CONTROL}
    else:
        return messages
    return messages[:-1] + [{**last, "content": new_content}]


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
            allowed = {b.get("id") for b in prev.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "tool_use"}
        kept = [b for b in content
                if not (isinstance(b, dict) and b.get("type") == "tool_result"
                        and b.get("tool_use_id") not in allowed)]
        if len(kept) != len(content):
            dropped = len(content) - len(kept)
            logger.warning(
                "Dropped %d orphaned tool_result block(s) at message %d "
                "(their tool_use turn is no longer in history)", dropped, i,
            )
        if kept:
            cleaned.append({**msg, "content": kept})
    return cleaned


class AnthropicProvider:
    """Provider speaking the native Anthropic Messages API."""

    @property
    def openai_client(self) -> OpenAI:
        """OpenAI client used for the web_search proxy and summarization."""
        return self._openai_client

    def __init__(self, client: Any, openai_client: OpenAI) -> None:
        # ``client`` is an ``anthropic.Anthropic``; typed Any so this module
        # imports even where the SDK is absent (it is built lazily).
        self._client = client
        self._openai_client = openai_client

    # -- streaming -----------------------------------------------------

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str,
    ) -> Iterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": model,
            "system": [{"type": "text", "text": system,
                        "cache_control": CACHE_CONTROL}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": _mark_history_cache_point(
                _strip_orphan_tool_results(history)),
        }
        translated = self._translate_tools(tools)
        if translated:
            translated[-1] = {**translated[-1], "cache_control": CACHE_CONTROL}
            kwargs["tools"] = translated

        effort = reasoning_effort.strip().lower() if reasoning_effort else ""
        if effort and effort != "none":
            if effort in ADAPTIVE_EFFORTS:
                # ``output_config`` is newer than the installed SDK's typed
                # signature, which rejects it as an unexpected keyword; send
                # it through the SDK's passthrough so the wire request is
                # correct regardless of SDK version.
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["extra_body"] = {"output_config": {"effort": effort}}
            else:
                logger.warning(
                    "Unknown reasoning_effort %r; continuing without thinking "
                    "(expected one of %s)", reasoning_effort,
                    sorted(ADAPTIVE_EFFORTS),
                )

        request_params = {k: v for k, v in kwargs.items()
                          if k not in ("messages", "system", "tools")}
        request_params["system_cache_control"] = CACHE_CONTROL
        request_params["tools_cached"] = bool(translated)

        text_chunks: list[str] = []
        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if (event.type == "content_block_delta"
                        and getattr(event.delta, "type", None) == "text_delta"):
                    text_chunks.append(event.delta.text)
                    yield StreamEvent(type="text_delta", delta=event.delta.text)
            final = stream.get_final_message()

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

        usage = getattr(final, "usage", None)
        usage_raw = self._usage_to_dict(usage)

        # The whole response object, verbatim. Normalized fields below are
        # derived from it; anything this dataclass has no name for (server
        # tool use, container info, future fields) survives here.
        try:
            raw_response = final.model_dump(mode="json")
        except Exception:  # noqa: BLE001 — logging must never break the loop
            raw_response = {}
        stop_reason = getattr(final, "stop_reason", "") or ""
        if stop_reason == "max_tokens":
            logger.warning(
                "Response hit the %d-token output limit and was cut off "
                "(model=%s); it is truncated, not complete.",
                DEFAULT_MAX_TOKENS, model,
            )

        yield StreamEvent(
            type="done",
            response=Response(
                id=getattr(final, "id", "") or "",
                text="".join(text_chunks),
                tool_calls=tool_calls,
                has_web_search=False,      # web search proxies through OpenAI
                input_tokens=int(usage_raw.get("input_tokens") or 0),
                output_tokens=int(usage_raw.get("output_tokens") or 0),
                cache_read_input_tokens=int(
                    usage_raw.get("cache_read_input_tokens") or 0),
                cache_write_input_tokens=int(
                    usage_raw.get("cache_creation_input_tokens") or 0),
                usage_raw=usage_raw,
                raw_output_items=raw_output_items,
                stop_reason=stop_reason,
                raw_response=raw_response,
                request_params=request_params,
            ),
        )

    # -- non-streaming -------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str:
        """Non-streaming completion; non-Claude ids fall back to OpenAI."""
        if not model.startswith(("claude-", "anthropic.", "us.anthropic.")):
            chat = [{"role": "system", "content": system}] + messages
            resp = self._openai_client.chat.completions.create(
                model=model, messages=chat, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        resp = self._client.messages.create(
            model=model,
            system=[{"type": "text", "text": system,
                     "cache_control": CACHE_CONTROL}],
            max_tokens=max_tokens, messages=messages)
        return "".join(b.text for b in resp.content if b.type == "text")

    # -- history helpers (native Anthropic shapes) ---------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": [{"type": "text", "text": message}]}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """All results go in one user message, as Anthropic pairs tool_result
        blocks with the preceding assistant tool_use turn."""
        content: list[dict[str, Any]] = [
            {"type": "tool_result", "tool_use_id": r["call_id"],
             "content": r["output"]}
            for r in results
        ]
        if images:
            for b64_data, media_type in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type,
                               "data": b64_data},
                })
            content.append({"type": "text",
                            "text": "Here is the image you requested. "
                                    "Analyze it carefully."})
        return [{"role": "user", "content": content}]

    def append_response_to_history(
        self, history: list[dict[str, Any]], response: Response,
    ) -> None:
        if response.raw_output_items:
            history.append({"role": "assistant",
                            "content": response.raw_output_items})

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _usage_to_dict(usage: Any) -> dict[str, Any]:
        """Every usage field, verbatim — nothing dropped.

        Recorded whole so cache behaviour stays visible in run logs; the
        Bedrock provider kept only two fields and made it unknowable.
        """
        if usage is None:
            return {}
        for attr in ("model_dump", "dict", "to_dict"):
            fn = getattr(usage, attr, None)
            if callable(fn):
                try:
                    d = fn()
                    if isinstance(d, dict):
                        return {k: v for k, v in d.items() if v is not None}
                except Exception:  # noqa: BLE001 — fall through to manual read
                    pass
        return {k: getattr(usage, k) for k in dir(usage)
                if not k.startswith("_")
                and isinstance(getattr(usage, k, None), (int, float, str))}

    @staticmethod
    def _block_to_param(block: Any) -> dict[str, Any]:
        """Response block -> input param, so replay is faithful (thinking
        blocks keep their signature)."""
        btype = block.type
        if btype == "text":
            return {"type": "text", "text": block.text}
        if btype == "tool_use":
            return {"type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input}
        if btype == "thinking":
            return {"type": "thinking", "thinking": block.thinking,
                    "signature": block.signature}
        if btype == "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.data}
        return block.model_dump()

    def _translate_tools(
        self, tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """OpenAI Responses tool schemas -> Anthropic ``tools``.

        The built-in web_search type becomes a function tool so the dispatcher
        routes it through OpenAI, matching every other provider here.
        """
        out: list[dict[str, Any]] = []
        for tool in tools:
            ttype = tool.get("type", "")
            if ttype == WEB_SEARCH_OPENAI_TYPE:
                out.append({
                    "name": PROXIED_WEB_SEARCH_NAME,
                    "description": ("Search the web for current information. "
                                    "Returns search results as text."),
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {
                            "type": "string",
                            "description": "The search query."}},
                        "required": ["query"],
                    },
                })
            elif ttype == "function":
                out.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                })
        return out or None
