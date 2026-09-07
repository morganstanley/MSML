"""xAI / grok provider — OpenAI-API-compatible upstream + OpenAI-backed web search.

The xAI gateway path (``/xai/v1/responses``) is OpenAI-API-compatible, so we
subclass :class:`~alpha_lab.providers.openai.OpenAIProvider` and reuse
its chat / Responses / history-helper machinery. Three divergences require
grok-specific handling (all verified live against ``grok-4.5``):

1. **Reasoning "compaction blob".** grok's Responses API returns ``reasoning``
   items whose ``encrypted_content`` must come back *unmodified* or grok rejects
   the next turn with HTTP 400 ``"Could not decode the compaction blob"``. The
   parent's ``_clean_output_item`` strips ``id``/``status`` from reasoning items,
   which corrupts that blob. Fix: never request
   ``include=["reasoning.encrypted_content"]`` and **drop reasoning items** from
   the next-turn input entirely. grok still reasons per-turn internally
   (``usage.output_tokens_details.reasoning_tokens``); we only lose cross-turn
   reasoning continuity, which is the price of a working multi-turn loop.

2. **Reasoning-effort tiers.** grok-4.5 accepts ``minimal | low | medium | high |
   xhigh`` and REJECTS ``none`` and ``max`` (HTTP 400). Configs written for the
   OpenAI tiers (which include ``none``/``max``) would 400, so we clamp
   unsupported values to the nearest supported grok tier.

3. **Web search.** grok's Responses API rejects the ``web_search`` tool
   literal; per policy web search always flows through OpenAI, so we translate
   that literal into a ``web_search`` function tool and expose an OpenAI proxy
   client (``openai_client``) for the dispatcher to route through.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import MappingProxyType
from typing import Any

from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse

from alpha_lab import token_metrics
from alpha_lab.providers.openai import OpenAIProvider
from alpha_lab.providers.types import Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.responses import (
    parse_responses_usage,
    raise_for_stream_failure,
    raise_stream_ended_early,
)
from alpha_lab.providers.utils.clients import get_grok_client, get_openai_client

# grok-4.5 reasoning-effort tiers (verified live). OpenAI's "none"/"max" are
# rejected, so callers' effort values are clamped into this set.
GROK_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
_GROK_EFFORT_CLAMP = MappingProxyType({
    "none": "minimal",   # grok has no "off" tier; use its lowest
    "": "minimal",
    "max": "xhigh",      # grok's highest tier
})


def resolve_grok_effort(effort: str | None) -> str:
    """Clamp an OpenAI-style effort value to a tier grok-4.5 accepts.

    Valid grok tiers pass through; ``none``/``""`` -> ``minimal``; ``max`` ->
    ``xhigh``; anything else unknown -> ``medium`` (a safe balanced default).
    """
    e = (effort or "").strip().lower()
    if e in GROK_EFFORTS:
        return e
    return _GROK_EFFORT_CLAMP.get(e, "medium")


def _build_web_search_fn_schema() -> dict[str, Any]:
    """Return a fresh ``web_search`` function-tool schema.

    A factory (not a shared module-level dict) so every translated tool is an
    independent object — there's no shared mutable state to accidentally
    corrupt across calls.
    """
    return {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web for current information. Returns search results as text. "
            "Routed through OpenAI's web_search; available regardless of the "
            "configured main provider."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


class GrokProvider(OpenAIProvider):
    """Provider for xAI / grok models (currently grok-4.5)."""

    @classmethod
    def from_config(
        cls, api_key: str | None = None, model: str = ""
    ) -> GrokProvider:
        """Build a GrokProvider from run config.

        ``api_key`` is the xAI key; the web_search proxy always uses OpenAI
        credentials, so its client is built from OpenAI env/token.
        """
        return cls(
            grok_client=get_grok_client(api_key),
            openai_client_for_proxy=get_openai_client(),
        )

    def __init__(self, grok_client: OpenAI, openai_client_for_proxy: OpenAI) -> None:
        super().__init__(grok_client)
        # The grok client stays as self._client for chat; override _openai_client
        # (read by the inherited openai_client property) so web_search routes
        # through a real OpenAI client — grok can't do web search itself.
        self._openai_client = openai_client_for_proxy

    @staticmethod
    def _clean_output_item(item: dict[str, Any]) -> dict[str, Any] | None:  # type: ignore[override]
        """Drop reasoning items so they never round-trip back to grok.

        Sending a cleaned (id/status-stripped) reasoning item back triggers
        grok's "Could not decode the compaction blob" 400. All other item
        types delegate to the parent.
        """
        if item.get("type") == "reasoning":
            return None
        return OpenAIProvider._clean_output_item(item)

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Swap built-in ``web_search`` for the proxy function tool."""
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "web_search":
                # fresh schema per call — no shared mutable state to corrupt
                out.append(_build_web_search_fn_schema())
            else:
                out.append(t)
        return out

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str,
    ) -> Iterator[StreamEvent]:
        """Stream via xAI's Responses API (grok compaction-blob / effort / tools handled)."""
        effort = resolve_grok_effort(reasoning_effort)
        translated = self._translate_tools(tools)
        # Defensively strip any reasoning items from input (compaction-blob quirk).
        cleaned_input = [
            item for item in history
            if not (isinstance(item, dict) and item.get("type") == "reasoning")
        ]

        stream = self._client.responses.create(
            model=model,
            instructions=system,
            input=cleaned_input,
            tools=translated,
            store=False,  # ZDR
            truncation="auto",
            stream=True,
            reasoning={"effort": effort},
            # NB: deliberately NO include=["reasoning.encrypted_content"] for grok.
        )

        full_text = ""
        raw_response: OpenAIResponse | None = None
        try:
            for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta
                    yield StreamEvent(type="text_delta", delta=event.delta)
                elif event.type == "response.completed":
                    raw_response = event.response
                else:
                    raise_for_stream_failure(event, "Grok")
        finally:
            try:
                stream.close()
            except Exception:
                pass

        if raw_response is None:
            raise_stream_ended_early("Grok")

        text_output = ""
        tool_calls: list[ToolCall] = []
        has_web_search = False
        raw_output_items: list[dict[str, Any]] = []

        for item in getattr(raw_response, "output", ()) or ():
            # Keep ALL items (incl. reasoning) in raw_output_items for logging;
            # the drop-from-history happens in _clean_output_item.
            try:
                if hasattr(item, "model_dump"):
                    raw_output_items.append(item.model_dump())
                elif hasattr(item, "to_dict"):
                    raw_output_items.append(item.to_dict())
                else:
                    raw_output_items.append(dict(item))
            except Exception:
                pass

            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text_output += content.text
            elif item.type == "function_call":
                tool_calls.append(ToolCall(
                    call_id=item.call_id, name=item.name, arguments=item.arguments,
                ))
            elif item.type == "web_search_call":
                has_web_search = True

        usage = parse_responses_usage(raw_response.usage)

        response = Response(
            id=raw_response.id,
            text=text_output or full_text,
            tool_calls=tool_calls,
            has_web_search=has_web_search,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_write_input_tokens=0,
            reasoning_tokens=usage.reasoning_tokens,
            usage_raw=usage.usage_raw,
            raw_output_items=raw_output_items,
        )
        token_metrics.record(
            usage.input_tokens, usage.output_tokens,
            cache_read=usage.cache_read_input_tokens, model=model, system="xai",
        )
        yield StreamEvent(type="done", response=response)

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> str:
        """Non-streaming completion for grok via the Responses API.

        The parent :meth:`OpenAIProvider.complete` uses Chat Completions
        (``client.chat.completions.create``). The only xAI gateway route we
        verified is ``/xai/v1/responses``, so bounded utility calls
        (summarization, etc.) go through that same known-good endpoint here
        rather than assuming the gateway proxies Chat Completions for grok.
        ``minimal`` effort keeps the call cheap; reasoning items are dropped
        from the input (compaction-blob quirk).
        """
        input_items = [
            m for m in messages
            if not (isinstance(m, dict) and m.get("type") == "reasoning")
        ]
        resp = self._client.responses.create(
            model=model,
            instructions=system,
            input=input_items,
            store=False,  # ZDR
            truncation="auto",
            reasoning={"effort": "minimal"},
            max_output_tokens=max_tokens,
        )
        if getattr(resp, "usage", None):
            token_metrics.record(
                resp.usage.input_tokens,
                resp.usage.output_tokens,
                model=model,
                system="xai",
            )
        text = getattr(resp, "output_text", "") or ""
        if text:
            return text
        # Fallback: aggregate output_text pieces from message items.
        parts: list[str] = []
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", None) or []:
                    if getattr(content, "type", None) == "output_text":
                        parts.append(content.text)
        return "".join(parts)


__all__ = ["GrokProvider", "resolve_grok_effort", "GROK_EFFORTS"]
