"""xAI / grok provider — OpenAI-API-compatible upstream + OpenAI-backed web search proxy.

The xAI gateway path (``/xai/v1/chat/completions`` and ``/xai/v1/responses``) is
OpenAI-API-compatible for chat and the Responses API, so we reuse most of
``OpenAIProvider``. Several divergences require provider-level handling:

  1. **Web search.** xAI's Responses API rejects ``{"type": "web_search_preview"}``
     (422 "unknown variant"). Per system policy, web search always flows through
     OpenAI regardless of the configured provider — so we translate the built-in
     literal into an OpenAI-style function tool ``web_search`` and let the
     dispatcher's existing ``_proxy_web_search`` route the call through a real
     OpenAI client. The ``openai_client`` property exposes that proxy client to
     the agent loop (the agent reads ``getattr(provider, "openai_client", None)``).

  2. **Reasoning items / "compaction blob".** xAI's Responses API returns reasoning
     items whose ``encrypted_content`` is described in their error path as a
     "compaction blob" that must come back ``unmodified``. The existing
     ``_clean_output_item`` strips ``id`` and ``status`` from reasoning items
     before they go back as input on the next turn — that modification causes
     grok to reject every subsequent request with HTTP 400 ``"Could not decode the
     compaction blob. Ensure it is unmodified from the compact response."``. The
     fix: don't request ``reasoning.encrypted_content`` for grok (skip the
     ``include`` param), AND skip reasoning items entirely when building input
     for the next turn. Grok performs reasoning per-turn internally (visible in
     ``usage.completion_tokens_details.reasoning_tokens``); we lose the cross-turn
     reasoning continuity that OpenAI's encrypted-content mechanism provides,
     but the alternative is a hard 400 on every multi-turn call.

  3. **/models endpoint** returns 404 — irrelevant here but worth knowing for
     debugging.

Reasoning-effort tiers (grok-4.5): ``minimal`` / ``low`` / ``medium`` / ``high`` /
``xhigh``. OpenAI's ``none`` / ``max`` are rejected (HTTP 400), so
``resolve_grok_effort`` clamps them (``none``/``""`` → ``minimal``, ``max`` →
``xhigh``) before the request. (grok-4.3 lacked the ``xhigh`` tier.)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab.provider import Response, StreamEvent, ToolCall
from alpha_lab.provider_openai import OpenAIProvider


# grok-4.5 reasoning-effort tiers. grok accepts minimal|low|medium|high|xhigh and
# REJECTS OpenAI's "none"/"max" (HTTP 400), so callers' effort values (which use
# the OpenAI vocabulary, incl. none/max) are clamped into this set before the
# request. grok-4.5 adds the "xhigh" tier over grok-4.3's minimal..high.
GROK_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
_GROK_EFFORT_CLAMP: dict[str, str] = {
    "none": "minimal",   # grok has no "off" tier; use its lowest
    "": "minimal",
    "max": "xhigh",      # grok's highest tier
}


def resolve_grok_effort(effort: str) -> str:
    """Clamp an OpenAI-style effort value to a tier grok-4.5 accepts.

    Valid grok tiers pass through; ``none``/``""`` -> ``minimal``; ``max`` ->
    ``xhigh``; anything else unknown -> ``medium`` (a safe balanced default).
    """
    e = (effort or "").strip().lower()
    if e in GROK_EFFORTS:
        return e
    return _GROK_EFFORT_CLAMP.get(e, "medium")


_WEB_SEARCH_FN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web for current information. Returns search results as text. "
        "Routed through OpenAI's web_search_preview; available regardless of the "
        "configured main provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


class GrokProvider(OpenAIProvider):
    """Provider for xAI / grok models.

    Inherits the chat / Responses / history-helper machinery from
    ``OpenAIProvider`` (since xAI is OpenAI-API-compatible). Adds:

      * an OpenAI proxy client for ``web_search`` (exposed via ``openai_client``
        so the agent loop can pass it through to ``_proxy_web_search``).
      * a ``stream_response`` override that translates the built-in
        ``web_search_preview`` tool literal into a function tool the dispatcher
        can route to the proxy.
    """

    def __init__(self, grok_client: OpenAI, openai_client_for_proxy: OpenAI) -> None:
        super().__init__(grok_client)
        # The OpenAIProvider stored ``grok_client`` as ``self._client`` for
        # chat/responses calls. We keep a separate handle to a real OpenAI
        # client used solely for the web_search proxy.
        self._proxy = openai_client_for_proxy

    @property
    def openai_client(self) -> OpenAI:  # type: ignore[override]
        """Return the OpenAI client used for the web_search proxy — NOT the
        grok client.

        The agent loop reads this attribute to thread an OpenAI client into
        ``execute_tool``'s ``_proxy_web_search`` dispatch. Web search must go
        through OpenAI regardless of the main provider; returning the grok
        client here would defeat the proxy.
        """
        return self._proxy

    @staticmethod
    def _clean_output_item(item: dict[str, Any]) -> dict[str, Any] | None:  # type: ignore[override]
        """Drop reasoning items when building the next-turn input.

        Grok returns reasoning items in its response output, and the parent
        ``OpenAIProvider._clean_output_item`` would round-trip them back into
        the next request — at which point xAI rejects the cleaned blob with
        ``"Could not decode the compaction blob. Ensure it is unmodified..."``
        because we stripped ``id`` and ``status`` from it. Drop reasoning
        items entirely so they never go back. The agent's ApiResponseEvent
        log still captures them (they remain in ``raw_output_items``); only
        the next-turn input loses them.

        All non-reasoning item types delegate to the parent's logic.
        """
        if item.get("type") == "reasoning":
            return None
        return OpenAIProvider._clean_output_item(item)

    def _translate_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Swap any ``{"type": "web_search_preview"}`` entries for the function-tool
        equivalent the dispatcher routes to ``_proxy_web_search``. All other
        tool definitions pass through unchanged.
        """
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "web_search_preview":
                out.append(_WEB_SEARCH_FN_SCHEMA)
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
        """Stream via xAI's Responses API.

        Diverges from ``OpenAIProvider.stream_response`` in two ways: tool
        translation (built-in ``web_search_preview`` → function tool), and no
        ``include=["reasoning.encrypted_content"]`` (xAI's compaction blob
        format isn't compatible with our cleaner — see module docstring).

        Also defensively strips any prior-turn ``reasoning`` items out of the
        input on the way out, in case ``_clean_output_item`` ever leaks them
        through here (e.g. a history loaded from a mixed-provider run).
        """
        effort = resolve_grok_effort(reasoning_effort)
        translated = self._translate_tools(tools)
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
            # NOTE: deliberately no include=["reasoning.encrypted_content"]
            # — grok returns blobs that fail "compaction blob" decode on
            # subsequent turns once our cleaner touches them. See module
            # docstring.
        )

        full_text = ""
        raw_response = None
        try:
            for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta
                    yield StreamEvent(type="text_delta", delta=event.delta)
                elif event.type == "response.completed":
                    raw_response = event.response
        finally:
            try:
                stream.close()
            except Exception:
                pass

        if raw_response is None:
            return

        # Parse the completed response into normalized form. Mirrors the
        # logic in OpenAIProvider.stream_response — kept locally because we
        # need full control over which output items get carried into
        # raw_output_items (we drop reasoning items unconditionally here).
        text_output = ""
        tool_calls: list[ToolCall] = []
        has_web_search = False
        raw_output_items: list[dict[str, Any]] = []

        for item in raw_response.output:
            # Serialize defensively. Best-effort: a single failure must not
            # drop the rest of the response.
            try:
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "to_dict"):
                    item_dict = item.to_dict()
                else:
                    item_dict = dict(item)
            except Exception:
                item_dict = None

            # Keep ALL items (including reasoning) in raw_output_items so
            # the agent's ApiResponseEvent log captures them — without that,
            # we lose visibility into the model's reasoning trace (summary,
            # token counts). The drop-on-history-build happens later in
            # ``_clean_output_item`` below, which returns None for reasoning
            # items so they never go back to grok as input.
            if item_dict is not None:
                raw_output_items.append(item_dict)

            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text_output += content.text
            elif item.type == "function_call":
                tool_calls.append(ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                ))
            elif item.type == "web_search_call":
                # Defensive: shouldn't appear for grok (we translate web search
                # to a function tool that routes through OpenAI). Treat the
                # same way OpenAIProvider does, just in case.
                has_web_search = True

        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        if hasattr(raw_response, "usage") and raw_response.usage:
            input_tokens = raw_response.usage.input_tokens
            output_tokens = raw_response.usage.output_tokens
            # xAI / grok exposes reasoning under output_tokens_details.reasoning_tokens
            # in /responses, matching OpenAI's shape. Always-on for Grok-4.3
            # regardless of reasoning_effort tier.
            details = getattr(raw_response.usage, "output_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        yield StreamEvent(
            type="done",
            response=Response(
                id=raw_response.id,
                text=text_output or full_text,
                tool_calls=tool_calls,
                has_web_search=has_web_search,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                raw_output_items=raw_output_items,
            ),
        )
