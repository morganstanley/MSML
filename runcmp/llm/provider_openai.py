"""OpenAI Responses API provider for alpha-lab.

Implements the Provider protocol using the OpenAI SDK's Responses API
with ZDR-compatible settings (store=False, local history tracking).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from runcmp.llm.provider import Provider, Response, StreamEvent, ToolCall


# ---------------------------------------------------------------------------
# Model alias resolution
# ---------------------------------------------------------------------------
# The MS AI Gateway requires the full OpenAI model id (e.g. "gpt-5.6-sol").
# The documented friendly alias "gpt-5.6" is NOT resolved by the gateway
# (it returns 403), so we map known aliases to their concrete gateway id here.
# Any model not in this map passes through unchanged, so existing configs
# (gpt-5.2, gpt-5.4, gpt-5.5, and the concrete gpt-5.6-sol/terra/luna ids)
# are unaffected — the resolution is purely additive / backwards compatible.
OPENAI_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.6": "gpt-5.6-sol",   # alias -> frontier tier (per the model card)
}


def resolve_openai_model(model: str) -> str:
    """Map a friendly OpenAI model alias to its gateway model id.

    Identity for any model not in :data:`OPENAI_MODEL_ALIASES`.
    """
    return OPENAI_MODEL_ALIASES.get(model, model)



def _dump(obj: Any) -> dict[str, Any]:
    """The provider's response object verbatim; {} if it will not serialize."""
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
    except Exception:  # noqa: BLE001 — logging must never break the loop
        pass
    return {}


def _stop_reason(resp: Any) -> str:
    """Normalized end-of-generation reason for a Responses API result."""
    status = getattr(resp, "status", "") or ""
    details = getattr(resp, "incomplete_details", None)
    reason = getattr(details, "reason", "") if details is not None else ""
    return f"{status}:{reason}" if reason else status


class OpenAIProvider:
    """Provider backed by the OpenAI Responses API."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    @property
    def openai_client(self) -> OpenAI:
        """Expose the underlying OpenAI client (used for web search proxy)."""
        return self._client

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
        response_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a response via the OpenAI Responses API."""
        model = resolve_openai_model(model)
        # Recorded alongside the response so the call is fully reconstructable
        # from the log; the prompt and tools are already in the request event.
        request_params = {
            "model": model,
            "store": False,
            "truncation": "auto",
            "reasoning": {"effort": reasoning_effort},
            "include": ["reasoning.encrypted_content"],
        }
        create_kwargs: dict[str, Any] = dict(
            model=model,
            instructions=system,
            input=history,
            tools=tools,
            store=False,  # ZDR: don't store on server
            truncation="auto",
            stream=True,
            reasoning={"effort": reasoning_effort},
            include=["reasoning.encrypted_content"],
        )
        if response_schema is not None:
            text_format = {"type": "json_schema", "name": "critic_verdict",
                          "schema": response_schema, "strict": True}
            create_kwargs["text"] = {"format": text_format}
            request_params["text"] = {"format": text_format}
        stream = self._client.responses.create(**create_kwargs)

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
            # Best-effort cleanup — swallow any cleanup exception so it
            # doesn't mask the actual streaming error the caller cares about.
            try:
                stream.close()
            except Exception:
                pass

        if raw_response is None:
            # The stream ended without a response.completed event — a
            # truncated response, not an empty one. Returning silently here
            # made callers see "no response" with nothing to retry: observed
            # killing a post-hoc investigation ("API unavailable" after a
            # single silent non-answer) while the gateway was healthy.
            # Raising routes it through every caller's existing retry path.
            raise RuntimeError(
                "OpenAI response stream ended before response.completed"
            )

        # Parse the completed response into normalized form
        text_output = ""
        tool_calls: list[ToolCall] = []
        has_web_search = False
        raw_output_items: list[dict[str, Any]] = []

        for item in raw_response.output:
            # Serialize to dict for raw_output_items. Best-effort: if
            # serialization fails we omit this item from raw_output_items
            # but still let the item.type parsing below run, so a
            # serialization failure doesn't cost us the actual text /
            # tool_call data the caller needs.
            try:
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "to_dict"):
                    item_dict = item.to_dict()
                else:
                    item_dict = dict(item)
                raw_output_items.append(item_dict)
            except Exception:
                pass

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
                has_web_search = True

        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read_input_tokens = 0
        usage_raw = {}
        if hasattr(raw_response, "usage") and raw_response.usage:
            input_tokens = raw_response.usage.input_tokens
            output_tokens = raw_response.usage.output_tokens
            input_details = getattr(raw_response.usage, "input_tokens_details", None)
            if input_details is not None:
                cache_read_input_tokens = getattr(input_details, "cached_tokens", 0) or 0
            # Record the FULL usage object (all fields, incl. nested *_details)
            # so recorded telemetry is complete and identical across systems.
            usage_raw = (raw_response.usage.model_dump()
                         if hasattr(raw_response.usage, "model_dump")
                         else dict(raw_response.usage))
            # Reasoning tokens live under ``output_tokens_details.reasoning_tokens``
            # in OpenAI's Responses API (and grok's xAI mirror). When the
            # field is missing (older endpoints, non-reasoning models) we
            # leave the value at 0. ``output_tokens`` already includes the
            # reasoning tokens — they are surfaced here so the central
            # token-usage log can attribute cost separately.
            details = getattr(raw_response.usage, "output_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        response = Response(
            id=raw_response.id,
            text=text_output,
            tool_calls=tool_calls,
            has_web_search=has_web_search,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            usage_raw=usage_raw,
            raw_output_items=raw_output_items,
            # Why generation ended. The Responses API reports "incomplete"
            # with a reason rather than a stop reason on the message itself;
            # a response cut off at the token limit otherwise looks complete.
            stop_reason=_stop_reason(raw_response),
            raw_response=_dump(raw_response),
            request_params=request_params,
        )

        yield StreamEvent(type="done", response=response)

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
        """Simple non-streaming completion via Chat Completions API."""
        model = resolve_openai_model(model)
        chat_messages = [{"role": "system", "content": system}] + messages
        # Newer models (gpt-5-mini, gpt-5.4+) require max_completion_tokens
        # instead of max_tokens
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=chat_messages,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e):
                response = self._client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    max_tokens=max_tokens,
                )
            else:
                raise
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        """Build OpenAI Responses API user input items."""
        return [{"role": "user", "content": message}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build function_call_output items + optional image injection."""
        items: list[dict[str, Any]] = []

        for r in results:
            items.append({
                "type": "function_call_output",
                "call_id": r["call_id"],
                "output": r["output"],
            })

        if images:
            image_content: list[dict[str, Any]] = []
            for b64_data, media_type in images:
                image_content.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{b64_data}",
                })
            image_content.append({
                "type": "input_text",
                "text": "Here is the image you requested. Analyze it carefully.",
            })
            items.append({
                "role": "user",
                "content": image_content,
            })

        return items

    def append_response_to_history(
        self,
        history: list[dict[str, Any]],
        response: Response,
    ) -> None:
        """Append cleaned output items to the conversation history."""
        for item_dict in response.raw_output_items:
            cleaned = self._clean_output_item(item_dict)
            if cleaned:
                history.append(cleaned)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_output_item(item: dict[str, Any]) -> dict[str, Any] | None:
        """Clean an output item for use as input in next turn.

        Removes response-specific fields that aren't valid for input.
        """
        item_type = item.get("type", "")
        exclude_fields = {"status", "id"}

        if item_type == "message":
            return {
                "type": item_type,
                "role": item.get("role", "assistant"),
                "content": item.get("content", ""),
            }

        if item_type == "reasoning":
            return {k: v for k, v in item.items() if k not in exclude_fields}

        if item_type == "function_call":
            return {
                "type": item_type,
                "call_id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            }

        cleaned = {k: v for k, v in item.items() if k not in exclude_fields}
        return cleaned if cleaned else None
