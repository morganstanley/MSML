"""OpenAI Responses API provider for alpha-lab.

Implements the Provider protocol using the OpenAI SDK's Responses API
with ZDR-compatible settings (store=False, local history tracking).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse

from alpha_lab import token_metrics
from alpha_lab.providers.types import Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.clients import get_openai_client
from alpha_lab.providers.utils.responses import (
    parse_responses_usage,
    raise_for_stream_failure,
    raise_stream_ended_early,
)

# The MS AI Gateway requires the full OpenAI model id (e.g. "gpt-5.6-sol"); the
# friendly alias "gpt-5.6" is NOT resolved by the gateway (403), so map known
# aliases to their concrete gateway id. Any model not in the map passes through
# unchanged. Immutable (read-only) so it can't be mutated at runtime.
OPENAI_MODEL_ALIASES: Mapping[str, str] = MappingProxyType({
    "gpt-5.6": "gpt-5.6-sol",   # alias -> frontier tier (per the model card)
})


def resolve_openai_model(model: str) -> str:
    """Map a friendly OpenAI model alias to its gateway model id.

    Identity for any model not in :data:`OPENAI_MODEL_ALIASES`.
    """
    return OPENAI_MODEL_ALIASES.get(model, model)


class OpenAIProvider:
    """Provider backed by the OpenAI Responses API."""

    @property
    def openai_client(self) -> OpenAI:
        """The OpenAI client used for the web_search proxy / summarization."""
        return self._openai_client

    def __init__(self, client: OpenAI) -> None:
        # For OpenAI the one client serves both the main API and the web_search
        # proxy, so _openai_client (read by the openai_client property) aliases _client.
        self._client = client
        self._openai_client = client

    @classmethod
    def from_config(
        cls, api_key: str | None = None, model: str = ""
    ) -> OpenAIProvider:
        """Build an OpenAIProvider from run config."""
        return cls(get_openai_client(api_key))

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
        """Stream a response via the OpenAI Responses API."""
        model = resolve_openai_model(model)
        stream = self._client.responses.create(
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
                    raise_for_stream_failure(event, "OpenAI")
        finally:
            # Best-effort cleanup — swallow any cleanup exception so it
            # doesn't mask the actual streaming error the caller cares about.
            try:
                stream.close()
            except Exception:
                pass

        if raw_response is None:
            raise_stream_ended_early("OpenAI")

        # Parse the completed response into normalized form
        text_output = ""
        tool_calls: list[ToolCall] = []
        has_web_search = False
        raw_output_items: list[dict[str, Any]] = []

        for item in getattr(raw_response, "output", ()) or ():
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

        # Token-usage metric
        token_metrics.record(
            usage.input_tokens,
            usage.output_tokens,
            cache_read=usage.cache_read_input_tokens,
            model=model,
            system="openai",
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
        token_metrics.record_chat_usage(response, model=model, system="openai")
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
