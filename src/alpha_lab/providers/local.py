"""Chat Completions provider — any OpenAI-compatible lab / local endpoint.

Supports lab-hosted models (gpu-host-1, gpu-host-2, Kimi, GLM, etc.) and any other
endpoint that speaks ``POST /v1/chat/completions`` but not the Responses API.

Key differences from ``OpenAIProvider`` (Responses API):

  1. **API**: ``chat.completions.create()`` with streaming, not ``responses.create()``.
  2. **History format**: standard OpenAI chat messages (role/content/tool_calls).
  3. **Tool schema**: Chat API ``{"type":"function","function":{...}}`` nesting,
     translated from the Responses-API flat format the system uses internally.
  4. **Web search**: proxied through an OpenAI client (lab models lack it natively).
  5. **Reasoning effort**: not forwarded — lab servers don't support this parameter.
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab.providers.bedrock import BedrockProvider
from alpha_lab.providers.litellm_proxy import models_by_tags
from alpha_lab.providers.local_models import (
    LocalModelBehavior,
    behavior_for,
    load_behaviors,
)
from alpha_lab.providers.types import Response, StreamEvent, ToolCall
from alpha_lab.providers.utils.clients import (
    get_bedrock_client,
    get_local_client,
    get_openai_client,
)

logger = logging.getLogger("alpha_lab.provider_local")


# Map reasoning_effort → Kimi thinking_budget (output tokens for internal CoT).
# "none" disables thinking entirely. Thinking tokens are output-only and do NOT
# consume input context, so they don't contribute to the 262k context overflow.
_THINKING_BUDGET: dict[str, int] = {
    "none": 0,
    "low": 2000,
    "medium": 8000,
    "high": 32000,
    "max": 32000,
}

# Hard cap on GLM streaming output tokens. GLM (vLLM) has no natural stop on a
# degraded/flapping server — it can stream non-terminating garbage, and with no
# max_tokens the request runs to max_model_len (~100k), hanging the dispatcher
# for many minutes (the per-chunk read timeout never fires because garbage
# keeps dribbling). A generous cap forces finish_reason="length" on runaway
# streams while leaving ample room for legitimate thinking + content (observed
# legit responses are well under this). GLM-scoped, so Kimi and the gpt-4o
# vision-proxy path are unaffected.
_GLM_MAX_OUTPUT_TOKENS = 32000

# GLM-5.1 recipe-recommended sampling (https://recipes.vllm.ai/zai-org/GLM-5.1):
# temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0 (these are also the vLLM
# defaults). We set temperature explicitly to match the recipe's examples; the
# rest are already the server defaults. GLM-scoped.
_GLM_TEMPERATURE = 1.0

# Text-based tool calling for GLM. Passing the native ``tools`` array to GLM-5.1
# on this vLLM deployment makes it degenerate into single-token repetition
# ("!!!!", no content, no tool call) once ~5+ real tool schemas are present —
# verified reproducibly, and immune to thinking on/off and sampling penalties.
# Plain-text prompts (even very large) never degenerate, and GLM emits valid
# tool-call JSON when asked. So for GLM we DON'T send ``tools``; we describe the
# tools in the system prompt and parse the tool call back out of the model's
# text. GLM-scoped — Kimi and the gpt-4o vision proxy keep the native tools API.
_GLM_TEXT_TOOL_INSTRUCTION = (
    "\n\n# Calling tools\n"
    "You have the tools listed below. To call one, emit a JSON object EXACTLY in "
    "this form (and nothing else after it). You may emit several objects, one per "
    "line, to call multiple tools in one turn:\n"
    '{"name": "<tool_name>", "arguments": {<arguments matching that tool\'s schema>}}\n'
    "Emit the tool-call JSON as your normal reply text (not in a thinking block). "
    "When you are done and want to hand back to the user, call the report_to_user "
    "tool the same way.\n\n"
    "## Available tools\n"
)


# Proxy function schema for web_search — same as GrokProvider's.
_WEB_SEARCH_FN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Returns search results as text. "
            "Routed through OpenAI's web_search; available regardless of "
            "the configured main provider."
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
    },
}


def _proxy_root(base_url: str) -> str:
    """Root URL for the LiteLLM admin API (``/v1/model/info``).

    ``models_by_tags`` appends ``/v1/model/info`` itself, and ``get_local_client``
    conversely appends ``/v1`` — so strip a trailing ``/v1`` here to avoid a
    doubled ``/v1/v1`` path.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


class LocalProvider:
    """Provider backed by the OpenAI Chat Completions API.

    Works with any OpenAI-compatible endpoint including lab-hosted models
    (gpu-host-1 / GLM-5.1, Kimi, etc.) that only expose ``/v1/chat/completions``.

    History is maintained in the standard Chat API messages format.
    Web search is proxied through ``openai_client_for_proxy`` (a real OpenAI
    client) using the same pattern as ``GrokProvider``.
    """

    @property
    def openai_client(self) -> OpenAI:
        """The OpenAI client used for the web_search proxy / summarization."""
        return self._openai_client

    @classmethod
    def from_config(
        cls,
        api_key: str | None = None,
        model: str = "",
        model_tags: list[str | list[str]] | None = None,
    ) -> LocalProvider:
        """Build a LocalProvider for a local vLLM / LiteLLM-proxy endpoint.

        The concrete model and its request-shaping behavior are resolved *per
        request* (not here): ``model_tags`` narrows the proxy's tagged pool to one
        model at call time (see :meth:`_select_model`), and that model is looked up
        in the ``LOCAL_MODEL_CONFIG`` behavior table (see
        ``providers.local_models``). Without tags, the plain ``model`` (or
        ``"auto"``) is used. This build step only wires the clients and config.
        """
        base_url = os.environ.get("LOCAL_BASE_URL", "")
        if not base_url:
            raise ValueError(
                "local provider requires LOCAL_BASE_URL to be set"
            )

        return cls(
            client=get_local_client(base_url),
            openai_client_for_proxy=get_openai_client(api_key),
            base_url=base_url,
            model=model,
            model_tags=list(model_tags) if model_tags else [],
            behaviors=load_behaviors(os.environ.get("LOCAL_MODEL_CONFIG")),
        )

    def __init__(
        self,
        client: OpenAI,
        openai_client_for_proxy: OpenAI,
        base_url: str = "",
        model: str = "",
        model_tags: list[str | list[str]] | None = None,
        behaviors: dict[str, LocalModelBehavior] | None = None,
    ) -> None:
        self._client = client
        self._openai_client = openai_client_for_proxy
        self._base_url = base_url
        # What to select from, per request: a tagged proxy pool, else a plain
        # model name (or "auto"). Behavior for the selected model comes from the
        # LOCAL_MODEL_CONFIG table (sniff-fallback when absent).
        self._model = model
        self._model_tags = model_tags or []
        self._behaviors = behaviors or {}
        self._pool: list[str] | None = None  # cached tag pool (queried once)
        self._resolved_model: str | None = None  # "auto" discovery cache
        # Vision fallback for text-only GLM image turns: a lazily-built, shared
        # BedrockProvider (opus-4.7). _vision_built distinguishes "not tried" from
        # "tried and unavailable" (cached None). None -> the gpt-4o proxy path.
        self._vision_provider: Any | None = None
        self._vision_built = False
        self._described_img_sigs: set[str] = set()
        # call_id -> tool name, for GLM text-tool mode: lets build_tool_result_items
        # label each result and detect text-mode calls regardless of which model
        # handles the next turn.
        self._textmode_call_names: dict[str, str] = {}

    def _select_model(self) -> str:
        """Pick the model for this request: random from the tagged pool, else the
        plain model (``"auto"`` auto-discovered). The tagged pool is queried once
        and cached; selection is per call so a tag load-balances across the pool.
        """
        if self._model_tags:
            if self._pool is None:
                pool = models_by_tags(_proxy_root(self._base_url), self._model_tags)
                if not pool:
                    raise ValueError(
                        f"no LiteLLM-proxy model matches model_tags {self._model_tags!r}"
                    )
                self._pool = pool
            model = random.choice(self._pool)
        else:
            model = self._resolve_model(self._model)
        return model

    def _behavior_for(self, model: str) -> LocalModelBehavior:
        """Request-shaping behavior for ``model`` (table entry, else sniff default)."""
        return behavior_for(model, self._behaviors)

    def _bedrock_vision(self) -> Any | None:
        """Lazily build (once) and return the shared opus-via-Bedrock vision
        provider, or ``None`` if unavailable (then the gpt-4o proxy is used)."""
        if not self._vision_built:
            self._vision_built = True
            try:
                self._vision_provider = BedrockProvider(
                    bedrock_client=get_bedrock_client(),
                    openai_client=self._openai_client,
                )
            except Exception as exc:
                logger.warning(
                    "opus vision fallback unavailable (%s); image turns will use "
                    "the gpt-4o proxy instead", exc,
                )
                self._vision_provider = None
        return self._vision_provider

    def _thinking_extra_body(self, reasoning_effort: str, thinking_style: str) -> dict[str, Any]:
        """Build the endpoint-specific ``extra_body`` for thinking control.

        Returns ``{}`` (no extra_body) or ``{"extra_body": {...}}`` ready to
        splat into ``chat.completions.create``.

        - ``glm``: thinking is ON BY DEFAULT (vLLM recipe
          https://recipes.vllm.ai/zai-org/GLM-5.1). The recipe's "thinking on"
          path sends NO thinking param at all, so for any non-"none" effort we
          send nothing and let GLM reason by default (reasoning is returned in
          ``message.reasoning``). Only ``reasoning_effort == "none"`` disables
          it, via ``chat_template_kwargs.enable_thinking=False`` (the recipe's
          documented "thinking off" form). GLM has no graded budget.
        - ``kimi`` (default): graded ``thinking_budget`` mapped from
          ``reasoning_effort``; ``budget == 0`` ("none") omits extra_body,
          which disables thinking.
        """
        if thinking_style == "glm":
            if (reasoning_effort or "none") == "none":
                return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            return {}
        budget = _THINKING_BUDGET.get(reasoning_effort or "none", 0)
        if budget > 0:
            return {"extra_body": {"enable_thinking": True, "thinking_budget": budget}}
        return {}

    def _resolve_model(self, model: str) -> str:
        """Return the actual model name, auto-discovering if model is 'auto'.

        Discovery queries ``GET /v1/models`` and takes the first id returned.
        The result is cached so the endpoint is hit at most once per provider
        instance. Falls back to ``zai-org/GLM-5.1`` (gpu-host-1 default) if
        discovery fails.

        A model pinned at build time (resolved from ``model_tags``, or a cached
        ``auto`` discovery) is authoritative and returned regardless of ``model``.
        """
        if self._resolved_model is not None:
            return self._resolved_model
        if model != "auto":
            return model
        try:
            ids = [m.id for m in self._client.models.list().data]
            if ids:
                self._resolved_model = ids[0]
                logger.info("LocalProvider: auto-discovered model %r", self._resolved_model)
                return self._resolved_model
            logger.warning("LocalProvider: /v1/models returned empty list; falling back to default")
        except Exception as e:
            logger.warning("LocalProvider: model auto-discovery failed: %s; falling back to default", e)
        self._resolved_model = "zai-org/GLM-5.1"
        return self._resolved_model

    def _stream_collect(self, client: OpenAI, kwargs: dict[str, Any]) -> tuple[str, str, bool]:
        """Stream a chat completion, aborting on a degenerate single-token run.

        Returns ``(content, reasoning, degenerated)``. Used for GLM text-tool
        retries so a degenerate retry stops in ~ms instead of running to the
        token cap. Honors the caller's kwargs (model, messages, extra_body,
        max_tokens), forcing streaming on.
        """
        text = ""
        reasoning = ""
        degenerated = False
        stream = client.chat.completions.create(
            **{**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        )
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                if getattr(d, "reasoning_content", None):
                    reasoning += d.reasoning_content
                if d.content:
                    text += d.content
                if _is_degenerate_tail(text) or _is_degenerate_tail(reasoning):
                    degenerated = True
                    break
        finally:
            try:
                stream.close()
            except Exception:
                pass
        return text, reasoning, degenerated

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
        """Stream via the Chat Completions API.

        Translates the Responses-API tool schemas to Chat API format, prepends
        the system message, and accumulates streaming tool-call deltas.
        ``reasoning_effort`` is not forwarded (lab endpoints don't support it).
        """
        # Route to GPT only when the CURRENT turn just injected fresh images (last ~5
        # messages). Stale images from earlier turns are stripped before sending to Kimi.
        # Checking the entire history caused permanent GPT routing after any view_image
        # call, which broke accumulated tool-call history format (400 from GPT).
        recent_has_images = _history_has_images(history[-5:]) if history else False
        # Select this turn's model (random from the tagged pool, else the plain
        # model) and its request-shaping behavior. The passed ``model`` is ignored
        # — local owns selection.
        selected = self._select_model()
        behavior = self._behavior_for(selected)
        # Vision fallback: route image turns to opus-4.7 (Bedrock) with a one-shot
        # description, then let the (text-only) local model proceed on the text.
        # Driven by the model's ``vision_provider`` setting; falls through to the
        # gpt-4o proxy path when unset or on any failure.
        if recent_has_images and behavior.vision_provider == "bedrock" and self._bedrock_vision() is not None:
            sigs = {s for _, _, s in _extract_images_from_history(history[-5:])}
            if sigs - self._described_img_sigs:
                try:
                    yield from self._vision_turn(system, history, reasoning_effort, behavior)
                    self._described_img_sigs |= sigs
                    return
                except Exception as e:
                    logger.warning("glm-5.2 opus vision fallback failed (%s); using gpt-4o proxy", e)
            else:
                # already described once — let GLM continue (images stripped below)
                recent_has_images = False
        if recent_has_images:
            client = self._openai_client
            actual_model = "gpt-4o"
            logger.info("LocalProvider: vision content in recent turn; routing to %r via proxy", actual_model)
            extra_body_kwargs: dict[str, Any] = {}
            # Send full history to GPT — it can handle image_url blocks.
            messages: list[dict[str, Any]] = [{"role": "system", "content": system}] + list(history)
        else:
            client = self._client
            actual_model = selected
            extra_body_kwargs = self._thinking_extra_body(reasoning_effort, behavior.thinking_style)
            # Strip any stale image blocks from history — these lab endpoints
            # don't support vision.
            clean_history = [_strip_images(m) for m in history]
            messages = [{"role": "system", "content": system}] + clean_history

        # GLM-5.1 text-tool mode: the native tools API degenerates GLM-5.1 into
        # "!!!!" (see _GLM_TEXT_TOOL_INSTRUCTION). Instead, describe the tools in
        # the system prompt and parse the tool call out of the reply text below.
        # GLM-5.2 fixed this (native tools work), so glm_native_tools bypasses the
        # workaround and uses the native tools= path (the else branch below).
        glm_text_mode = (
            behavior.thinking_style == "glm" and not recent_has_images and bool(tools)
            and not behavior.glm_native_tools
        )

        kwargs: dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **extra_body_kwargs,
        }
        if glm_text_mode:
            # Describe tools in the system prompt; do NOT send the native tools
            # array (that is what breaks GLM). Thinking OFF here: it yields a
            # consistent, parseable ``<tool_call>NAME {args}`` form, whereas
            # thinking-on intermittently garbles the arguments (leaked
            # <arg_key>/<arg_value> special tokens). Tool-calling turns trade
            # the visible reasoning trace for reliable tool extraction.
            messages[0] = {
                "role": "system",
                "content": system + _GLM_TEXT_TOOL_INSTRUCTION + _tools_as_text(tools),
            }
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            chat_tools = _translate_tools(tools)
            if chat_tools:
                kwargs["tools"] = chat_tools
                kwargs["tool_choice"] = "auto"

        # GLM endpoint path (not Kimi, not the gpt-4o vision proxy):
        #  - temperature per the vLLM recipe (1.0).
        #  - bound output so a degraded/flapping server can't hang the request
        #    indefinitely (see _GLM_MAX_OUTPUT_TOKENS).
        if behavior.thinking_style == "glm" and not recent_has_images:
            kwargs["temperature"] = behavior.temperature if behavior.temperature is not None else _GLM_TEMPERATURE
            kwargs["max_tokens"] = behavior.max_tokens if behavior.max_tokens is not None else _GLM_MAX_OUTPUT_TOKENS

        full_text = ""
        reasoning_text = ""  # GLM reasoning channel (parse fallback in text-tool mode)
        # {index: {"id": ..., "name": ..., "arguments": ...}}
        tc_buf: dict[int, dict[str, str]] = {}
        usage = None
        degenerated = False  # GLM "!!!!" single-token runaway detected

        stream = None
        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_text += delta.reasoning_content
                if delta.content:
                    full_text += delta.content
                    yield StreamEvent(type="text_delta", delta=delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_buf:
                            tc_buf[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_buf[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc_buf[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc_buf[idx]["arguments"] += tc_delta.function.arguments
                # Abort GLM's degenerate single-token runaway ("!!!!") as soon as
                # it appears, instead of streaming to the token cap (~7 min wasted
                # per occurrence). GLM-scoped.
                if glm_text_mode and (_is_degenerate_tail(full_text)
                                      or _is_degenerate_tail(reasoning_text)):
                    degenerated = True
                    break
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        if glm_text_mode:
            # Parse tool calls out of the model's reply text (content; fall back
            # to the reasoning channel). Store the raw reply as the assistant
            # message so the conversation stays coherent for the next turn.
            parsed = _parse_text_tool_calls(full_text) or _parse_text_tool_calls(reasoning_text)
            # GLM sometimes garbles its native tool-call arguments (leaked
            # <arg_key>/<arg_value> tokens) or degenerates into "!!!!". When it
            # clearly tried to call a tool but we couldn't extract one, retry a
            # couple of times. The retry is bounded (max_tokens) so a degenerate
            # retry terminates in ~1 min rather than running to the 32k cap.
            attempt = 0
            while (not parsed
                   and (degenerated or "<tool_call>" in (full_text + reasoning_text))
                   and attempt < 3):
                attempt += 1
                try:
                    # Streaming retry with the same degeneration abort, so a
                    # degenerate retry stops in ~ms (not at the token cap). The
                    # server flaps, so a few quick retries usually catch a good
                    # window and the model emits a real tool call.
                    rtext, rreason, degenerated = self._stream_collect(client, kwargs)
                    parsed = _parse_text_tool_calls(rtext) or _parse_text_tool_calls(rreason)
                    if parsed:
                        full_text = rtext  # store the successful retry as the turn
                except Exception as e:
                    logger.warning("LocalProvider glm text-tool retry failed: %s", e)
                    break
            tool_calls = []
            for name, args_str in parsed:
                cid = "glmtext-" + uuid.uuid4().hex[:16]
                tool_calls.append(ToolCall(call_id=cid, name=name, arguments=args_str))
                self._textmode_call_names[cid] = name
            # Never store a degenerate "!!!!" turn in history — it would prime the
            # next turn to degenerate too. Drop it to an empty assistant message;
            # the agent loop then nudges and the model retries cleanly.
            content_to_store = "" if (not parsed and degenerated) else (full_text or "")
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content_to_store}
        else:
            tool_calls = [
                ToolCall(
                    call_id=tc_buf[i]["id"],
                    name=tc_buf[i]["name"],
                    arguments=tc_buf[i]["arguments"],
                )
                for i in sorted(tc_buf)
            ]
            # Build the assistant message to append to history later via
            # append_response_to_history. Stored in raw_output_items in chat format.
            assistant_msg = {
                "role": "assistant",
                "content": full_text if full_text else (None if tool_calls else ""),
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ]

        input_tokens = 0
        output_tokens = 0
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        yield StreamEvent(
            type="done",
            response=Response(
                id="",
                text=full_text,
                tool_calls=tool_calls,
                has_web_search=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_output_items=[assistant_msg],
                model=actual_model,
            ),
        )

    def _vision_turn(
        self, system: str, history: list[dict[str, Any]], reasoning_effort: str,
        behavior: LocalModelBehavior,
    ) -> Iterator[StreamEvent]:
        """One-shot: send the recent image(s) to the vision provider (opus-4.7 via
        Bedrock) and return its textual description as this turn's assistant
        message. Stateless w.r.t. the chat history (avoids cross-provider history
        format issues) — exactly like the web-search proxy."""
        images = _extract_images_from_history(history[-5:])
        if not images:
            raise RuntimeError("no extractable image_url in recent history")
        img_blocks = [{"image": {"format": fmt, "source": {"bytes": raw}}}
                      for raw, fmt, _ in images]
        ctx = _recent_text_context(history[-5:])
        ask = ("Describe and analyze the image(s) above in thorough, decision-useful detail "
               "for an AI research agent that cannot see them: axes, scales, units, legends, "
               "trends, distributions, outliers/anomalies, and anything notable.")
        user_content = img_blocks + [{"text": (ctx + "\n\n" if ctx else "") + ask}]
        vsys = (system + "\n\nYou are interpreting images on behalf of a text-only model. "
                "Return a detailed textual description only.")
        last = None
        for ev in self._bedrock_vision().stream_response(
                model=behavior.vision_model, system=vsys,
                history=[{"role": "user", "content": user_content}],
                tools=[], reasoning_effort=(reasoning_effort or "low")):
            if ev.type == "text_delta":
                yield ev
            elif ev.type == "done":
                last = ev
        text = (getattr(last.response, "text", "") if last else "") or "(vision model returned no description)"
        text = "[Image interpreted by opus-4.7 vision fallback for text-only GLM]\n" + text
        it = getattr(last.response, "input_tokens", 0) if last else 0
        ot = getattr(last.response, "output_tokens", 0) if last else 0
        yield StreamEvent(
            type="done",
            response=Response(
                id="", text=text, tool_calls=[], has_web_search=False,
                input_tokens=it, output_tokens=ot,
                raw_output_items=[{"role": "assistant", "content": text}],
                model=behavior.vision_model,
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
        """Non-streaming completion via Chat Completions API.

        Used for bounded utility work (summarization). For GLM, thinking is
        disabled here: it is on-by-default and would spend much of the bounded
        ``max_tokens`` budget on internal CoT (verified on the live endpoint:
        a 4k-budget summary used 615 tokens with thinking on vs 66 with it
        off), risking a truncated or empty summary. Kimi's thinking budget is
        output-only and separate, so it needs no special handling here.
        """
        actual_model = self._select_model()
        beh = self._behavior_for(actual_model)
        chat_messages = [{"role": "system", "content": system}] + messages
        # GLM: temperature per the recipe, and disable thinking for this bounded
        # utility call (the recipe's "thinking off" form) so the token budget
        # goes to the summary rather than chain-of-thought.
        glm_extra = (
            {"temperature": beh.temperature if beh.temperature is not None else _GLM_TEMPERATURE,
             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if beh.thinking_style == "glm" else {}
        )
        try:
            logger.info(f"LocalProvider.complete: calling {actual_model!r} with {len(chat_messages)} messages, max_tokens={max_tokens}")
            response = self._client.chat.completions.create(
                model=actual_model,
                messages=chat_messages,
                max_tokens=max_tokens,
                **glm_extra,
            )
        except Exception as e:
            if "max_tokens" in str(e):
                response = self._client.chat.completions.create(
                    model=actual_model,
                    messages=chat_messages,
                    max_completion_tokens=max_tokens,
                    **glm_extra,
                )
            else:
                raise
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # History helpers — Chat API format
    # ------------------------------------------------------------------

    def build_user_items(self, message: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": message}]

    def build_tool_result_items(
        self,
        results: list[dict[str, Any]],
        images: list[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build tool-result messages.

        Format follows how the *originating* call was made, not the current
        model: a GLM-5.1 text-tool call was parsed from reply text and stored as
        a plain assistant turn (no native tool_call_ids), so its results go back
        as a plain ``user`` message; native calls (Kimi, GLM-5.2, others) use
        ``role: tool``. Text-mode calls are recognized by their tracked call_ids
        (``_textmode_call_names``), which stays correct even though the model is
        selected per request.
        """
        if any(r.get("call_id", "") in self._textmode_call_names for r in results):
            parts: list[str] = []
            for r in results:
                name = self._textmode_call_names.get(r.get("call_id", ""), "tool")
                parts.append(f"Result of tool `{name}`:\n{r['output']}")
            text = "\n\n".join(parts) if parts else "(no tool output)"
            if images:
                text += "\n\n[An image was produced; it cannot be shown to this model.]"
            return [{"role": "user", "content": text}]

        items: list[dict[str, Any]] = []
        for r in results:
            items.append({
                "role": "tool",
                "tool_call_id": r["call_id"],
                "content": r["output"],
            })
        if images:
            image_content: list[dict[str, Any]] = []
            for b64_data, media_type in images:
                image_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                })
            image_content.append({
                "type": "text",
                "text": "Here is the image you requested. Analyze it carefully.",
            })
            items.append({"role": "user", "content": image_content})
        return items

    def append_response_to_history(
        self,
        history: list[dict[str, Any]],
        response: Response,
    ) -> None:
        """Append the assistant turn (chat format) to history."""
        for item in response.raw_output_items:
            history.append(item)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _history_has_images(history: list[dict[str, Any]]) -> bool:
    """Return True if any message in history contains image_url content."""
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _extract_images_from_history(messages: list[dict[str, Any]]) -> list[tuple[bytes, str, str]]:
    """Pull image_url data-URL blocks out of recent messages as (bytes, format, sig).
    Used only by the GLM-5.2 opus vision fallback."""
    import base64 as _b64, hashlib as _hl
    out: list[tuple[bytes, str, str]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not (url.startswith("data:") and "," in url):
                continue
            head, b64 = url.split(",", 1)
            fmt = head.split("image/", 1)[1].split(";", 1)[0] if "image/" in head else "png"
            fmt = {"jpg": "jpeg"}.get(fmt, fmt) or "png"
            try:
                raw = _b64.b64decode(b64)
            except Exception:
                continue
            out.append((raw, fmt, _hl.md5(b64.encode()).hexdigest()))
    return out


def _recent_text_context(messages: list[dict[str, Any]]) -> str:
    """Concatenate recent non-image text (for the vision prompt's context). Capped."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") != "image_url" and (p.get("text") or "").strip():
                    parts.append(p["text"].strip())
    return "\n".join(parts[-6:])[:2000]


def _is_degenerate_tail(text: str, run: int = 80) -> bool:
    """True if ``text`` ends with a long run (>= ``run``) of one non-space char.

    GLM degenerates into repeated single tokens (e.g. ``!!!!``) on the broken
    tool path; detecting the run lets the stream abort early instead of
    generating to the token cap.
    """
    if len(text) < run:
        return False
    tail = text[-run:]
    c = tail[0]
    return c not in " \t\r\n" and tail == c * run


def _tools_as_text(tools: list[dict[str, Any]]) -> str:
    """Render Responses-format tool schemas as plain text for the prompt.

    Used by GLM text-tool mode (see _GLM_TEXT_TOOL_INSTRUCTION) instead of the
    native ``tools`` array. Input is the Responses-API tool format the rest of
    the system uses: ``{"type":"function","name","description","parameters"}``
    plus ``{"type":"web_search"}``.
    """
    lines: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "web_search":
            lines.append('### web_search\nSearch the web for current information.\n'
                         'parameters JSON Schema: {"type":"object","properties":'
                         '{"query":{"type":"string"}},"required":["query"]}')
            continue
        if t.get("type") != "function":
            continue
        name = t.get("name", "")
        desc = (t.get("description", "") or "").strip()
        params = t.get("parameters", {}) or {}
        lines.append(f"### {name}\n{desc}\nparameters JSON Schema: {json.dumps(params)}")
    return "\n\n".join(lines)


def _brace_object_at(text: str, start: int) -> tuple[Any, int]:
    """Parse the balanced ``{...}`` starting at index ``start`` (string-aware).

    Returns ``(parsed_or_None, end_index)`` where end_index is the position of
    the matching ``}`` (or the last index if unbalanced).
    """
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:pos + 1]), pos
                    except Exception:
                        return None, pos
    return None, len(text) - 1


def _skip_whitespace(text: str, pos: int) -> int:
    """Advance ``pos`` past any leading whitespace in ``text``."""
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1
    return pos


def _arguments_json(value: Any) -> str:
    """Normalize a tool-call ``arguments`` value to a JSON string."""
    return value if isinstance(value, str) else json.dumps(value)


def _named_call_from_object(obj: Any) -> tuple[str, str] | None:
    """Return ``(name, arguments_json)`` if ``obj`` is a ``{"name","arguments"}`` call."""
    if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "arguments" in obj:
        return obj["name"], _arguments_json(obj["arguments"])
    return None


def _parse_native_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract GLM's native ``<tool_call>NAME {args...}`` tool calls.

    The model falls back to its trained tool-call form even when asked for JSON;
    with thinking off the args follow as a clean ``{...}`` object. The name is
    taken right after the tag; the first balanced ``{...}`` after it is the
    arguments. If that object is itself a ``{"name","arguments"}`` call, its
    inner name/args win over the tag name.
    """
    calls: list[tuple[str, str]] = []
    tag = "<tool_call>"
    search_from = text.find(tag)
    while search_from != -1:
        cursor = _skip_whitespace(text, search_from + len(tag))
        name_start = cursor
        while cursor < len(text) and (text[cursor].isalnum() or text[cursor] == "_"):
            cursor += 1
        name = text[name_start:cursor]
        if name:
            brace = text.find("{", cursor)
            if brace != -1:
                obj, _ = _brace_object_at(text, brace)
                if isinstance(obj, dict):
                    calls.append(_named_call_from_object(obj) or (name, json.dumps(obj)))
        search_from = text.find(tag, cursor)
    return calls


def _parse_json_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract instructed bare ``{"name": "...", "arguments": {...}}`` call objects."""
    calls: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos] != "{":
            pos += 1
            continue
        obj, end = _brace_object_at(text, pos)
        call = _named_call_from_object(obj)
        if call is not None:
            calls.append(call)
            pos = end + 1
        else:
            pos += 1
    return calls


def _parse_text_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract tool calls emitted as text by GLM in text-tool mode.

    Returns ``[(tool_name, arguments_json_string), ...]``. Handles two shapes,
    preferring GLM's native tag form and falling back to the instructed JSON:
      1. GLM's native ``<tool_call>NAME {args...}`` (see _parse_native_tool_calls).
      2. The instructed ``{"name": "...", "arguments": {...}}`` JSON object.
    """
    if not text:
        return []
    return _parse_native_tool_calls(text) or _parse_json_tool_calls(text)


def _strip_images(msg: dict[str, Any]) -> dict[str, Any]:
    """Remove image_url content blocks — Chat endpoints without vision support."""
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    text_parts = [p for p in content if isinstance(p, dict) and p.get("type") != "image_url"]
    if len(text_parts) == len(content):
        return msg
    return {**msg, "content": text_parts if text_parts else ""}


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses-API tool schemas to Chat API function schemas.

    Input (Responses API style, what the rest of the system uses):
      ``[{"type": "function", "name": "...", "description": "...", "parameters": {...}},
         {"type": "web_search"}]``

    Output (Chat API style):
      ``[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]``

    ``web_search`` is replaced with a proxy function schema so the
    dispatcher routes it through OpenAI.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type", "")
        if t_type == "web_search":
            out.append(_WEB_SEARCH_FN_SCHEMA)
        elif t_type == "function":
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            })
    return out
