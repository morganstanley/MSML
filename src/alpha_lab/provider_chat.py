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
import re
import uuid
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from alpha_lab.provider import Response, StreamEvent, ToolCall

logger = logging.getLogger("alpha_lab.provider_chat")

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

# Backstop on lab-model streaming output tokens. A degraded/flapping vLLM
# server can stream non-terminating garbage; with no max_tokens the request
# runs to max_model_len (1,048,576 on these deployments), hanging the
# dispatcher for many minutes (the per-chunk read timeout never fires because
# garbage keeps dribbling). The PRIMARY protection is _is_degenerate_tail,
# which aborts a repeating stream mid-flight; this is only the backstop.
#
# Raised 32,000 -> 200,000 on 2026-08-09 by user order, after measuring that
# the old value truncated legitimate work:
#   * live, same one-line question, natural finish (finish_reason="stop"):
#     34,731 / 40,486 / 32,367 / 60,069 completion tokens — all above 32,000.
#   * historical: 54,524 GLM replies across 30 runs, none above 32,000 and 22
#     landing exactly on it (0.040%) — the ceiling was cutting real answers.
#   * the captured long reply was coherent (271 thinking lines, 257 distinct,
#     ending on a conclusion), not degenerate.
# 200,000 is >3x the largest natural finish observed and still ~5x below the
# context length, so a degenerate stream stays bounded.
_GLM_MAX_OUTPUT_TOKENS = 200000

# Server-side pattern for messages[].tool_calls[].function.name. A stored
# name outside it draws a 400 on EVERY subsequent request that replays the
# turn, until the item leaves the context window.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


def _screen_tool_calls(calls, model, finish_reason=""):
    """Validate tool calls BEFORE they enter replayable history.

    Measured 2026-08-08 over 51 runs / 201,654 tool calls: 33 calls whose
    arguments were not valid JSON (the stream ended mid-string) and 12 whose
    name violated the server pattern (deepseek fusing arguments into the
    name slot: ``read_file" path="…``; GLM-5.1-era unicode garbage). Each
    one, stored verbatim, drew "Unterminated string" / "function.name does
    not match pattern" 400s on every later request carrying it. Drop the
    broken call loudly instead of storing it — the agent loop's nudge
    machinery makes the model retry, the same recovery path degenerate
    turns already use.

    Returns ``(kept, dropped_count)``.
    """
    kept = []
    dropped = 0
    for tc in calls:
        reason = ""
        if not _TOOL_NAME_RE.match(tc.name or ""):
            reason = f"invalid tool name {(tc.name or '')[:60]!r}"
        else:
            try:
                json.loads(tc.arguments or "{}")
            except Exception as e:
                reason = (f"unparseable JSON arguments "
                          f"({e}; {len(tc.arguments or '')} chars)")
        if reason:
            dropped += 1
            logger.warning(
                "ChatProvider: DROPPED tool call from %r before history: %s%s"
                " — model will be nudged to retry",
                model, reason,
                f" [finish_reason={finish_reason}]" if finish_reason else "",
            )
        else:
            kept.append(tc)
    return kept, dropped

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
            "Routed through OpenAI's web_search_preview; available regardless of "
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


class ChatProvider:
    """Provider backed by the OpenAI Chat Completions API.

    Works with any OpenAI-compatible endpoint including lab-hosted models
    (gpu-host-1 / GLM-5.1, Kimi, etc.) that only expose ``/v1/chat/completions``.

    History is maintained in the standard Chat API messages format.
    Web search is proxied through ``openai_client_for_proxy`` (a real OpenAI
    client) using the same pattern as ``GrokProvider``.
    """

    def __init__(
        self,
        client: OpenAI,
        openai_client_for_proxy: OpenAI,
        thinking_style: str = "kimi",
        glm_native_tools: bool = False,
        vision_provider: Any | None = None,
        vision_model: str = "claude-opus-4-8",
        vision_dialect: str = "anthropic",
    ) -> None:
        self._client = client
        self._proxy = openai_client_for_proxy
        # Vision fallback: GLM-5.2 is text-only. When it is the driving model
        # (glm_native_tools) and a turn carries images, route that turn to a
        # vision-capable provider (opus via the native Anthropic gateway by
        # default, or Bedrock) — analogous to web search proxying through
        # OpenAI. One-shot describe; GLM continues with the textual
        # description. None -> fall back to the gpt-4o proxy path.
        # ``vision_dialect`` names the content-block format the wired provider
        # expects ("anthropic" Messages blocks / "bedrock" Converse blocks) —
        # set by get_provider() to match what it built.
        self._vision_provider = vision_provider
        self._vision_model = vision_model
        self._vision_dialect = vision_dialect
        self._described_img_sigs: set[str] = set()
        # Which "thinking" dialect this endpoint speaks. "kimi" (moonshot):
        # graded internal-CoT budget via top-level enable_thinking/thinking_budget.
        # "glm" (zai-org/GLM-5.1 on vLLM): thinking is on-by-default and binary,
        # toggled via chat_template_kwargs.enable_thinking; a graded budget is
        # not honored. See _thinking_extra_body.
        self._thinking_style = thinking_style
        # GLM-5.1's native tools API degenerated into "!!!!", so "glm" defaults to
        # a text-tool workaround (tools described in the prompt, the call parsed
        # from reply text). GLM-5.2 handles native tool calls correctly, so set
        # glm_native_tools=True (env GLM_NATIVE_TOOLS) to use the standard tools=
        # path with role:tool results for it.
        self._glm_native_tools = glm_native_tools
        # Replay each turn's reasoning into the next request's assistant
        # history message for EVERY chat model (CHAT_REPLAY_REASONING=0 opts
        # out) — the model should see its own past traces, as the Anthropic
        # (thinking blocks) and OpenAI (encrypted reasoning items) paths do.
        # See _should_replay_reasoning for per-endpoint rendering caveats.
        import os as _os
        self._replay_reasoning = _os.environ.get(
            "CHAT_REPLAY_REASONING", "1") != "0"
        self._resolved_model: str | None = None  # cached auto-discovery result
        # call_id -> tool name, for GLM text-tool mode: lets build_tool_result_items
        # label each result with the tool that produced it.
        self._textmode_call_names: dict[str, str] = {}

    def _should_replay_reasoning(self, actual_model: str) -> bool:
        """Whether this turn's reasoning goes back into the next request's
        assistant history message.

        Default ON for every chat model (CHAT_REPLAY_REASONING=0 opts out):
        the model should see its own past traces, like the Anthropic path
        (thinking blocks) and the OpenAI path (encrypted reasoning items)
        already do. Before 2026-08-07 only kimi-k3 replayed; GLM and
        deepseek ran every campaign blind to their own reasoning
        (d4_glm_cond: 0 of 4,744 requests carried reasoning while 1,275
        responses produced it; d4_dsv4_cond_direct: 0 of 2,556 vs 1,865).

        kimi-k3 replays regardless of the flag — its model card (§6 Model
        Usage) requires the assistant message passed back INCLUDING
        reasoning_content for multi-turn and tool-call flows.

        The endpoint's chat template decides whether the field is RENDERED
        into the prompt: kimi-k3 on gpu-host-1:8000 renders it (prompt_tokens
        179 vs 153 in a live two-turn probe, 2026-08-07); deepseek-v4-flash
        on gpu-host-2:8001 accepts the field but its current template DROPS it
        (39 = 39 in the same probe) — harness side is correct either way,
        but making deepseek actually see its traces needs a serving-side
        template change. GLM's endpoint was down on 2026-08-07, untested.
        """
        if self._thinking_style == "mlrllm" and "kimi-k3" in actual_model.lower():
            return True
        return self._replay_reasoning

    def _thinking_extra_body(self, reasoning_effort: str, model: str = "") -> dict[str, Any]:
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
        if self._thinking_style == "glm":
            kw: dict[str, Any] = {}
            if (reasoning_effort or "none") == "none":
                kw["enable_thinking"] = False
            if self._replay_reasoning:
                # GLM's own chat template KNOWS reasoning_content but renders
                # it for history messages only when told not to clear it —
                # by default replayed reasoning is silently dropped before
                # tokenization. clear_thinking=false flips the gate (verified
                # live 2026-08-07 on gpu-host-2:8000: prompt_tokens 73 vs 47 for
                # the same two-turn history with replayed reasoning).
                kw["clear_thinking"] = False
            return {"extra_body": {"chat_template_kwargs": kw}} if kw else {}
        if self._thinking_style == "mlrllm":
            # Lab-hosted OpenAI-compatible endpoints (mlrllm gateway or the
            # direct vLLM deployments), dialect keyed per model:
            # - deepseek-v4-flash: thinking is OFF by default and the model
            #   degrades without it; the vLLM chat_template_kwargs.thinking
            #   toggle enables it (verified live 2026-08-05 on
            #   gpu-host-2.example.com:8001: baseline wrong / rc=0, thinking on
            #   correct / rc=9038). Plain thinking=true out-thought the
            #   graded reasoning_effort forms (9038 vs 3943 rc chars) — no
            #   effort cap is sent.
            # - kimi-k3: thinking always on; the OFFICIAL control (model card
            #   §6 Model Usage) is top-level reasoning_effort low|high|max,
            #   default max (verified live 2026-08-05 on gpu-host-1:8000:
            #   low→785 rc chars, max→2443, all correct answers).
            # - gemma-4: no reasoning channel; nothing to send.
            m = (model or "").lower()
            if "deepseek" in m:
                on = (reasoning_effort or "none") != "none"
                kw: dict[str, Any] = {"thinking": on}
                if self._replay_reasoning:
                    # DeepSeek's own encoder (encoding_dsv4.py, shipped in the
                    # model dir) drops history reasoning unless told not to:
                    # drop_thinking=false renders replayed reasoning_content
                    # (verified live 2026-08-07 on gpu-host-2:8001: prompt_tokens
                    # 66 vs 39 for the same two-turn history).
                    kw["drop_thinking"] = False
                return {"extra_body": {"chat_template_kwargs": kw}}
            if "kimi-k3" in m:
                eff = {"none": "low", "low": "low", "medium": "high",
                       "high": "high", "max": "max"}.get(
                           (reasoning_effort or "max"), "max")
                return {"extra_body": {"reasoning_effort": eff}}
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
        """
        if model != "auto":
            return model
        if self._resolved_model is not None:
            return self._resolved_model
        try:
            ids = [m.id for m in self._client.models.list().data]
            if ids:
                self._resolved_model = ids[0]
                logger.info("ChatProvider: auto-discovered model %r", self._resolved_model)
                return self._resolved_model
            logger.warning("ChatProvider: /v1/models returned empty list; falling back to default")
        except Exception as e:
            logger.warning("ChatProvider: model auto-discovery failed: %s; falling back to default", e)
        self._resolved_model = "zai-org/GLM-5.1"
        return self._resolved_model

    @property
    def openai_client(self) -> OpenAI:
        """Real OpenAI client used for web_search proxy."""
        return self._proxy

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
                _rd = (getattr(d, "reasoning_content", None)
                       or getattr(d, "reasoning", None))
                if _rd:
                    reasoning += _rd
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
        # GLM-5.2 vision fallback: route image turns to the wired vision
        # provider (opus via the native Anthropic gateway by default; Bedrock
        # via GLM_VISION_PROVIDER=bedrock) for a one-shot description, then let
        # GLM proceed on the text. Only when GLM-5.2 is the driver
        # (glm_native_tools) and a vision provider is wired. Falls through to
        # the gpt-4o proxy path on any failure.
        if recent_has_images and self._glm_native_tools and self._vision_provider is not None:
            sigs = {s for _, _, s in _extract_images_from_history(history[-5:])}
            if sigs - self._described_img_sigs:
                try:
                    yield from self._vision_turn(system, history, reasoning_effort)
                    self._described_img_sigs |= sigs
                    return
                except Exception as e:
                    logger.warning(
                        "glm-5.2 %s vision fallback failed (%s); using gpt-4o proxy",
                        self._vision_dialect, e)
            else:
                # already described once — let GLM continue (images stripped below)
                recent_has_images = False
        if recent_has_images:
            client = self._proxy
            actual_model = "gpt-4o"
            logger.info("ChatProvider: vision content in recent turn; routing to %r via proxy", actual_model)
            extra_body_kwargs: dict[str, Any] = {}
            # Send the history to GPT — it can handle image_url blocks. But
            # gpt-4o's context (128k tokens) is far below the lab models', so a
            # long agent history that the lab model handles fine 400s on the
            # proxy and kills the whole turn (seen 2026-08-05: 151,084 tokens
            # from a kimi-k3 builder aborted msml's Phase 2). Bound the proxy
            # request to the system message + the longest history tail that fits.
            messages: list[dict[str, Any]] = _fit_proxy_context(
                [{"role": "system", "content": system}] + list(history))
        else:
            client = self._client
            actual_model = self._resolve_model(model)
            extra_body_kwargs = self._thinking_extra_body(reasoning_effort, actual_model)
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
            self._thinking_style == "glm" and not recent_has_images and bool(tools)
            and not self._glm_native_tools
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
        if self._thinking_style == "glm" and not recent_has_images:
            kwargs["temperature"] = _GLM_TEMPERATURE
            kwargs["max_tokens"] = _GLM_MAX_OUTPUT_TOKENS
        # mlrllm gateway models get the same runaway-stream guarantee (no
        # temperature override — only the termination bound). gemma-4 was
        # observed streaming an unterminated phrase-repetition loop; without
        # a cap the request never ends (16.9 GB strategist record, 2026-08-05).
        if self._thinking_style == "mlrllm" and not recent_has_images:
            kwargs["max_tokens"] = _GLM_MAX_OUTPUT_TOKENS

        full_text = ""
        reasoning_text = ""  # GLM reasoning channel (parse fallback in text-tool mode)
        # {index: {"id": ..., "name": ..., "arguments": ...}}
        tc_buf: dict[int, dict[str, str]] = {}
        usage = None
        finish_reason = ""  # server's own stop reason (stop/length/tool_calls…)
        degenerated = False  # GLM "!!!!" single-token runaway detected
        # The server's own identity claim. Two deployments of "the same"
        # model are indistinguishable post-hoc without this (2026-08-06:
        # proving which GLM served a July run took distribution forensics
        # because nothing on disk carried the server's model string or the
        # endpoint) — persist both via raw_response below.
        server_meta: dict[str, Any] = {}
        endpoint = str(getattr(client, "base_url", "") or "")

        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not server_meta:
                    for k in ("model", "id", "created", "system_fingerprint"):
                        v = getattr(chunk, k, None)
                        if v:
                            server_meta[k] = v
                if not chunk.choices:
                    continue
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                delta = chunk.choices[0].delta
                # Thinking channel: GLM-5.1-era servers stream it as
                # `reasoning_content`; the gpu-host-1 vLLM build streams `reasoning`.
                _rdelta = (getattr(delta, "reasoning_content", None)
                           or getattr(delta, "reasoning", None))
                if _rdelta:
                    reasoning_text += _rdelta
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
                if (glm_text_mode or self._thinking_style == "mlrllm") and (
                        _is_degenerate_tail(full_text)
                        or _is_degenerate_tail(reasoning_text)):
                    degenerated = True
                    break
        finally:
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
                    logger.warning("ChatProvider glm text-tool retry failed: %s", e)
                    break
            tool_calls = []
            for name, args_str in parsed:
                cid = "glmtext-" + uuid.uuid4().hex[:16]
                tool_calls.append(ToolCall(call_id=cid, name=name, arguments=args_str))
            tool_calls, _ = _screen_tool_calls(tool_calls, actual_model, finish_reason)
            for tc in tool_calls:
                self._textmode_call_names[tc.call_id] = tc.name
            # Never store a degenerate "!!!!" turn in history — it would prime the
            # next turn to degenerate too. Drop it to an empty assistant message;
            # the agent loop then nudges and the model retries cleanly.
            content_to_store = "" if (not parsed and degenerated) else (full_text or "")
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content_to_store}
        else:
            if degenerated:
                # Never store a degenerate repetition turn in history — it
                # primes the next turn to loop too, and each subsequent
                # request re-ships (and re-logs) the garbage. Drop it loudly;
                # the agent loop's empty-response nudge retries the turn.
                logger.warning(
                    "ChatProvider: aborted degenerate repetition stream from %r "
                    "(%d text chars, %d reasoning chars dropped)",
                    actual_model, len(full_text), len(reasoning_text),
                )
                full_text = ""
                reasoning_text = ""
            tool_calls = [
                ToolCall(
                    call_id=tc_buf[i]["id"],
                    name=tc_buf[i]["name"],
                    arguments=tc_buf[i]["arguments"],
                )
                for i in sorted(tc_buf)
            ]
            tool_calls, _ = _screen_tool_calls(tool_calls, actual_model, finish_reason)
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
            if reasoning_text and self._should_replay_reasoning(actual_model):
                assistant_msg["reasoning_content"] = reasoning_text

        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read = 0
        cache_write = 0
        usage_raw: dict[str, Any] = {}
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            try:
                usage_raw = usage.model_dump(exclude_none=True)
            except Exception:
                usage_raw = {}
            # Lab endpoints report prefix-cache counters under
            # prompt_tokens_details (observed since the 2026-08-07 server
            # restarts: cached_tokens + the vLLM-specific
            # created_cache_tokens). Surface them on the Response so the
            # runtime token ledger and cost dashboards see cache traffic —
            # they read cache_read/write_input_tokens, which stayed 0 for
            # every lab model while the analysis layer had to compensate.
            ptd = usage_raw.get("prompt_tokens_details") or {}
            cache_read = int(ptd.get("cached_tokens") or 0)
            cache_write = int(ptd.get("created_cache_tokens")
                              or ptd.get("cache_write_tokens") or 0)

        yield StreamEvent(
            type="done",
            response=Response(
                id="",
                text=full_text,
                tool_calls=tool_calls,
                has_web_search=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_read_input_tokens=cache_read,
                cache_write_input_tokens=cache_write,
                stop_reason=finish_reason,
                raw_output_items=[assistant_msg],
                usage_raw=usage_raw,
                # The reasoning channel (delta.reasoning_content) is not part of
                # the chat-format assistant message, so it would otherwise leave
                # no trace anywhere. ApiResponseEvent persists raw_response
                # verbatim to the agent's jsonl — the only on-disk copy.
                # Same for the server's identity claim and the endpoint that
                # answered: without them "which deployment served this run?"
                # is unanswerable after the deployment is gone.
                raw_response=(
                    ({"reasoning_content": reasoning_text}
                     if reasoning_text else {})
                    | ({"server": server_meta} if server_meta else {})
                    | ({"endpoint": endpoint} if endpoint else {})
                ),
            ),
        )

    def _vision_turn(
        self, system: str, history: list[dict[str, Any]], reasoning_effort: str
    ) -> Iterator[StreamEvent]:
        """One-shot: send the recent image(s) to the wired vision provider and
        return its textual description as this turn's assistant message.
        Stateless w.r.t. the chat history (avoids cross-provider history
        format issues) — exactly like the web-search proxy. Content blocks are
        built in the provider's own dialect: Anthropic Messages format or
        Bedrock Converse format."""
        images = _extract_images_from_history(history[-5:])
        if not images:
            raise RuntimeError("no extractable image_url in recent history")
        ctx = _recent_text_context(history[-5:])
        ask = ("Describe and analyze the image(s) above in thorough, decision-useful detail "
               "for an AI research agent that cannot see them: axes, scales, units, legends, "
               "trends, distributions, outliers/anomalies, and anything notable.")
        prompt_text = (ctx + "\n\n" if ctx else "") + ask
        if self._vision_dialect == "bedrock":
            img_blocks: list[dict[str, Any]] = [
                {"image": {"format": fmt, "source": {"bytes": raw}}}
                for raw, fmt, _ in images]
            user_content = img_blocks + [{"text": prompt_text}]
        else:  # anthropic Messages format
            import base64 as _b64
            img_blocks = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": f"image/{fmt}",
                            "data": _b64.b64encode(raw).decode()}}
                for raw, fmt, _ in images]
            user_content = img_blocks + [{"type": "text", "text": prompt_text}]
        vsys = (system + "\n\nYou are interpreting images on behalf of a text-only model. "
                "Return a detailed textual description only.")
        last = None
        for ev in self._vision_provider.stream_response(
                model=self._vision_model, system=vsys,
                history=[{"role": "user", "content": user_content}],
                tools=[], reasoning_effort=(reasoning_effort or "low")):
            if ev.type == "text_delta":
                yield ev
            elif ev.type == "done":
                last = ev
        text = (getattr(last.response, "text", "") if last else "") or "(vision model returned no description)"
        text = (f"[Image interpreted by {self._vision_model} vision fallback "
                f"for text-only GLM]\n") + text
        it = getattr(last.response, "input_tokens", 0) if last else 0
        ot = getattr(last.response, "output_tokens", 0) if last else 0
        yield StreamEvent(
            type="done",
            response=Response(
                id="", text=text, tool_calls=[], has_web_search=False,
                input_tokens=it, output_tokens=ot, reasoning_tokens=0,
                raw_output_items=[{"role": "assistant", "content": text}],
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
        actual_model = self._resolve_model(model)
        chat_messages = [{"role": "system", "content": system}] + messages
        # GLM: temperature per the recipe, and disable thinking for this bounded
        # utility call (the recipe's "thinking off" form) so the token budget
        # goes to the summary rather than chain-of-thought.
        glm_extra = (
            {"temperature": _GLM_TEMPERATURE,
             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if self._thinking_style == "glm" else {}
        )
        if self._thinking_style == "mlrllm" and "deepseek" in actual_model.lower():
            # Bounded utility call: spend the budget on the summary, not CoT.
            glm_extra = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
        if self._thinking_style == "mlrllm" and "kimi-k3" in actual_model.lower():
            # kimi-k3 cannot disable thinking; its official graded control
            # caps the internal budget for this bounded utility call instead.
            glm_extra = {"extra_body": {"reasoning_effort": "low"}}
        try:
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

        For GLM-5.1 (text-tool mode) the tool call was parsed from the reply text
        and the assistant turn was stored as plain text — there were no native
        tool_call_ids — so results go back as a plain ``user`` message. For Kimi,
        GLM-5.2 (native tools), and others, native ``role: tool`` messages are used.
        """
        if self._thinking_style == "glm" and not self._glm_native_tools:
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


# ~110k tokens at ~4 chars/token — headroom under gpt-4o's 128k context, so a
# vision-proxy request built from a longer lab-model history always fits.
_PROXY_CONTEXT_CHAR_BUDGET = 440_000


def _fit_proxy_context(
    messages: list[dict[str, Any]], budget: int = _PROXY_CONTEXT_CHAR_BUDGET
) -> list[dict[str, Any]]:
    """Bound a vision-proxy request to ``budget`` serialized chars.

    Keeps ``messages[0]`` (the system message) plus the longest tail of the
    rest that fits — the tail holds the fresh image turn that triggered the
    proxy route, so recency is what matters. The final message is always kept
    even if it alone busts the budget. Leading orphan ``tool`` results (whose
    assistant tool-call turn got trimmed) are dropped so the Chat API doesn't
    reject the pairing. No-op (same list) when everything fits; trims loudly
    otherwise.
    """
    def _size(m: dict[str, Any]) -> int:
        try:
            return len(json.dumps(m, default=str))
        except (TypeError, ValueError):
            return len(str(m))

    if sum(_size(m) for m in messages) <= budget:
        return messages
    head, rest = messages[0], messages[1:]
    used = _size(head)
    kept: list[dict[str, Any]] = []
    for m in reversed(rest):
        s = _size(m)
        if kept and used + s > budget:
            break
        kept.append(m)
        used += s
    kept.reverse()
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    logger.warning(
        "ChatProvider: vision-proxy context trimmed %d -> %d messages "
        "(~%d chars) to fit gpt-4o's 128k window",
        len(messages), 1 + len(kept), used,
    )
    return [head] + kept


def _is_degenerate_tail(text: str, run: int = 80, window: int = 1200) -> bool:
    """True if ``text`` ends in degenerate repetition.

    Two forms, both observed on live lab endpoints:

    - a long run (>= ``run``) of one non-space char — GLM's broken-tool-path
      collapse into repeated single tokens (e.g. ``!!!!``);
    - a short phrase repeated over the last ``window`` chars — gemma-4 on
      the mlrllm gateway looped "**Wait, I'll call the tool.**" without
      terminating, which unchecked produced a 16.9 GB strategist record
      (2026-08-05). Detected by counting occurrences of the trailing 40
      chars inside the window: a looping stream repeats them every cycle,
      while legit prose/code virtually never contains the same exact 40-char
      string 6+ times in its last 1200 chars. Worst case a false abort costs
      one bounded retry/nudge, never a runaway.
    """
    if len(text) < run:
        return False
    tail = text[-run:]
    c = tail[0]
    if c not in " \t\r\n" and tail == c * run:
        return True
    if len(text) >= window:
        t = text[-window:]
        unit = t[-40:]
        if unit.strip() and t.count(unit) >= 6:
            return True
    return False


def _tools_as_text(tools: list[dict[str, Any]]) -> str:
    """Render Responses-format tool schemas as plain text for the prompt.

    Used by GLM text-tool mode (see _GLM_TEXT_TOOL_INSTRUCTION) instead of the
    native ``tools`` array. Input is the Responses-API tool format the rest of
    the system uses: ``{"type":"function","name","description","parameters"}``
    plus ``{"type":"web_search_preview"}``.
    """
    lines: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "web_search_preview":
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


def _brace_object_at(text: str, i: int) -> tuple[Any, int]:
    """Parse the balanced ``{...}`` starting at index ``i`` (string-aware).

    Returns ``(parsed_or_None, end_index)`` where end_index is the position of
    the matching ``}`` (or the last index if unbalanced).
    """
    depth, j, in_str, esc = 0, i, False, False
    while j < len(text):
        ch = text[j]
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1]), j
                    except Exception:
                        return None, j
        j += 1
    return None, len(text) - 1


def _parse_text_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract tool calls emitted as text by GLM in text-tool mode.

    Returns ``[(tool_name, arguments_json_string), ...]``. Handles two shapes:
      1. GLM's native ``<tool_call>NAME {args...}`` (the model falls back to its
         trained tool-call form even when asked for JSON; with thinking off the
         args follow as a clean ``{...}`` object). The name is taken right after
         the tag; the first balanced ``{...}`` after it is the arguments.
      2. The instructed ``{"name": "...", "arguments": {...}}`` JSON object.
    """
    if not text:
        return []
    out: list[tuple[str, str]] = []

    # 1) Native <tool_call>NAME {args}
    tag = "<tool_call>"
    k = text.find(tag)
    while k != -1:
        p = k + len(tag)
        while p < len(text) and text[p] in " \t\n\r":
            p += 1
        s = p
        while p < len(text) and (text[p].isalnum() or text[p] == "_"):
            p += 1
        name = text[s:p]
        if name:
            b = text.find("{", p)
            if b != -1:
                obj, _ = _brace_object_at(text, b)
                if isinstance(obj, dict):
                    if isinstance(obj.get("name"), str) and "arguments" in obj:
                        a = obj["arguments"]
                        out.append((obj["name"], a if isinstance(a, str) else json.dumps(a)))
                    else:
                        out.append((name, json.dumps(obj)))
        k = text.find(tag, p)
    if out:
        return out

    # 2) Bare {"name": ..., "arguments": ...}
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        obj, end = _brace_object_at(text, i)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "arguments" in obj:
            a = obj["arguments"]
            out.append((obj["name"], a if isinstance(a, str) else json.dumps(a)))
            i = end + 1
        else:
            i += 1
    return out


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
         {"type": "web_search_preview"}]``

    Output (Chat API style):
      ``[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]``

    ``web_search_preview`` is replaced with a proxy function schema so the
    dispatcher routes it through OpenAI.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type", "")
        if t_type == "web_search_preview":
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
