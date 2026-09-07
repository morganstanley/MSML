"""Unit tests for the xAI / grok provider helpers.

Pure-function / staticmethod coverage — no network, no client construction —
for the grok-specific behaviour: effort clamping, web-search tool translation,
reasoning-item dropping, and the OpenAI model-alias resolution.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from alpha_lab.providers.grok import (
    GROK_EFFORTS,
    GrokProvider,
    resolve_grok_effort,
)
from alpha_lab.providers.openai import resolve_openai_model


class TestResolveGrokEffort:
    def test_valid_tiers_pass_through(self):
        for tier in GROK_EFFORTS:  # minimal/low/medium/high/xhigh
            assert resolve_grok_effort(tier) == tier

    def test_none_empty_clamp_to_minimal(self):
        assert resolve_grok_effort("none") == "minimal"
        assert resolve_grok_effort("") == "minimal"
        assert resolve_grok_effort(None) == "minimal"

    def test_max_clamps_to_xhigh(self):
        assert resolve_grok_effort("max") == "xhigh"

    def test_case_insensitive(self):
        assert resolve_grok_effort("HIGH") == "high"
        assert resolve_grok_effort(" Medium ") == "medium"

    def test_unknown_defaults_to_medium(self):
        assert resolve_grok_effort("bogus") == "medium"


class TestResolveOpenAIModel:
    def test_alias_resolves_to_sol(self):
        assert resolve_openai_model("gpt-5.6") == "gpt-5.6-sol"

    def test_all_tiers_and_legacy_pass_through(self):
        for m in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.2", "gpt-5.5"]:
            assert resolve_openai_model(m) == m


class TestTranslateTools:
    def test_web_search_becomes_function_tool(self):
        out = GrokProvider._translate_tools([{"type": "web_search"}])
        assert len(out) == 1
        tool = out[0]
        assert tool["type"] == "function"
        assert tool["name"] == "web_search"
        assert "query" in tool["parameters"]["properties"]

    def test_web_search_tool_is_fresh_copy_each_call(self):
        # _translate_tools builds a fresh web_search schema per call (via the
        # _build_web_search_fn_schema factory, not a shared dict), so mutating
        # one returned tool can't affect any other.
        a = GrokProvider._translate_tools([{"type": "web_search"}])[0]
        b = GrokProvider._translate_tools([{"type": "web_search"}])[0]
        assert a == b and a is not b

    def test_other_tools_unchanged(self):
        fn = {"type": "function", "name": "add", "parameters": {"type": "object"}}
        assert GrokProvider._translate_tools([fn]) == [fn]

    def test_mixed_tools(self):
        fn = {"type": "function", "name": "add"}
        out = GrokProvider._translate_tools([{"type": "web_search"}, fn])
        assert out[0]["name"] == "web_search"
        assert out[1] == fn


class TestCleanOutputItem:
    def test_reasoning_item_is_dropped(self):
        # The compaction-blob quirk: reasoning items must NOT round-trip back.
        item = {"type": "reasoning", "id": "r1", "status": "completed",
                "encrypted_content": "blob"}
        assert GrokProvider._clean_output_item(item) is None

    def test_function_call_preserved(self):
        item = {"type": "function_call", "call_id": "c1", "name": "add",
                "arguments": "{}", "status": "completed", "id": "x"}
        out = GrokProvider._clean_output_item(item)
        assert out is not None
        assert out["type"] == "function_call"
        assert out["call_id"] == "c1" and out["name"] == "add"

    def test_message_preserved(self):
        item = {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
                "id": "m1", "status": "completed"}
        out = GrokProvider._clean_output_item(item)
        assert out is not None
        assert out["type"] == "message" and out["role"] == "assistant"


class TestComplete:
    """grok.complete() must route through the Responses API, not Chat Completions."""

    def test_complete_uses_responses_not_chat(self):
        captured: dict = {}

        class _Resp:
            usage = None
            output_text = "the summary"
            output: list = []

        class _Responses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return _Resp()

        class _Chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError("grok.complete must not use chat.completions")

        class _FakeClient:
            responses = _Responses()
            chat = _Chat()

        gp = GrokProvider(grok_client=_FakeClient(), openai_client_for_proxy=MagicMock())
        out = gp.complete(
            model="grok-4.5",
            system="sys",
            messages=[
                {"type": "reasoning", "id": "r"},           # must be stripped
                {"role": "user", "content": "long text"},
            ],
            max_tokens=123,
        )

        assert out == "the summary"
        assert captured["model"] == "grok-4.5"
        assert captured["instructions"] == "sys"
        assert captured["max_output_tokens"] == 123
        assert captured["reasoning"] == {"effort": "minimal"}
        # reasoning items are dropped from the Responses input (compaction-blob quirk)
        assert all(
            not (isinstance(i, dict) and i.get("type") == "reasoning")
            for i in captured["input"]
        )

    def test_complete_records_token_metrics_as_xai(self, monkeypatch):
        records: list[dict] = []

        class _Resp:
            usage = SimpleNamespace(input_tokens=11, output_tokens=7)
            output_text = "the summary"
            output: list = []

        class _Responses:
            def create(self, **kwargs):
                return _Resp()

        class _FakeClient:
            responses = _Responses()

        monkeypatch.setattr(
            "alpha_lab.providers.grok.token_metrics.record",
            lambda *args, **kwargs: records.append({"args": args, "kwargs": kwargs}),
        )

        gp = GrokProvider(grok_client=_FakeClient(), openai_client_for_proxy=MagicMock())

        assert gp.complete(
            model="grok-4.5",
            system="sys",
            messages=[{"role": "user", "content": "long text"}],
            max_tokens=123,
        ) == "the summary"
        assert records == [
            {
                "args": (11, 7),
                "kwargs": {"model": "grok-4.5", "system": "xai"},
            }
        ]


class TestStreamResponse:
    def test_stream_response_records_token_metrics_as_xai(self, monkeypatch):
        records: list[dict] = []

        class _Stream:
            def __iter__(self):
                raw_response = SimpleNamespace(
                    id="resp_1",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text="done")],
                            model_dump=lambda: {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "done"}],
                            },
                        )
                    ],
                    usage=SimpleNamespace(
                        input_tokens=13,
                        output_tokens=5,
                        input_tokens_details=SimpleNamespace(cached_tokens=3),
                    ),
                )
                yield SimpleNamespace(type="response.completed", response=raw_response)

            def close(self):
                pass

        class _Responses:
            def create(self, **kwargs):
                return _Stream()

        class _FakeClient:
            responses = _Responses()

        monkeypatch.setattr(
            "alpha_lab.providers.grok.token_metrics.record",
            lambda *args, **kwargs: records.append({"args": args, "kwargs": kwargs}),
        )

        gp = GrokProvider(grok_client=_FakeClient(), openai_client_for_proxy=MagicMock())
        events = list(
            gp.stream_response(
                model="grok-4.5",
                system="sys",
                history=[],
                tools=[],
                reasoning_effort="high",
            )
        )

        assert events[-1].type == "done"
        assert events[-1].response is not None
        assert events[-1].response.text == "done"
        assert records == [
            {
                "args": (13, 5),
                "kwargs": {
                    "cache_read": 3,
                    "model": "grok-4.5",
                    "system": "xai",
                },
            }
        ]
