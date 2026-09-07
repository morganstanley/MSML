"""Unit tests for native Anthropic provider helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from alpha_lab.providers.anthropic import (
    CACHE_CONTROL,
    PROXIED_WEB_SEARCH_NAME,
    AnthropicProvider,
    _mark_history_cache_point,
    _strip_orphan_tool_results,
)
from alpha_lab.providers.types import Response, ToolCall


def _make_provider() -> AnthropicProvider:
    return AnthropicProvider(client=MagicMock(), openai_client=MagicMock())


def test_translate_tools_rewrites_web_search() -> None:
    provider = _make_provider()

    translated = provider._translate_tools([{"type": "web_search"}])

    # The gateway rejects a Messages-API tool literally named "web_search"
    # (reserved for Anthropic's hosted tool), so the proxy travels renamed.
    assert translated == [
        {
            "name": "search_web",
            "description": "Search the web for current information. Returns search results as text.",
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
        }
    ]


def test_translate_tools_converts_function_schema() -> None:
    provider = _make_provider()
    tool = {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }

    translated = provider._translate_tools([tool])

    assert translated == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": tool["parameters"],
        }
    ]


def test_translate_tools_returns_none_for_empty_input() -> None:
    assert _make_provider()._translate_tools([]) is None


def test_append_response_to_history_preserves_raw_output_items() -> None:
    provider = _make_provider()
    history: list[dict] = []
    raw_items = [
        {"type": "text", "text": "I will call a tool."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "x"}},
    ]
    response = Response(
        id="msg_1",
        text="I will call a tool.",
        tool_calls=[ToolCall(call_id="toolu_1", name="read_file", arguments='{"path":"x"}')],
        has_web_search=False,
        input_tokens=10,
        output_tokens=5,
        raw_output_items=raw_items,
    )

    provider.append_response_to_history(history, response)

    assert history == [{"role": "assistant", "content": raw_items}]


def test_append_response_to_history_ignores_empty_raw_output_items() -> None:
    provider = _make_provider()
    history: list[dict] = []
    response = Response(
        id="msg_1",
        text="hello",
        tool_calls=[],
        has_web_search=False,
        input_tokens=1,
        output_tokens=1,
    )

    provider.append_response_to_history(history, response)

    assert history == []


def test_complete_routes_claude_ids_to_anthropic(monkeypatch) -> None:
    provider = _make_provider()
    captured: list[dict] = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[
                SimpleNamespace(type="text", text="hello "),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="world"),
            ],
        )

    provider._client.messages.create.side_effect = fake_create
    provider._openai_client.chat.completions.create.side_effect = AssertionError(
        "Claude ids must not route through OpenAI"
    )
    monkeypatch.setattr("alpha_lab.providers.anthropic.token_metrics.record", lambda *args, **kwargs: None)

    for model in (
        "claude-opus-4-8",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8-v1",
    ):
        assert provider.complete(
            model=model,
            system="sys",
            messages=[{"role": "user", "content": "summarize"}],
            max_tokens=123,
        ) == "hello world"

    assert [call["model"] for call in captured] == [
        "claude-opus-4-8",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8-v1",
    ]
    assert all(call["system"] == "sys" for call in captured)
    assert all(call["max_tokens"] == 123 for call in captured)


def test_complete_routes_non_anthropic_us_prefixed_ids_to_openai(monkeypatch) -> None:
    provider = _make_provider()

    class _Choice:
        message = SimpleNamespace(content="openai summary")

    provider._openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[_Choice()]
    )
    provider._client.messages.create.side_effect = AssertionError(
        "Non-Anthropic us.* ids must not route through Anthropic"
    )
    monkeypatch.setattr(
        "alpha_lab.providers.anthropic.token_metrics.record_chat_usage",
        lambda *args, **kwargs: None,
    )

    assert provider.complete(
        model="us.openai.gpt-5.4",
        system="sys",
        messages=[{"role": "user", "content": "summarize"}],
        max_tokens=456,
    ) == "openai summary"

    provider._openai_client.chat.completions.create.assert_called_once_with(
        model="us.openai.gpt-5.4",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summarize"},
        ],
        max_tokens=456,
    )


def test_mark_history_cache_point_returns_empty_unchanged() -> None:
    assert _mark_history_cache_point([]) == []


def test_mark_history_cache_point_empty_content_returns_unchanged() -> None:
    messages = [{"role": "user", "content": []}]

    assert _mark_history_cache_point(messages) == messages


def test_mark_history_cache_point_marks_last_cacheable_block() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    marked = _mark_history_cache_point(messages)

    assert marked == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi", "cache_control": CACHE_CONTROL}],
        }
    ]


def test_mark_history_cache_point_skips_trailing_thinking_block() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}},
                {"type": "thinking", "thinking": "...", "signature": "sig"},
            ],
        },
    ]

    marked = _mark_history_cache_point(messages)

    content = marked[-1]["content"]
    assert content[1] == {"type": "thinking", "thinking": "...", "signature": "sig"}
    assert content[0]["cache_control"] == CACHE_CONTROL


def test_mark_history_cache_point_no_cacheable_block_returns_unchanged() -> None:
    messages = [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "...", "signature": "sig"}]},
    ]

    marked = _mark_history_cache_point(messages)

    assert marked == messages
    assert "cache_control" not in marked[-1]["content"][0]


def test_mark_history_cache_point_does_not_mutate_input() -> None:
    original_block = {"type": "text", "text": "hi"}
    messages = [{"role": "user", "content": [original_block]}]

    _mark_history_cache_point(messages)

    assert original_block == {"type": "text", "text": "hi"}
    assert messages[0]["content"][0] is original_block


def test_mark_history_cache_point_only_touches_last_message() -> None:
    first = {"role": "user", "content": [{"type": "text", "text": "first"}]}
    messages = [first, {"role": "user", "content": [{"type": "text", "text": "second"}]}]

    marked = _mark_history_cache_point(messages)

    assert marked[0] is first
    assert marked[1]["content"][0]["cache_control"] == CACHE_CONTROL


class _FakeStream:
    """Context-manager stub standing in for client.messages.stream(...)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return SimpleNamespace(
            id="msg_1",
            content=[],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def _stream_kwargs(provider, monkeypatch) -> dict:
    """Drive stream_response once and capture the request kwargs."""
    captured: dict = {}

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    provider._client.messages.stream = fake_stream
    monkeypatch.setattr(
        "alpha_lab.providers.anthropic.token_metrics.record",
        lambda *args, **kwargs: None,
    )
    list(provider.stream_response(
        model="anthropic.claude-opus-4-8",
        system="sys",
        history=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[{"type": "function", "name": "t", "description": "d",
                "parameters": {"type": "object", "properties": {}}}],
        reasoning_effort="none",
    ))
    return captured


def test_prompt_caching_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_PROMPT_CACHING", raising=False)
    kwargs = _stream_kwargs(_make_provider(), monkeypatch)

    assert kwargs["system"][0]["cache_control"] == CACHE_CONTROL
    assert kwargs["tools"][-1]["cache_control"] == CACHE_CONTROL
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] == CACHE_CONTROL


def test_prompt_caching_disabled_sends_no_markers(monkeypatch) -> None:
    # Zero-retention deployments: ANTHROPIC_PROMPT_CACHING=0 must remove every
    # cache marker so nothing asks the service to retain the prompt prefix.
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHING", "0")
    kwargs = _stream_kwargs(_make_provider(), monkeypatch)

    assert kwargs["system"] == "sys"
    assert "cache_control" not in json.dumps(kwargs["messages"])
    assert "cache_control" not in json.dumps(kwargs["tools"])


def test_effort_travels_via_extra_body_not_typed_kwarg(monkeypatch) -> None:
    # anthropic 0.60.0's Messages.stream() has no output_config parameter and
    # rejects it with TypeError; extra_body puts it on the wire regardless of
    # the installed SDK's signature.
    captured: dict = {}
    provider = _make_provider()

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    provider._client.messages.stream = fake_stream
    monkeypatch.setattr(
        "alpha_lab.providers.anthropic.token_metrics.record",
        lambda *args, **kwargs: None,
    )
    list(provider.stream_response(
        model="anthropic.claude-opus-4-8", system="sys",
        history=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[], reasoning_effort="high",
    ))

    assert "output_config" not in captured
    assert captured["extra_body"] == {"output_config": {"effort": "high"}}
    assert captured["thinking"] == {"type": "adaptive"}


def test_orphan_tool_results_are_stripped_before_send(monkeypatch) -> None:
    # History trimming can drop the assistant turn that requested a tool while
    # keeping its result; the API then rejects the whole request. The orphan
    # is dropped; a paired result directly after its tool_use turn survives.
    captured: dict = {}
    provider = _make_provider()

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    provider._client.messages.stream = fake_stream
    monkeypatch.setattr(
        "alpha_lab.providers.anthropic.token_metrics.record",
        lambda *args, **kwargs: None,
    )
    history = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_gone", "content": "orphan"},
            {"type": "text", "text": "kept text"},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_live", "name": "t", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_live", "content": "paired"},
        ]},
    ]
    list(provider.stream_response(
        model="anthropic.claude-opus-4-8", system="sys",
        history=history, tools=[], reasoning_effort="none",
    ))

    sent = json.dumps(captured["messages"])
    assert "toolu_gone" not in sent
    assert "toolu_live" in sent
    assert "kept text" in sent


def test_strip_orphans_drops_message_left_empty() -> None:
    history = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_gone", "content": "x"},
    ]}]

    assert _strip_orphan_tool_results(history) == []


def test_proxied_search_tool_use_maps_back_to_web_search(monkeypatch) -> None:
    # On the wire the proxy is named search_web (the gateway rejects the
    # reserved name); the agent-facing ToolCall must still say web_search so
    # tool dispatch is unchanged.
    from types import SimpleNamespace as NS

    provider = _make_provider()

    class _ToolUseStream(_FakeStream):
        def get_final_message(self):
            return NS(
                id="msg_1",
                content=[NS(type="tool_use", id="toolu_9",
                            name=PROXIED_WEB_SEARCH_NAME, input={"query": "q"})],
                usage=NS(input_tokens=0, output_tokens=0),
            )

    provider._client.messages.stream = lambda **kwargs: _ToolUseStream()
    provider._block_to_param = lambda block: {"type": "tool_use", "id": block.id,
                                              "name": block.name, "input": block.input}
    monkeypatch.setattr(
        "alpha_lab.providers.anthropic.token_metrics.record",
        lambda *args, **kwargs: None,
    )
    events = list(provider.stream_response(
        model="anthropic.claude-opus-4-8", system="sys",
        history=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[], reasoning_effort="none",
    ))

    done = [e for e in events if e.type == "done"][0]
    assert [tc.name for tc in done.response.tool_calls] == ["web_search"]
