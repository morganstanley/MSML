"""Tests for LocalProvider's GLM text-tool parsing helpers.

GLM-5.1 runs in "text-tool" mode (native tool calls degenerate it), so it emits
tool calls as free text. These helpers parse them back into (name, args) tuples;
they are pure functions, so we exercise the format edge cases directly.
"""

from __future__ import annotations

import json
import types

import pytest

from alpha_lab.providers.local import (
    _GLM_MAX_OUTPUT_TOKENS,
    _GLM_TEMPERATURE,
    _GLM_TEXT_TOOL_INSTRUCTION,
    _WEB_SEARCH_FN_SCHEMA,
    LocalProvider,
    _brace_object_at,
    _parse_text_tool_calls,
    _translate_tools,
)
from alpha_lab.providers.local_models import sniff_dialect as _sniff_dialect


# --- _brace_object_at --------------------------------------------------------

def test_brace_object_simple() -> None:
    text = '{"a": 1}'
    obj, end = _brace_object_at(text, 0)
    assert obj == {"a": 1}
    assert end == len(text) - 1  # index of the matching '}'


def test_brace_object_nested() -> None:
    text = '{"a": {"b": {"c": 2}}}'
    obj, end = _brace_object_at(text, 0)
    assert obj == {"a": {"b": {"c": 2}}}
    assert text[end] == "}"
    assert end == len(text) - 1


def test_brace_object_ignores_braces_inside_strings() -> None:
    text = '{"a": "}{ not real braces {"}'
    obj, end = _brace_object_at(text, 0)
    assert obj == {"a": "}{ not real braces {"}
    assert text[end] == "}"


def test_brace_object_handles_escaped_quote() -> None:
    text = r'{"a": "say \"hi\" }"}'
    obj, _ = _brace_object_at(text, 0)
    assert obj == {"a": 'say "hi" }'}


def test_brace_object_starts_at_offset() -> None:
    text = 'prefix noise {"k": "v"} trailing'
    start = text.index("{")
    obj, end = _brace_object_at(text, start)
    assert obj == {"k": "v"}
    assert text[end] == "}"


def test_brace_object_unbalanced_returns_none_and_last_index() -> None:
    text = '{"a": 1'
    obj, end = _brace_object_at(text, 0)
    assert obj is None
    assert end == len(text) - 1


def test_brace_object_balanced_but_invalid_json() -> None:
    text = "{not valid json}"
    obj, end = _brace_object_at(text, 0)
    assert obj is None
    assert text[end] == "}"


# --- _parse_text_tool_calls: empty / none -----------------------------------

def test_parse_empty_text() -> None:
    assert _parse_text_tool_calls("") == []


def test_parse_no_tool_calls() -> None:
    assert _parse_text_tool_calls("just some prose, no tools here") == []


# --- _parse_text_tool_calls: native <tool_call> form ------------------------

def test_parse_native_tool_call() -> None:
    calls = _parse_text_tool_calls('<tool_call>read_file {"path": "x"}')
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "read_file"
    assert json.loads(args) == {"path": "x"}


def test_parse_native_tag_wrapping_name_arguments_object() -> None:
    # When the object after the tag is itself a {"name","arguments"} call, its
    # inner name/args win over the tag name.
    text = '<tool_call>ignored {"name": "shell_exec", "arguments": {"cmd": "ls"}}'
    calls = _parse_text_tool_calls(text)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "shell_exec"
    assert json.loads(args) == {"cmd": "ls"}


def test_parse_native_tag_whitespace_before_name() -> None:
    calls = _parse_text_tool_calls('<tool_call>\n  read_file {"path": "y"}')
    assert calls == [("read_file", json.dumps({"path": "y"}))]


def test_parse_multiple_native_calls() -> None:
    text = (
        '<tool_call>read_file {"path": "a"}\n'
        '<tool_call>write_file {"path": "b", "content": "hi"}'
    )
    calls = _parse_text_tool_calls(text)
    assert [c[0] for c in calls] == ["read_file", "write_file"]
    assert json.loads(calls[1][1]) == {"path": "b", "content": "hi"}


def test_parse_native_args_with_nested_and_brace_strings() -> None:
    text = '<tool_call>patch {"body": "func() {{ return 1; }}", "meta": {"n": 3}}'
    calls = _parse_text_tool_calls(text)
    assert len(calls) == 1
    assert json.loads(calls[0][1]) == {"body": "func() {{ return 1; }}", "meta": {"n": 3}}


# --- _parse_text_tool_calls: bare JSON form ---------------------------------

def test_parse_bare_json_object() -> None:
    text = 'sure: {"name": "shell_exec", "arguments": {"cmd": "ls -la"}}'
    calls = _parse_text_tool_calls(text)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "shell_exec"
    assert json.loads(args) == {"cmd": "ls -la"}


def test_parse_bare_json_arguments_as_string() -> None:
    # arguments already a JSON string is passed through unchanged (not re-encoded).
    text = '{"name": "f", "arguments": "{\\"k\\": 1}"}'
    calls = _parse_text_tool_calls(text)
    assert calls == [("f", '{"k": 1}')]


def test_parse_bare_json_arguments_as_dict_normalized_to_string() -> None:
    calls = _parse_text_tool_calls('{"name": "f", "arguments": {"k": 1}}')
    assert calls == [("f", json.dumps({"k": 1}))]


def test_parse_multiple_bare_json_calls() -> None:
    text = (
        'first {"name": "a", "arguments": {}} '
        'then {"name": "b", "arguments": {"x": 2}}'
    )
    calls = _parse_text_tool_calls(text)
    assert [c[0] for c in calls] == ["a", "b"]
    assert json.loads(calls[1][1]) == {"x": 2}


def test_parse_bare_json_skips_non_call_objects() -> None:
    text = '{"unrelated": true} {"name": "go", "arguments": {"n": 1}}'
    calls = _parse_text_tool_calls(text)
    assert calls == [("go", json.dumps({"n": 1}))]


# --- precedence --------------------------------------------------------------

def test_native_takes_precedence_over_bare_json() -> None:
    # Text has both a native tag and a stray bare call object. Once the native
    # pass finds a call, the bare-JSON pass never runs — so "other" is ignored.
    text = (
        '<tool_call>read_file {"path": "real"}\n'
        'leftover {"name": "other", "arguments": {"x": 9}}'
    )
    calls = _parse_text_tool_calls(text)
    assert calls == [("read_file", json.dumps({"path": "real"}))]


# --- _sniff_dialect ----------------------------------------------------------

@pytest.mark.parametrize(
    "model",
    ["glm", "GLM", "GLM-5.2", "zai-org/GLM-5.1", "glm-5.2"],
)
def test_sniff_dialect_glm(model: str) -> None:
    assert _sniff_dialect(model) == "glm"


@pytest.mark.parametrize(
    "model",
    ["kimi", "Kimi", "Kimi-K2.6", "moonshotai/Kimi-K2.6"],
)
def test_sniff_dialect_kimi(model: str) -> None:
    assert _sniff_dialect(model) == "kimi"


@pytest.mark.parametrize("model", ["gpt-5.2", "claude-opus-4-8", "", "auto"])
def test_sniff_dialect_unknown_raises(model: str) -> None:
    with pytest.raises(ValueError):
        _sniff_dialect(model)


# --- _translate_tools --------------------------------------------------------

def test_translate_tools_function_schema() -> None:
    tools = [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    out = _translate_tools(tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


def test_translate_tools_web_search_becomes_proxy_schema() -> None:
    out = _translate_tools([{"type": "web_search"}])
    assert out == [_WEB_SEARCH_FN_SCHEMA]


def test_translate_tools_skips_non_dict_and_unknown_types() -> None:
    tools = [
        "not a dict",
        {"type": "code_interpreter"},  # unknown -> skipped
        {"type": "function", "name": "f", "description": "", "parameters": {}},
    ]
    out = _translate_tools(tools)
    assert [t["function"]["name"] for t in out] == ["f"]


def test_translate_tools_empty() -> None:
    assert _translate_tools([]) == []


# --- LocalProvider.stream_response routing ------------------------------------
#
# Drive the generator against a fake client that records the kwargs passed to
# ``chat.completions.create`` and returns an empty stream, so we can assert how
# each turn is routed (native tools vs GLM text-tool mode vs vision proxy)
# without hitting a real endpoint.

class _FakeStream:
    """Iterable, closeable stand-in for an OpenAI streaming response (no chunks)."""

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        pass


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream()


def _fake_client() -> types.SimpleNamespace:
    completions = _RecordingCompletions()
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions),
        _completions=completions,
    )


_FN_TOOL = {
    "type": "function",
    "name": "do_thing",
    "description": "Do a thing.",
    "parameters": {"type": "object", "properties": {}},
}


def _drive(provider: LocalProvider, **kwargs) -> None:
    # Exhaust the generator so ``create`` is actually invoked.
    list(provider.stream_response(**kwargs))


def test_stream_response_kimi_uses_native_tools() -> None:
    client = _fake_client()
    proxy = _fake_client()
    # Behavior is selected per request from the (plain) model — kimi here.
    provider = LocalProvider(
        client=client, openai_client_for_proxy=proxy, model="moonshotai/Kimi-K2.6"
    )

    _drive(
        provider,
        model="moonshotai/Kimi-K2.6",
        system="SYS",
        history=[],
        tools=[_FN_TOOL],
        reasoning_effort="low",
    )

    assert proxy._completions.calls == []  # no vision -> proxy untouched
    kwargs = client._completions.calls[0]
    assert kwargs["model"] == "moonshotai/Kimi-K2.6"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {"name": "do_thing", "description": "Do a thing.", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    # System message is sent verbatim (no text-tool preamble); GLM-only knobs absent.
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_stream_response_glm_uses_text_tool_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client()
    proxy = _fake_client()
    # GLM sniff-fallback reads GLM_NATIVE_TOOLS for the native-vs-text decision;
    # keep it unset so this exercises the text-tool workaround.
    monkeypatch.delenv("GLM_NATIVE_TOOLS", raising=False)
    provider = LocalProvider(
        client=client, openai_client_for_proxy=proxy, model="zai-org/GLM-5.1"
    )

    _drive(
        provider,
        model="zai-org/GLM-5.1",
        system="SYS",
        history=[],
        tools=[_FN_TOOL],
        reasoning_effort="low",
    )

    kwargs = client._completions.calls[0]
    # Text-tool mode: no native tools array; tools described in the system prompt.
    assert "tools" not in kwargs
    assert _GLM_TEXT_TOOL_INSTRUCTION in kwargs["messages"][0]["content"]
    assert "do_thing" in kwargs["messages"][0]["content"]
    # Thinking forced off for parseable tool calls; GLM output knobs applied.
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert kwargs["temperature"] == _GLM_TEMPERATURE
    assert kwargs["max_tokens"] == _GLM_MAX_OUTPUT_TOKENS


def test_stream_response_routes_vision_turn_to_gpt4o_proxy() -> None:
    client = _fake_client()
    proxy = _fake_client()
    # kimi model -> no opus fallback; a recent image turn routes to the gpt-4o proxy.
    provider = LocalProvider(
        client=client, openai_client_for_proxy=proxy, model="moonshotai/Kimi-K2.6"
    )

    image_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }

    _drive(
        provider,
        model="moonshotai/Kimi-K2.6",
        system="SYS",
        history=[image_msg],
        tools=[],
        reasoning_effort="low",
    )

    assert client._completions.calls == []  # main endpoint bypassed
    kwargs = proxy._completions.calls[0]
    assert kwargs["model"] == "gpt-4o"
    # Full history forwarded unstripped (proxy can handle image_url blocks).
    assert kwargs["messages"][-1]["content"][1]["type"] == "image_url"
