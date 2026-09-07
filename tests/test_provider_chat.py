"""Unit tests for ChatProvider stream-degeneration detection.

Regression for the gemma-4 phrase-repetition runaway on the mlrllm gateway
(16.9 GB strategist record, 2026-08-05): `_is_degenerate_tail` must catch
short-phrase loops, not just GLM's single-char runs, and must not fire on
legitimate long output. Pure-function tests — no network, no client.
"""
from __future__ import annotations

from alpha_lab.provider_chat import _is_degenerate_tail


class TestDegenerateTail:
    def test_single_char_run(self) -> None:
        assert _is_degenerate_tail("prefix " + "!" * 80)

    def test_phrase_loop(self) -> None:
        text = "thinking... " + "**Wait, I'll call the tool.**\n\n" * 60
        assert _is_degenerate_tail(text)

    def test_legit_prose_negative(self) -> None:
        text = " ".join(f"sentence number {i} about payup modeling." for i in range(80))
        assert not _is_degenerate_tail(text)

    def test_short_text_negative(self) -> None:
        assert not _is_degenerate_tail("**Wait, I'll call the tool.**")

    def test_whitespace_tail_negative(self) -> None:
        assert not _is_degenerate_tail("x" * 30 + "\n" * 2000)


class TestFitProxyContext:
    """Regression for the vision-proxy 400 (151,084 tokens vs gpt-4o's 128k,
    2026-08-05): the proxy request must be bounded to the system message +
    the longest history tail that fits."""

    def test_under_budget_untouched(self) -> None:
        from alpha_lab.provider_chat import _fit_proxy_context
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"}]
        assert _fit_proxy_context(msgs, budget=10_000) is msgs

    def test_trims_oldest_keeps_system_and_tail(self) -> None:
        from alpha_lab.provider_chat import _fit_proxy_context
        msgs = ([{"role": "system", "content": "sys"}]
                + [{"role": "user", "content": f"old {i} " + "x" * 100} for i in range(50)]
                + [{"role": "user", "content": "IMAGE TURN"}])
        out = _fit_proxy_context(msgs, budget=600)
        assert out[0]["content"] == "sys"
        assert out[-1]["content"] == "IMAGE TURN"
        assert len(out) < len(msgs)

    def test_final_message_kept_even_if_huge(self) -> None:
        from alpha_lab.provider_chat import _fit_proxy_context
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "y" * 5000}]
        out = _fit_proxy_context(msgs, budget=100)
        assert out[-1]["content"] == "y" * 5000

    def test_orphan_tool_results_dropped(self) -> None:
        from alpha_lab.provider_chat import _fit_proxy_context
        msgs = ([{"role": "system", "content": "sys"}]
                + [{"role": "user", "content": "z" * 300}]
                + [{"role": "tool", "tool_call_id": "t1", "content": "r" * 50}]
                + [{"role": "user", "content": "IMAGE TURN"}])
        out = _fit_proxy_context(msgs, budget=250)
        assert all(m.get("role") != "tool" for m in out)
        assert out[-1]["content"] == "IMAGE TURN"


class TestMlrllmThinkingDialect:
    """Verified-live request shaping for the lab models (2026-08-05):
    kimi-k3 uses the official top-level reasoning_effort (model card §6);
    deepseek uses chat_template_kwargs.thinking; gemma sends nothing."""

    def _body(self, effort, model):
        from unittest.mock import MagicMock
        from alpha_lab.provider_chat import ChatProvider
        prov = ChatProvider(client=MagicMock(), openai_client_for_proxy=MagicMock(),
                            thinking_style="mlrllm")
        return prov._thinking_extra_body(effort, model)

    def test_kimi_k3_max(self):
        assert self._body("max", "kimi-k3") == {"extra_body": {"reasoning_effort": "max"}}

    def test_kimi_k3_high_maps_to_high(self):
        assert self._body("high", "kimi-k3") == {"extra_body": {"reasoning_effort": "high"}}

    def test_deepseek_thinking_no_cap(self):
        # drop_thinking rides along by default since the reasoning-replay fix.
        assert self._body("max", "deepseek-v4-flash") == {
            "extra_body": {"chat_template_kwargs": {"thinking": True,
                                                    "drop_thinking": False}}}

    def test_gemma_sends_nothing(self):
        assert self._body("high", "gemma-4-31b") == {}


class TestReplayReasoning:
    """Every chat model replays its reasoning into history by default
    (CHAT_REPLAY_REASONING=0 opts out); kimi-k3 replays unconditionally.
    Before 2026-08-07 only kimi-k3 did — GLM and deepseek ran blind to
    their own past reasoning."""

    @staticmethod
    def _prov(style, monkeypatch, env=None):
        from unittest.mock import MagicMock
        from alpha_lab.provider_chat import ChatProvider
        if env is None:
            monkeypatch.delenv("CHAT_REPLAY_REASONING", raising=False)
        else:
            monkeypatch.setenv("CHAT_REPLAY_REASONING", env)
        return ChatProvider(client=MagicMock(), openai_client_for_proxy=MagicMock(),
                            thinking_style=style)

    def test_default_on_for_glm(self, monkeypatch):
        assert self._prov("glm", monkeypatch)._should_replay_reasoning("glm-5.2")

    def test_default_on_for_deepseek(self, monkeypatch):
        assert self._prov("mlrllm", monkeypatch)._should_replay_reasoning(
            "deepseek-v4-flash")

    def test_opt_out_disables_glm_and_deepseek(self, monkeypatch):
        assert not self._prov("glm", monkeypatch, "0")._should_replay_reasoning(
            "glm-5.2")
        assert not self._prov("mlrllm", monkeypatch, "0")._should_replay_reasoning(
            "deepseek-v4-flash")

    def test_kimi_k3_unconditional(self, monkeypatch):
        assert self._prov("mlrllm", monkeypatch, "0")._should_replay_reasoning(
            "kimi-k3")

    def test_glm_requests_clear_thinking_false(self, monkeypatch):
        # GLM's template drops replayed history reasoning unless the request
        # carries clear_thinking=false (live-verified 73 vs 47 prompt tokens).
        body = self._prov("glm", monkeypatch)._thinking_extra_body("high", "glm-5.2")
        assert body["extra_body"]["chat_template_kwargs"]["clear_thinking"] is False

    def test_glm_opt_out_sends_no_clear_thinking(self, monkeypatch):
        body = self._prov("glm", monkeypatch, "0")._thinking_extra_body(
            "high", "glm-5.2")
        assert body == {}

    def test_deepseek_requests_drop_thinking_false(self, monkeypatch):
        # DeepSeek's encoder drops history reasoning unless drop_thinking is
        # false (live-verified 66 vs 39 prompt tokens on iml39:8001).
        body = self._prov("mlrllm", monkeypatch)._thinking_extra_body(
            "high", "deepseek-v4-flash")
        kw = body["extra_body"]["chat_template_kwargs"]
        assert kw == {"thinking": True, "drop_thinking": False}

    def test_deepseek_opt_out_keeps_old_shape(self, monkeypatch):
        body = self._prov("mlrllm", monkeypatch, "0")._thinking_extra_body(
            "high", "deepseek-v4-flash")
        assert body == {"extra_body": {"chat_template_kwargs": {"thinking": True}}}


class TestScreenToolCalls:
    """History-poisoning guard: a tool call whose bytes would 400 every
    subsequent request (unparseable JSON arguments, invalid name) must be
    dropped BEFORE it enters replayable history. Measured 2026-08-08:
    33 poisoned-argument + 12 poisoned-name calls across 51 runs, each
    drawing 'Unterminated string' / 'function.name does not match pattern'
    400s on replay."""

    def _tc(self, name, args):
        from alpha_lab.provider import ToolCall
        return ToolCall(call_id="c1", name=name, arguments=args)

    def test_valid_call_kept(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        kept, dropped = _screen_tool_calls(
            [self._tc("read_file", '{"path": "playbook.md"}')], "m")
        assert len(kept) == 1 and dropped == 0

    def test_truncated_arguments_dropped(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        # the observed shape: value string opens and never closes
        kept, dropped = _screen_tool_calls(
            [self._tc("write_debrief", '{"summary": "Implemented Exp')], "m")
        assert kept == [] and dropped == 1

    def test_fused_name_dropped(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        # deepseek fusing arguments into the name slot (observed verbatim)
        kept, dropped = _screen_tool_calls(
            [self._tc('read_file" path="backtest/strategy.py', "{}")], "m")
        assert kept == [] and dropped == 1

    def test_unicode_garbage_name_dropped(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        kept, dropped = _screen_tool_calls([self._tc("分析", "{}")], "m")
        assert kept == [] and dropped == 1

    def test_empty_arguments_treated_as_empty_object(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        kept, dropped = _screen_tool_calls([self._tc("read_board", "")], "m")
        assert len(kept) == 1 and dropped == 0

    def test_mixed_batch_drops_only_bad(self):
        from alpha_lab.provider_chat import _screen_tool_calls
        kept, dropped = _screen_tool_calls(
            [self._tc("read_file", '{"path": "a"}'),
             self._tc("shell_exec", '{"command": "ls')], "m")
        assert [t.name for t in kept] == ["read_file"] and dropped == 1
