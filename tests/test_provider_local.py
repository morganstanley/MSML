"""Unit tests for LocalProvider runtime tag→model selection + behavior table.

No network / real client construction: `models_by_tags` and `random.choice` are
monkeypatched, so we can assert which model gets selected per request and that the
request is shaped by that model's `LocalModelBehavior` (table entry or sniff
fallback).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_lab.providers import local as local_mod
from alpha_lab.providers.local import LocalProvider, _proxy_root
from alpha_lab.providers.local_models import (
    LocalModelBehavior,
    behavior_for,
    load_behaviors,
    sniff_dialect,
)


@pytest.fixture
def stub_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub client construction so from_config does no network / auth."""
    monkeypatch.setenv("LOCAL_BASE_URL", "http://proxy:4001")
    monkeypatch.delenv("GLM_NATIVE_TOOLS", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_CONFIG", raising=False)
    monkeypatch.setattr(local_mod, "get_local_client", lambda base_url: MagicMock())
    monkeypatch.setattr(local_mod, "get_openai_client", lambda api_key=None: MagicMock())


class TestProxyRoot:
    def test_strips_trailing_v1(self) -> None:
        assert _proxy_root("http://proxy:4001/v1") == "http://proxy:4001"
        assert _proxy_root("http://proxy:4001/v1/") == "http://proxy:4001"

    def test_leaves_bare_root(self) -> None:
        assert _proxy_root("http://proxy:4001/") == "http://proxy:4001"


class TestSelectModel:
    def test_tag_pool_random_pick_is_cached_and_queried_once(
        self, stub_clients: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}
        calls = {"n": 0}

        def fake_models_by_tags(baseurl, spec, timeout=10):
            calls["n"] += 1
            seen["baseurl"] = baseurl
            seen["spec"] = spec
            return ["glm-5.2-local1", "glm-5.2-local2"]

        monkeypatch.setattr(local_mod, "models_by_tags", fake_models_by_tags)
        monkeypatch.setattr(local_mod.random, "choice", lambda pool: pool[-1])

        provider = LocalProvider.from_config(model_tags=["local1"])
        assert provider._select_model() == "glm-5.2-local2"
        assert provider._select_model() == "glm-5.2-local2"

        # Pool queried once (cached); proxy root stripped of /v1; tag spec passed through.
        assert calls["n"] == 1
        assert seen["baseurl"] == "http://proxy:4001"
        assert seen["spec"] == ["local1"]

    def test_empty_pool_raises(
        self, stub_clients: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(local_mod, "models_by_tags", lambda *a, **k: [])
        provider = LocalProvider.from_config(model_tags=["nope"])
        with pytest.raises(ValueError, match="no LiteLLM-proxy model matches"):
            provider._select_model()

    def test_plain_model_path_selects_that_model(self, stub_clients: None) -> None:
        provider = LocalProvider.from_config(model="glm-5.2")
        assert provider._select_model() == "glm-5.2"


class TestBehaviorDrivesRequest:
    """The kwargs sent to chat.completions reflect the selected model's behavior."""

    def _driven_kwargs(self, provider: LocalProvider) -> dict:
        rec: list[dict] = []
        provider._client.chat.completions.create = lambda **kw: rec.append(kw) or _EmptyStream()
        list(provider.stream_response(
            model="ignored", system="SYS", history=[],
            tools=[{"type": "function", "name": "f", "description": "", "parameters": {}}],
            reasoning_effort="low",
        ))
        return rec[0]

    def test_glm_selection_uses_text_tool_and_caps(
        self, stub_clients: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(local_mod, "models_by_tags", lambda *a, **k: ["openai/glm-5.2"])
        monkeypatch.setattr(local_mod.random, "choice", lambda pool: pool[0])
        provider = LocalProvider.from_config(model_tags=["local1"])
        kwargs = self._driven_kwargs(provider)
        assert kwargs["model"] == "openai/glm-5.2"
        assert "tools" not in kwargs  # text-tool workaround (glm, non-native)
        assert kwargs["max_tokens"] == local_mod._GLM_MAX_OUTPUT_TOKENS

    def test_kimi_selection_uses_native_tools(
        self, stub_clients: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(local_mod, "models_by_tags", lambda *a, **k: ["moonshotai/Kimi-K2.6"])
        monkeypatch.setattr(local_mod.random, "choice", lambda pool: pool[0])
        provider = LocalProvider.from_config(model_tags=["fast"])
        kwargs = self._driven_kwargs(provider)
        assert kwargs["model"] == "moonshotai/Kimi-K2.6"
        assert kwargs["tool_choice"] == "auto"  # native tools path
        assert "max_tokens" not in kwargs

    def test_done_response_reports_the_selected_model(
        self, stub_clients: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Per-request accurate labeling: the done Response carries the model that
        # actually served the turn, so AgentLoop can set gen_ai.response.model.
        monkeypatch.setattr(local_mod, "models_by_tags", lambda *a, **k: ["glm-5.2-local1"])
        monkeypatch.setattr(local_mod.random, "choice", lambda pool: pool[0])
        provider = LocalProvider.from_config(model_tags=["local1"])
        provider._client.chat.completions.create = lambda **kw: _EmptyStream()
        events = list(provider.stream_response(
            model="ignored", system="SYS", history=[], tools=[], reasoning_effort="low",
        ))
        done = [e for e in events if e.type == "done"][0]
        assert done.response.model == "glm-5.2-local1"


class _EmptyStream:
    def __iter__(self):
        return iter(())

    def close(self) -> None:
        pass


class TestBehaviorTable:
    def test_load_behaviors_parses_and_defaults(self, tmp_path) -> None:
        path = tmp_path / "models.json"
        path.write_text(
            '{"openai/glm-5.2": {"glm_native_tools": true, "max_tokens": 999},'
            ' "moonshotai/Kimi-K2.6": {"thinking_style": "kimi"}}'
        )
        table = load_behaviors(str(path))
        glm = table["openai/glm-5.2"]
        assert glm.thinking_style == "glm"  # sniffed (omitted in JSON)
        assert glm.glm_native_tools is True
        assert glm.max_tokens == 999
        assert table["moonshotai/Kimi-K2.6"].thinking_style == "kimi"

    def test_load_behaviors_rejects_unknown_key(self, tmp_path) -> None:
        # A typo in a field name must fail loudly (naming the file + model), not
        # be silently dropped.
        path = tmp_path / "models.json"
        path.write_text('{"openai/glm-5.2": {"visoin": "bedrock"}}')
        with pytest.raises(ValueError, match="openai/glm-5.2"):
            load_behaviors(str(path))

    def test_load_behaviors_rejects_bad_enum_value(self, tmp_path) -> None:
        path = tmp_path / "models.json"
        path.write_text('{"openai/glm-5.2": {"thinking_style": "gpt"}}')
        with pytest.raises(ValueError, match="invalid settings"):
            load_behaviors(str(path))

    def test_load_behaviors_empty_path(self) -> None:
        assert load_behaviors(None) == {}
        assert load_behaviors("") == {}

    def test_behavior_for_table_hit(self) -> None:
        beh = LocalModelBehavior(thinking_style="kimi", max_tokens=5)
        assert behavior_for("whatever", {"whatever": beh}) is beh

    def test_behavior_for_sniff_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GLM_NATIVE_TOOLS", "1")
        beh = behavior_for("openai/glm-5.2", {})
        assert beh.thinking_style == "glm"
        assert beh.glm_native_tools is True
        assert beh.vision_provider == "bedrock"  # glm + native

    def test_behavior_for_unknown_dialect_raises(self) -> None:
        with pytest.raises(ValueError):
            behavior_for("gpt-4o", {})


class TestSniffDialect:
    @pytest.mark.parametrize("model", ["glm", "GLM-5.2", "openai/glm-5.2"])
    def test_glm(self, model: str) -> None:
        assert sniff_dialect(model) == "glm"

    @pytest.mark.parametrize("model", ["kimi", "moonshotai/Kimi-K2.6"])
    def test_kimi(self, model: str) -> None:
        assert sniff_dialect(model) == "kimi"

    @pytest.mark.parametrize("model", ["gpt-5.2", "", "auto"])
    def test_unknown_raises(self, model: str) -> None:
        with pytest.raises(ValueError):
            sniff_dialect(model)
