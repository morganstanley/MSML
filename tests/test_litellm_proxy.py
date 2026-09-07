"""Unit tests for ``models_by_tags`` (LiteLLM proxy admin API).

Covers the parts the LocalProvider tests mock out: parsing the ``/v1/model/info``
response (``data[].model_info.key`` + ``litellm_params.tags``) and the clause
matching — OR across clauses, AND within a nested-list clause. ``requests.get``
is monkeypatched, so there is no network.
"""
from __future__ import annotations

import pytest

from alpha_lab.providers import litellm_proxy
from alpha_lab.providers.litellm_proxy import models_by_tags

# Distinct keys so each assertion is unambiguous.
DATA = [
    {"model_info": {"key": "openai/glm-5.2"},   "litellm_params": {"tags": ["local1", "default"]}},
    {"model_info": {"key": "openai/glm-5.2-b"}, "litellm_params": {"tags": ["local2", "default", "gpu"]}},
    {"model_info": {"key": "moonshotai/kimi"},  "litellm_params": {"tags": ["cpu"]}},
    {"model_info": {"key": "untagged/model"},   "litellm_params": {}},  # no tags key
]


class _FakeResp:
    def __init__(self, data: list[dict], status_error: Exception | None = None) -> None:
        self._data = data
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return {"data": self._data}


@pytest.fixture()
def fake_get(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Patch requests.get to return DATA and record the call args."""
    calls: list[dict] = []

    def _get(url, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        return _FakeResp(DATA)

    monkeypatch.setattr(litellm_proxy.requests, "get", _get)
    return calls


def test_no_spec_returns_all_keys(fake_get) -> None:
    assert models_by_tags("http://proxy:4001") == [
        "openai/glm-5.2", "openai/glm-5.2-b", "moonshotai/kimi", "untagged/model",
    ]


def test_single_string_clause_matches_that_tag(fake_get) -> None:
    assert models_by_tags("http://proxy:4001", ["default"]) == [
        "openai/glm-5.2", "openai/glm-5.2-b",
    ]


def test_multiple_string_clauses_are_ored(fake_get) -> None:
    assert models_by_tags("http://proxy:4001", ["local1", "cpu"]) == [
        "openai/glm-5.2", "moonshotai/kimi",
    ]


def test_nested_list_clause_is_anded(fake_get) -> None:
    # Only a model carrying BOTH "default" and "gpu" matches.
    assert models_by_tags("http://proxy:4001", [["default", "gpu"]]) == ["openai/glm-5.2-b"]


def test_mixed_and_or_clauses(fake_get) -> None:
    # (default AND gpu) OR cpu
    assert models_by_tags("http://proxy:4001", [["default", "gpu"], "cpu"]) == [
        "openai/glm-5.2-b", "moonshotai/kimi",
    ]


def test_no_match_returns_empty_and_untagged_never_matches(fake_get) -> None:
    assert models_by_tags("http://proxy:4001", ["nope"]) == []
