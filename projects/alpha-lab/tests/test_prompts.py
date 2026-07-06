"""Tests for the prompt system."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.config import TaskConfig
from alpha_lab.knowledge import TopicKnowledgeStore
from alpha_lab.memory import MemoryStore
from alpha_lab.prompts import (
    PROMPT_REGISTRY,
    build_step_prompt,
    build_system_prompt,
)


class TestBuildSystemPrompt:
    def test_basic_prompt(self) -> None:
        prompt = build_system_prompt(workspace="/ws", learnings=None)
        assert "Alpha Lab" in prompt
        assert "/ws" in prompt

    def test_includes_learnings(self) -> None:
        prompt = build_system_prompt(
            workspace="/ws",
            learnings="# Key findings\n- Data has gaps",
        )
        assert "Key findings" in prompt
        assert "Data has gaps" in prompt

    def test_includes_config(self) -> None:
        config = TaskConfig(
            data_path="/data/prices.csv",
            description="Analyze crypto prices",
            target="close",
        )
        prompt = build_system_prompt(workspace="/ws", learnings=None, config=config)
        assert "prices.csv" in prompt
        assert "Analyze crypto prices" in prompt
        assert "close" in prompt

    def test_memory_guidance_requires_consent_for_user_provided_reference_notes(self) -> None:
        prompt = build_system_prompt(workspace="/ws", learnings=None)
        assert "ask for consent before storing it as `reference`" in prompt
        assert "never store secrets or credentials" in prompt

    def test_no_workspace(self) -> None:
        prompt = build_system_prompt(workspace=None, learnings=None)
        assert "Alpha Lab" in prompt  # Base prompt still present


class TestBuildStepPrompt:
    def test_valid_key(self) -> None:
        prompt = build_step_prompt("phase2_builder", "/ws", None)
        assert "Builder" in prompt

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt key"):
            build_step_prompt("nonexistent_key", "/ws", None)

    def test_includes_extra_context(self) -> None:
        prompt = build_step_prompt(
            "phase2_critic",
            "/ws",
            None,
            extra_context="## Experiment Details\nTest LSTM model",
        )
        assert "Experiment Details" in prompt
        assert "Test LSTM model" in prompt

    def test_includes_workspace(self) -> None:
        prompt = build_step_prompt("phase2_tester", "/my/workspace", None)
        assert "/my/workspace" in prompt

    def test_includes_learnings(self) -> None:
        prompt = build_step_prompt(
            "phase3_strategist",
            "/ws",
            "Data has 5000 rows of BTC-USD",
        )
        assert "5000 rows of BTC-USD" in prompt

    def test_includes_config(self) -> None:
        config = TaskConfig(data_path="/data/test.csv", description="Test task", target="returns")
        prompt = build_step_prompt("phase3_worker_implement", "/ws", None, config=config)
        assert "test.csv" in prompt
        assert "Test task" in prompt

    def test_includes_relevant_prior_memories(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        store = MemoryStore(str(ws))
        store.store(
            content="Fold construction leaked future information into validation.",
            tags=["phase3", "validation"],
            summary="Validation leakage",
            kind="failure",
            phase="phase3",
            agent="worker",
        )
        store.store(
            content="Unrelated CPU note.",
            tags=["infra"],
            summary="Infra note",
            kind="decision",
            phase="phase2",
        )

        config = TaskConfig(
            data_path="/data/test.csv",
            description="Investigate validation leakage in experiments",
            target="returns",
        )
        prompt = build_step_prompt("phase3_strategist", str(ws), None, config=config)
        assert "Relevant Prior Memories" in prompt
        assert "Validation leakage" in prompt
        assert "failure/phase3/worker" in prompt

    def test_prior_memory_recall_prioritizes_reference_topics(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        TopicKnowledgeStore(str(ws)).save_topic(
            "data_access.exchange_rates",
            "Use the managed exchange rate export and request the FX entitlement first.",
            title="Exchange rate data access",
            tags=["data_access"],
        )
        MemoryStore(str(ws)).store(
            content="Exchange rate validation should use walk-forward splits.",
            tags=["phase3"],
            summary="Exchange validation split",
            kind="failure",
            phase="phase3",
        )

        config = TaskConfig(
            data_path="/data/exchange_rates.csv",
            description="Analyze exchange rate data access and prediction quality",
            target="returns",
        )
        prompt = build_step_prompt("phase3_strategist", str(ws), None, config=config)

        assert "Relevant Prior Memories" in prompt
        assert "Exchange rate data access" in prompt
        assert "reference/topic_knowledge" in prompt


class TestPromptRegistry:
    def test_all_keys_present(self) -> None:
        expected_keys = [
            "phase1",
            "phase2_builder",
            "phase2_critic",
            "phase2_tester",
            "phase3_strategist",
            "phase3_worker_implement",
            "phase3_worker_analyze",
            "phase3_reporter",
        ]
        for key in expected_keys:
            assert key in PROMPT_REGISTRY, f"Missing prompt key: {key}"

    def test_all_prompts_nonempty(self) -> None:
        for key, prompt in PROMPT_REGISTRY.items():
            assert len(prompt) > 100, f"Prompt '{key}' seems too short: {len(prompt)} chars"
