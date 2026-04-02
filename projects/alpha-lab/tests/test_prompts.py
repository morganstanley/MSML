"""Tests for the prompt system."""

from __future__ import annotations

import pytest

from alpha_lab.config import TaskConfig
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
