"""Unit tests for alpha_lab.evaluations — pure-unit, no LLM calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from alpha_lab.adapter import DomainAdapter
from alpha_lab.evaluations import (
    DEFAULT_FILE_WEIGHTS,
    DEFAULT_SECTION_WEIGHTS,
    _build_similarity_judge,
    _extract_file_composite,
    _rubric_hash,
    build_criteria_judges,
    build_dataset,
    build_threshold_evaluators,
    load_eval_config,
    load_template_variables,
    parse_phase0_metrics,
    render_criteria,
    resolve_file_content,
    ThresholdEvaluator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "evaluations"
FIXTURE_YAML = FIXTURE_DIR / "adapter_evaluation.yaml"


@dataclass
class _FakeEvaluationResult:
    """Minimal stand-in for ``pydantic_evals.reporting.EvaluationResult``."""
    name: str
    value: Any
    reason: str | None = None
    source: Any = None


@dataclass
class _FakeReportCase:
    """Minimal stand-in for ``pydantic_evals.reporting.ReportCase``."""
    name: str
    assertions: dict[str, _FakeEvaluationResult] = field(default_factory=dict)
    scores: dict[str, _FakeEvaluationResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TestLoadEvalConfig
# ---------------------------------------------------------------------------

class TestLoadEvalConfig:

    def test_valid_config_loads(self):
        config = load_eval_config(FIXTURE_YAML)
        assert "semantic_similarity_rubric" in config
        assert "llm_speedrun_pleias" in config
        assert "hourly_road_traffic" in config

    def test_missing_similarity_rubric_loads_successfully(self, tmp_path):
        p = tmp_path / "no_sim.yaml"
        p.write_text("some_eval:\n  judge:\n    provider: openai\n    model: gpt-4o-mini\n")
        config = load_eval_config(p)
        assert "semantic_similarity_rubric" not in config
        assert "some_eval" in config

    def test_only_reserved_and_underscore_keys_raises(self, tmp_path):
        # Also includes a non-reserved string value (semantic_similarity_rubric_1)
        # to confirm it is not treated as an eval section under the new filter.
        p = tmp_path / "empty_sections.yaml"
        p.write_text(
            "semantic_similarity_rubric: test rubric\n"
            "semantic_similarity_rubric_1: another rubric string\n"
            "_preset: &preset\n"
            "  foo: bar\n"
        )
        with pytest.raises(ValueError, match="at least one evaluation section"):
            load_eval_config(p)

    def test_dict_section_without_judge_key_does_not_count(self, tmp_path):
        # A mapping that lacks a "judge" key is not a runnable eval section;
        # the config should raise because no valid sections remain.
        p = tmp_path / "no_judge.yaml"
        p.write_text(
            "looks_like_a_section:\n"
            "  weights:\n"
            "    criteria: 1.0\n"
        )
        with pytest.raises(ValueError, match="at least one evaluation section"):
            load_eval_config(p)

    def test_underscore_keys_present_in_returned_dict(self, tmp_path):
        p = tmp_path / "anchors.yaml"
        p.write_text(
            "semantic_similarity_rubric: test rubric\n"
            "_defaults: &defaults\n"
            "  foo: bar\n"
            "my_eval:\n"
            "  judge:\n"
            "    provider: openai\n"
            "    model: gpt-4o-mini\n"
        )
        config = load_eval_config(p)
        assert "_defaults" in config
        assert "my_eval" in config

    def test_template_keys_excluded_from_eval_sections(self, tmp_path):
        p = tmp_path / "tpl.yaml"
        p.write_text(
            "_template_source: benchmark_manifest.json\n"
            "my_eval:\n"
            "  judge:\n"
            "    provider: openai\n"
            "    model: gpt-4o-mini\n"
        )
        config = load_eval_config(p)
        assert "_template_source" in config
        assert "my_eval" in config


# ---------------------------------------------------------------------------
# TestLoadTemplateVariables
# ---------------------------------------------------------------------------

class TestLoadTemplateVariables:

    def test_happy_path(self, tmp_path):
        manifest = {"generator": {"n_features": 3, "n_classes": 2, "seed": 42}}
        (tmp_path / "benchmark_manifest.json").write_text(json.dumps(manifest))
        config = {"_template_source": "benchmark_manifest.json"}
        variables = load_template_variables(tmp_path, config)
        assert variables == manifest

    def test_preserves_native_types(self, tmp_path):
        data = {"count": 5, "name": "test", "nested": {"x": 10}}
        (tmp_path / "data.json").write_text(json.dumps(data))
        config = {"_template_source": "data.json"}
        variables = load_template_variables(tmp_path, config)
        assert variables["count"] == 5
        assert isinstance(variables["count"], int)
        assert variables["name"] == "test"
        assert variables["nested"] == {"x": 10}

    def test_no_template_source_returns_empty(self, tmp_path):
        variables = load_template_variables(tmp_path, {})
        assert variables == {}

    def test_missing_file_raises(self, tmp_path):
        config = {"_template_source": "missing.json"}
        with pytest.raises(FileNotFoundError):
            load_template_variables(tmp_path, config)


# ---------------------------------------------------------------------------
# TestRenderCriteria
# ---------------------------------------------------------------------------

class TestRenderCriteria:

    def test_simple_substitution(self):
        criteria = {"Features": "The dataset has {{ n_features }} features."}
        variables = {"n_features": 3}
        result = render_criteria(criteria, variables)
        assert result == {"Features": "The dataset has 3 features."}

    def test_multiple_variables_in_one_rubric(self):
        criteria = {"Shape": "{{ n_train }} train samples, {{ n_features }} features, {{ n_classes }} classes."}
        variables = {"n_train": 102, "n_features": 3, "n_classes": 2}
        result = render_criteria(criteria, variables)
        assert result == {"Shape": "102 train samples, 3 features, 2 classes."}

    def test_missing_variable_raises_with_name(self):
        criteria = {"Features": "Has {{ n_features }} features and {{ n_missing }} something."}
        variables = {"n_features": 3}
        with pytest.raises(ValueError, match="Features.*n_missing"):
            render_criteria(criteria, variables)

    def test_empty_criteria_returns_empty(self):
        assert render_criteria({}, {"n_features": 3}) == {}

    def test_no_placeholders_unchanged(self):
        criteria = {"Static": "This rubric has no placeholders."}
        result = render_criteria(criteria, {"n_features": 3})
        assert result == {"Static": "This rubric has no placeholders."}

    def test_nested_dot_access(self):
        criteria = {"Shape": "{{ generator.n_features }} features, {{ generator.n_classes }} classes."}
        variables = {"generator": {"n_features": 3, "n_classes": 2}}
        result = render_criteria(criteria, variables)
        assert result == {"Shape": "3 features, 2 classes."}


# ---------------------------------------------------------------------------
# TestResolveFileContent
# ---------------------------------------------------------------------------

class TestResolveFileContent:

    @staticmethod
    def _make_adapters():
        ws = DomainAdapter(
            domain_knowledge="ws domain knowledge",
            prompts={"phase1": "ws phase1", "phase2_builder": "ws phase2_builder"},
        )
        ref = DomainAdapter(
            domain_knowledge="ref domain knowledge",
            prompts={"phase1": "ref phase1", "phase2_builder": "ref phase2_builder"},
        )
        return ws, ref

    def test_domain_knowledge_key(self):
        ws, ref = self._make_adapters()
        candidate, reference = resolve_file_content(ws, ref, "domain_knowledge")
        assert candidate == "ws domain knowledge"
        assert reference == "ref domain knowledge"

    def test_prompt_key(self):
        ws, ref = self._make_adapters()
        candidate, reference = resolve_file_content(ws, ref, "phase1")
        assert candidate == "ws phase1"
        assert reference == "ref phase1"

    def test_missing_prompt_key_raises(self):
        ws, ref = self._make_adapters()
        with pytest.raises(KeyError):
            resolve_file_content(ws, ref, "nonexistent_key")


# ---------------------------------------------------------------------------
# TestRubricHash
# ---------------------------------------------------------------------------

class TestRubricHash:

    def test_deterministic(self):
        rubric = "Score on a scale of 1-5"
        assert _rubric_hash(rubric) == _rubric_hash(rubric)

    def test_different_rubrics_different_hashes(self):
        assert _rubric_hash("rubric A") != _rubric_hash("rubric B")

    def test_length_is_eight(self):
        assert len(_rubric_hash("any rubric text")) == 8
        assert len(_rubric_hash("")) == 8


# ---------------------------------------------------------------------------
# TestBuildCriteriaJudges
# ---------------------------------------------------------------------------

class TestBuildCriteriaJudges:

    def test_empty_criteria_returns_empty_list(self):
        model = MagicMock()
        judges = build_criteria_judges(model, {})
        assert judges == []

    def test_no_criteria_key_returns_empty_list(self):
        model = MagicMock()
        judges = build_criteria_judges(model, {"weights": {"criteria": 0.6, "similarity": 0.4}})
        assert judges == []

    def test_n_criteria_returns_n_judges(self):
        model = MagicMock()
        file_config = {
            "criteria": {
                "Columns": "Lists column names",
                "Primary metric": "Identifies the metric",
                "Data quality": "Notes quality issues",
            }
        }
        judges = build_criteria_judges(model, file_config)
        assert len(judges) == 3

    def test_judge_evaluation_names_contain_rubric_hash(self):
        model = MagicMock()
        rubric_text = "Lists column names"
        file_config = {"criteria": {"Columns": rubric_text}}
        judges = build_criteria_judges(model, file_config)
        expected_hash = _rubric_hash(rubric_text)
        # The score config dict contains the evaluation_name
        assert len(judges) == 1
        judge = judges[0]
        assert expected_hash in judge.score["evaluation_name"]
        assert "Columns" in judge.score["evaluation_name"]


# ---------------------------------------------------------------------------
# TestBuildSimilarityJudge
# ---------------------------------------------------------------------------

class TestBuildSimilarityJudge:

    def test_returns_judge_with_expected_output(self):
        model = MagicMock()
        judge = _build_similarity_judge(model, "test rubric")
        assert judge.include_expected_output is True
        assert judge.include_input is False

    def test_evaluation_name_contains_hash(self):
        model = MagicMock()
        rubric = "test rubric"
        judge = _build_similarity_judge(model, rubric)
        expected_hash = _rubric_hash(rubric)
        assert expected_hash in judge.score["evaluation_name"]
        assert "semantic_similarity" in judge.score["evaluation_name"]


# ---------------------------------------------------------------------------
# TestParsePhase0Metrics
# ---------------------------------------------------------------------------

class TestParsePhase0Metrics:

    def test_basic_extraction(self, tmp_path):
        log_path = tmp_path / "phase0_customize.jsonl"
        events = [
            {"type": "api_response", "timestamp": 100.0, "usage": {"input_tokens": 500, "output_tokens": 200}},
            {"type": "tool_call", "timestamp": 101.0, "tool": "read_file"},
            {"type": "api_response", "timestamp": 102.0, "usage": {"input_tokens": 300, "output_tokens": 150}},
            {"type": "tool_call", "timestamp": 103.0, "tool": "shell_exec"},
            {"type": "tool_call", "timestamp": 105.0, "tool": "write_file"},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

        metrics = parse_phase0_metrics(log_path)
        assert metrics["input_tokens"] == 800.0
        assert metrics["output_tokens"] == 350.0
        assert metrics["tool_calls"] == 3.0
        assert metrics["duration_seconds"] == 5.0

    def test_empty_lines_skipped(self, tmp_path):
        log_path = tmp_path / "phase0_customize.jsonl"
        lines = [
            "",
            json.dumps({"type": "api_response", "timestamp": 10.0, "usage": {"input_tokens": 100, "output_tokens": 50}}),
            "   ",
            json.dumps({"type": "tool_call", "timestamp": 12.0, "tool": "read_file"}),
            "",
        ]
        log_path.write_text("\n".join(lines) + "\n")

        metrics = parse_phase0_metrics(log_path)
        assert metrics["input_tokens"] == 100.0
        assert metrics["output_tokens"] == 50.0
        assert metrics["tool_calls"] == 1.0
        assert metrics["duration_seconds"] == 2.0

    def test_single_event_zero_duration(self, tmp_path):
        log_path = tmp_path / "phase0_customize.jsonl"
        log_path.write_text(
            json.dumps({"type": "api_response", "timestamp": 50.0, "usage": {"input_tokens": 10, "output_tokens": 5}}) + "\n"
        )
        metrics = parse_phase0_metrics(log_path)
        assert metrics["duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# TestThresholdEvaluator
# ---------------------------------------------------------------------------

class TestThresholdEvaluator:

    def _evaluate(self, metric_name, value, floor, ceiling):
        evaluator = ThresholdEvaluator(metric_name, value, floor, ceiling)
        ctx = MagicMock()
        return evaluator.evaluate(ctx)

    def test_below_floor(self):
        result = self._evaluate("input_tokens", 100.0, 500.0, 20000.0)
        key = "threshold_input_tokens (below_floor)"
        assert key in result
        assert result[key] == 100.0

    def test_in_band(self):
        result = self._evaluate("input_tokens", 5000.0, 500.0, 20000.0)
        key = "threshold_input_tokens (ok)"
        assert key in result
        assert result[key] == 5000.0

    def test_above_ceiling(self):
        result = self._evaluate("input_tokens", 25000.0, 500.0, 20000.0)
        key = "threshold_input_tokens (above_ceiling)"
        assert key in result
        assert result[key] == 25000.0

    def test_at_floor_boundary_is_ok(self):
        result = self._evaluate("tool_calls", 3.0, 3.0, 25.0)
        assert "threshold_tool_calls (ok)" in result

    def test_at_ceiling_boundary_is_ok(self):
        result = self._evaluate("tool_calls", 25.0, 3.0, 25.0)
        assert "threshold_tool_calls (ok)" in result


# ---------------------------------------------------------------------------
# TestBuildThresholdEvaluators
# ---------------------------------------------------------------------------

class TestBuildThresholdEvaluators:

    def test_all_in_band(self):
        metrics = {"input_tokens": 1000.0, "output_tokens": 500.0}
        thresholds = {
            "input_tokens": {"floor": 500, "ceiling": 20000},
            "output_tokens": {"floor": 200, "ceiling": 15000},
        }
        evaluators, penalty = build_threshold_evaluators(metrics, thresholds)
        assert len(evaluators) == 2
        assert penalty == 1.0

    def test_none_in_band(self):
        metrics = {"input_tokens": 100.0, "output_tokens": 100.0}
        thresholds = {
            "input_tokens": {"floor": 500, "ceiling": 20000},
            "output_tokens": {"floor": 200, "ceiling": 15000},
        }
        _, penalty = build_threshold_evaluators(metrics, thresholds)
        assert penalty == 0.0

    def test_mixed_penalty(self):
        metrics = {"input_tokens": 1000.0, "output_tokens": 100.0, "tool_calls": 10.0}
        thresholds = {
            "input_tokens": {"floor": 500, "ceiling": 20000},
            "output_tokens": {"floor": 200, "ceiling": 15000},
            "tool_calls": {"floor": 3, "ceiling": 25},
        }
        _, penalty = build_threshold_evaluators(metrics, thresholds)
        assert penalty == pytest.approx(2.0 / 3.0)

    def test_missing_metric_defaults_to_zero(self):
        metrics = {}
        thresholds = {"input_tokens": {"floor": 500, "ceiling": 20000}}
        evaluators, penalty = build_threshold_evaluators(metrics, thresholds)
        assert len(evaluators) == 1
        assert penalty == 0.0  # 0.0 < 500 → below_floor

    def test_empty_thresholds_penalty_is_one(self):
        _, penalty = build_threshold_evaluators({"input_tokens": 1000.0}, {})
        assert penalty == 1.0


# ---------------------------------------------------------------------------
# TestExtractFileComposite
# ---------------------------------------------------------------------------

class TestExtractFileComposite:

    def test_with_criteria(self):
        case_result = _FakeReportCase(
            name="eval/domain_knowledge",
            assertions={
                "Columns [abc12345]": _FakeEvaluationResult(name="Columns", value=True),
                "Primary metric [def67890]": _FakeEvaluationResult(name="Primary metric", value=False),
                "Data quality [ghi11111]": _FakeEvaluationResult(name="Data quality", value=True),
            },
            scores={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="similarity", value=4.0),
            },
        )
        weights = {"criteria": 0.6, "similarity": 0.4}
        composite = _extract_file_composite(case_result, criteria_count=3, weights=weights)
        # criteria_pass_rate = 2/3, normalized_similarity = 4.0/5.0
        expected = 0.6 * (2.0 / 3.0) + 0.4 * (4.0 / 5.0)
        assert composite == pytest.approx(expected)

    def test_similarity_only_default_weights(self):
        case_result = _FakeReportCase(
            name="eval/phase1",
            assertions={},
            scores={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="similarity", value=3.5),
            },
        )
        composite = _extract_file_composite(
            case_result, criteria_count=0, weights=DEFAULT_FILE_WEIGHTS,
        )
        # criteria: 0.0 * 0.0 + 1.0 * (3.5/5.0)
        expected = 3.5 / 5.0
        assert composite == pytest.approx(expected)

    def test_default_weights_applied_when_keys_missing(self):
        case_result = _FakeReportCase(
            name="eval/phase1",
            assertions={},
            scores={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="similarity", value=5.0),
            },
        )
        # Pass empty dict — should fall back to DEFAULT_FILE_WEIGHTS
        composite = _extract_file_composite(case_result, criteria_count=0, weights={})
        expected = DEFAULT_FILE_WEIGHTS["criteria"] * 0.0 + DEFAULT_FILE_WEIGHTS["similarity"] * (5.0 / 5.0)
        assert composite == pytest.approx(expected)

    def test_all_criteria_pass(self):
        case_result = _FakeReportCase(
            name="eval/domain_knowledge",
            assertions={
                "Columns [abc12345]": _FakeEvaluationResult(name="Columns", value=True),
                "Metric [def67890]": _FakeEvaluationResult(name="Metric", value=True),
            },
            scores={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="similarity", value=5.0),
            },
        )
        weights = {"criteria": 0.6, "similarity": 0.4}
        composite = _extract_file_composite(case_result, criteria_count=2, weights=weights)
        assert composite == pytest.approx(1.0)

    def test_similarity_assertion_skipped_in_criteria_count(self):
        """semantic_similarity entries in assertions are skipped by the criteria loop."""
        case_result = _FakeReportCase(
            name="eval/dk",
            assertions={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="sim", value=True),
                "Columns [bbb11111]": _FakeEvaluationResult(name="Columns", value=True),
            },
            scores={
                "semantic_similarity [aaa00000]": _FakeEvaluationResult(name="similarity", value=4.0),
            },
        )
        weights = {"criteria": 0.6, "similarity": 0.4}
        composite = _extract_file_composite(case_result, criteria_count=1, weights=weights)
        # Only "Columns" counted as criteria (1 pass / 1 total), similarity assertion skipped
        expected = 0.6 * 1.0 + 0.4 * (4.0 / 5.0)
        assert composite == pytest.approx(expected)

    def test_no_similarity_score_criteria_only(self):
        """When no similarity judge ran, only criteria contribute."""
        case_result = _FakeReportCase(
            name="eval/domain_knowledge",
            assertions={
                "Columns [abc12345]": _FakeEvaluationResult(name="Columns", value=True),
                "Metric [def67890]": _FakeEvaluationResult(name="Metric", value=False),
            },
            scores={},
        )
        weights = {"criteria": 1.0, "similarity": 0.0}
        composite = _extract_file_composite(case_result, criteria_count=2, weights=weights)
        expected = 1.0 * (1.0 / 2.0) + 0.0 * 0.0
        assert composite == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestSectionComposite
# ---------------------------------------------------------------------------

class TestSectionComposite:
    """Pure arithmetic tests for the section composite formula.

    ``section_composite = w_files * mean(file_composites) + w_mechanical * penalty``

    This formula is inline in ``async_main`` — we test it directly.
    """

    @staticmethod
    def _compute(file_composites: list[float], penalty: float, weights: dict[str, float]) -> float:
        files_aggregate = sum(file_composites) / len(file_composites) if file_composites else 0.0
        w_files = weights.get("files", DEFAULT_SECTION_WEIGHTS["files"])
        w_mechanical = weights.get("mechanical", DEFAULT_SECTION_WEIGHTS["mechanical"])
        return w_files * files_aggregate + w_mechanical * penalty

    def test_equal_file_aggregation(self):
        composites = [0.8, 0.6, 0.4]
        penalty = 1.0
        result = self._compute(composites, penalty, {"files": 0.8, "mechanical": 0.2})
        expected = 0.8 * (0.6) + 0.2 * 1.0  # mean = 0.6
        assert result == pytest.approx(expected)

    def test_default_section_weights(self):
        composites = [0.5, 0.5]
        penalty = 0.75
        result = self._compute(composites, penalty, DEFAULT_SECTION_WEIGHTS)
        expected = 0.8 * 0.5 + 0.2 * 0.75
        assert result == pytest.approx(expected)

    def test_custom_section_weights(self):
        composites = [1.0]
        penalty = 0.0
        result = self._compute(composites, penalty, {"files": 0.5, "mechanical": 0.5})
        expected = 0.5 * 1.0 + 0.5 * 0.0
        assert result == pytest.approx(expected)

    def test_perfect_score(self):
        composites = [1.0, 1.0, 1.0]
        penalty = 1.0
        result = self._compute(composites, penalty, {"files": 0.8, "mechanical": 0.2})
        assert result == pytest.approx(1.0)

    def test_zero_score(self):
        composites = [0.0, 0.0]
        penalty = 0.0
        result = self._compute(composites, penalty, {"files": 0.8, "mechanical": 0.2})
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestBuildDataset
# ---------------------------------------------------------------------------

class TestBuildDataset:

    @staticmethod
    def _stub_judge():
        """Build a real LLMJudge that satisfies Dataset validation."""
        return _build_similarity_judge("stub-model", "stub rubric")

    def test_dataset_has_correct_name_and_cases(self):
        from pydantic_evals import Case
        cases = [
            Case(name="test/dk", inputs="candidate dk", expected_output="reference dk"),
            Case(name="test/phase1", inputs="candidate p1", expected_output="reference p1"),
        ]
        sim_judge = self._stub_judge()
        dataset = build_dataset("test", cases, sim_judge)
        assert dataset.name == "test"
        assert len(dataset.cases) == 2
        assert dataset.cases[0].name == "test/dk"
        assert dataset.cases[1].name == "test/phase1"

    def test_similarity_judge_is_dataset_level_evaluator(self):
        from pydantic_evals import Case
        cases = [Case(name="test/dk", inputs="x", expected_output="y")]
        sim_judge = self._stub_judge()
        dataset = build_dataset("test", cases, sim_judge)
        assert len(dataset.evaluators) == 1
        assert dataset.evaluators[0] is sim_judge

    def test_none_similarity_judge_gives_empty_evaluators(self):
        from pydantic_evals import Case
        cases = [Case(name="test/dk", inputs="x", expected_output="y")]
        dataset = build_dataset("test", cases, None)
        assert len(dataset.evaluators) == 0
        assert dataset.name == "test"
        assert len(dataset.cases) == 1
