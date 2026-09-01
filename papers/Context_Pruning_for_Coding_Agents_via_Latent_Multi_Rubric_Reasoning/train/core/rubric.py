import math
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field


RUBRIC_DIMENSIONS = ["semantic", "syntax", "dependency", "context"]


def clamp01(value: float) -> float:
    """Clamp a numeric score into the closed interval [0, 1]."""
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


class RubricVector(BaseModel):
    """Line-level multi-objective pruning target."""

    semantic: float = Field(0.0, ge=0.0, le=1.0)
    syntax: float = Field(0.0, ge=0.0, le=1.0)
    dependency: float = Field(0.0, ge=0.0, le=1.0)
    context: float = Field(0.0, ge=0.0, le=1.0)

    def as_list(self) -> List[float]:
        return [self.semantic, self.syntax, self.dependency, self.context]


class DependencyEdge(BaseModel):
    """A static dependency between two 1-based code lines."""

    source_line: int = Field(..., ge=1)
    target_line: int = Field(..., ge=1)
    kind: str
    source_name: str = ""
    target_name: str = ""


class ScopeBoundary(BaseModel):
    """A syntactic scope boundary in 1-based line coordinates."""

    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    kind: str
    name: str = ""
    header_start_line: int = Field(..., ge=1)
    header_end_line: int = Field(..., ge=1)


class ControlBoundary(BaseModel):
    """A control-flow block boundary in 1-based line coordinates."""

    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    kind: str
    header_start_line: int = Field(..., ge=1)
    header_end_line: int = Field(..., ge=1)


class SymbolDefinition(BaseModel):
    """A named definition or binding used for structural repair."""

    name: str
    kind: str
    line: int = Field(..., ge=1)
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    header_start_line: int = Field(..., ge=1)
    header_end_line: int = Field(..., ge=1)


class AstAnalysis(BaseModel):
    """Static-analysis payload used to derive rubric labels."""

    language: str = "python"
    line_count: int = Field(..., ge=0)
    line_spans: List[List[int]] = Field(default_factory=list)
    parse_ok: bool = True
    used_tree_sitter: bool = False
    syntax_error_lines: List[int] = Field(default_factory=list)
    dependency_edges: List[DependencyEdge] = Field(default_factory=list)
    scope_boundaries: List[ScopeBoundary] = Field(default_factory=list)
    control_boundaries: List[ControlBoundary] = Field(default_factory=list)
    symbol_definitions: List[SymbolDefinition] = Field(default_factory=list)


def make_rubric_vector(
    semantic: float,
    syntax: float,
    dependency: float,
    context: float,
) -> List[float]:
    return RubricVector(
        semantic=clamp01(semantic),
        syntax=clamp01(syntax),
        dependency=clamp01(dependency),
        context=clamp01(context),
    ).as_list()


def line_spans_for_code(code: str) -> List[List[int]]:
    """Return character spans for code.split("\\n") lines.

    This intentionally mirrors train.utils.line_chunker.split_code_into_lines so
    labels align with the existing line-level labeling pipeline.
    """
    spans: List[List[int]] = []
    offset = 0
    for line in code.split("\n"):
        start = offset
        end = start + len(line)
        spans.append([start, end])
        offset = end + 1
    return spans


def validate_rubric_scores(
    rubric_scores: Sequence[Sequence[float]],
    line_count: int,
) -> None:
    if len(rubric_scores) != line_count:
        raise ValueError(
            f"rubric_scores length {len(rubric_scores)} does not match "
            f"line_count {line_count}"
        )
    for line_idx, vector in enumerate(rubric_scores, start=1):
        if len(vector) != len(RUBRIC_DIMENSIONS):
            raise ValueError(
                f"line {line_idx} rubric vector has length {len(vector)}, "
                f"expected {len(RUBRIC_DIMENSIONS)}"
            )
        for dim, score in zip(RUBRIC_DIMENSIONS, vector):
            score = float(score)
            if math.isnan(score) or math.isinf(score) or score < 0.0 or score > 1.0:
                raise ValueError(
                    f"line {line_idx} dimension {dim} has invalid score {score}"
                )


def validate_rubric_item(item: Dict[str, Any]) -> None:
    code = item.get("code", "")
    line_count = len(line_spans_for_code(code))
    validate_rubric_scores(item.get("rubric_scores", []), line_count)

    line_spans = item.get("line_spans")
    if line_spans is None:
        return
    if len(line_spans) != line_count:
        raise ValueError(
            f"line_spans length {len(line_spans)} does not match line_count {line_count}"
        )
    for idx, (actual, expected) in enumerate(
        zip(line_spans, line_spans_for_code(code)), start=1
    ):
        if list(actual) != expected:
            raise ValueError(
                f"line {idx} span {actual} does not match expected span {expected}"
            )
