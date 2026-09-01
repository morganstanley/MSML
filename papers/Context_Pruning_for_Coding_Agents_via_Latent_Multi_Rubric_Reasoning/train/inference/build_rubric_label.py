"""
Build SWE-Pruner v2 multi-objective rubric labels.

Recommended input rows come from the repaired v2 dataset:
  {
    "query": str,
    "code": str,
    "score": float,
    "final_kept_frags": [1-based lines],
    "accepted": bool,
    ...
  }

Output rows preserve all original fields and add:
  - rubric_schema: ["semantic", "syntax", "dependency", "context"]
  - rubric_scores: one 4-float vector per code line
  - line_spans: character spans for each code.split("\\n") line
  - structural_metadata: optional AST/dependency metadata
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import typer
from rich.console import Console
from tqdm import tqdm

from train.core.rubric import (
    RUBRIC_DIMENSIONS,
    AstAnalysis,
    DependencyEdge,
    clamp01,
    make_rubric_vector,
    validate_rubric_item,
)
from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.row_identity import build_row_identity, load_processed_row_ids


app = typer.Typer(help="Build AST-aware multi-objective rubric labels")
console = Console()


COMPOUND_PREFIXES = (
    "if ",
    "elif ",
    "else:",
    "for ",
    "async for ",
    "while ",
    "with ",
    "async with ",
    "try:",
    "except",
    "finally:",
    "def ",
    "async def ",
    "class ",
    "match ",
    "case ",
)


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def normalize_kept_frags(raw_kept_frags: Any, line_count: int) -> Set[int]:
    kept: Set[int] = set()
    if not isinstance(raw_kept_frags, list):
        return kept
    for line_no in raw_kept_frags:
        try:
            line_no = int(line_no)
        except Exception:
            continue
        if 1 <= line_no <= line_count:
            kept.add(line_no)
    return kept


def resolve_semantic_seed_lines(
    item: Dict[str, Any],
    line_count: int,
    semantic_label_source: str = "auto",
) -> tuple[str, Set[int]]:
    if semantic_label_source == "auto":
        candidates = (
            "final_kept_frags",
            "kept_frags",
            "repaired_kept_frags",
            "original_kept_frags",
        )
    elif semantic_label_source == "final-kept-frags":
        candidates = ("final_kept_frags",)
    elif semantic_label_source == "kept-frags":
        candidates = ("kept_frags",)
    elif semantic_label_source == "repaired-kept-frags":
        candidates = ("repaired_kept_frags",)
    elif semantic_label_source == "original-kept-frags":
        candidates = ("original_kept_frags",)
    else:
        raise ValueError(
            "semantic_label_source must be one of: auto, final-kept-frags, "
            "kept-frags, repaired-kept-frags, original-kept-frags"
        )

    for field_name in candidates:
        kept = normalize_kept_frags(item.get(field_name, []), line_count)
        if kept:
            return field_name, kept
    return candidates[0], set()


def detect_parent_dataset(item: Dict[str, Any]) -> str:
    if any(
        key in item
        for key in (
            "final_kept_frags",
            "repaired_kept_frags",
            "original_kept_frags",
            "judge_evaluation",
            "accepted",
        )
    ):
        return "v2-repaired"
    return "v1-labeled"


def _iter_edge_lines(edges: Iterable[DependencyEdge]) -> Iterable[int]:
    for edge in edges:
        yield edge.source_line
        yield edge.target_line


def _is_compound_line(stripped: str) -> bool:
    return stripped.endswith(":") or stripped.startswith(COMPOUND_PREFIXES)


def compute_semantic_scores(
    score: float,
    kept_frags: Set[int],
    line_count: int,
    semantic_mode: str,
) -> List[float]:
    score = clamp01(score)
    scores = [0.0] * line_count
    if semantic_mode == "score-only":
        return [score] * line_count
    if semantic_mode != "kept-frags":
        raise ValueError(
            f"Unsupported semantic_mode={semantic_mode!r}; expected kept-frags or score-only"
        )
    for line_no in kept_frags:
        scores[line_no - 1] = score
    return scores


def compute_syntax_scores(code: str, analysis: AstAnalysis) -> List[float]:
    scope_starts = {
        scope.start_line
        for scope in analysis.scope_boundaries
        if scope.kind != "module"
    }
    error_lines = set(analysis.syntax_error_lines)
    scores: List[float] = []

    for idx, line in enumerate(code.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            scores.append(0.0)
            continue
        if idx in error_lines:
            scores.append(0.0)
            continue

        value = 0.15 if analysis.parse_ok else 0.05
        if idx in scope_starts:
            value = max(value, 1.0)
        if _is_compound_line(stripped):
            value = max(value, 1.0)
        if stripped.startswith("@"):
            value = max(value, 0.75)
        if stripped.startswith("import ") or stripped.startswith("from "):
            value = max(value, 0.60)
        if stripped.endswith(("\\", "(", "[", "{", ",", "|")):
            value = max(value, 0.70)
        scores.append(clamp01(value))

    return scores


def compute_dependency_scores(
    analysis: AstAnalysis,
    kept_frags: Set[int],
    dependency_hops: int,
) -> List[float]:
    scores = [0.0] * analysis.line_count
    source_to_targets: Dict[int, Set[int]] = defaultdict(set)
    target_to_sources: Dict[int, Set[int]] = defaultdict(set)

    for edge in analysis.dependency_edges:
        if not (1 <= edge.source_line <= analysis.line_count):
            continue
        if not (1 <= edge.target_line <= analysis.line_count):
            continue
        source_to_targets[edge.source_line].add(edge.target_line)
        target_to_sources[edge.target_line].add(edge.source_line)

    if not kept_frags:
        for line_no in _iter_edge_lines(analysis.dependency_edges):
            if 1 <= line_no <= analysis.line_count:
                scores[line_no - 1] = max(scores[line_no - 1], 0.20)
        return scores

    for line_no in kept_frags:
        scores[line_no - 1] = max(scores[line_no - 1], 0.50)

    visited = set(kept_frags)
    frontier = set(kept_frags)
    for hop in range(max(1, dependency_hops)):
        next_frontier: Set[int] = set()
        dependency_weight = max(0.35, 1.0 - hop * 0.20)
        dependent_weight = max(0.25, 0.75 - hop * 0.20)

        for line_no in frontier:
            for target_line in source_to_targets.get(line_no, set()):
                scores[target_line - 1] = max(scores[target_line - 1], dependency_weight)
                if target_line not in visited:
                    next_frontier.add(target_line)
            for source_line in target_to_sources.get(line_no, set()):
                scores[source_line - 1] = max(scores[source_line - 1], dependent_weight)
                if source_line not in visited:
                    next_frontier.add(source_line)

        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return scores


def compute_context_scores(
    analysis: AstAnalysis,
    kept_frags: Set[int],
    dependency_scores: List[float],
    context_window: int,
) -> List[float]:
    scores = [0.0] * analysis.line_count

    for line_no in kept_frags:
        scores[line_no - 1] = max(scores[line_no - 1], 0.75)
        start = max(1, line_no - context_window)
        end = min(analysis.line_count, line_no + context_window)
        for neighbor in range(start, end + 1):
            scores[neighbor - 1] = max(scores[neighbor - 1], 0.50)

    for scope in analysis.scope_boundaries:
        if scope.kind == "module":
            continue
        if any(scope.start_line <= kept <= scope.end_line for kept in kept_frags):
            scores[scope.start_line - 1] = max(scores[scope.start_line - 1], 1.0)
            if 1 <= scope.end_line <= analysis.line_count:
                scores[scope.end_line - 1] = max(scores[scope.end_line - 1], 0.50)

    for idx, dependency_score in enumerate(dependency_scores, start=1):
        if dependency_score >= 0.75:
            scores[idx - 1] = max(scores[idx - 1], 0.50)

    return scores


def compute_rubric_scores(
    item: Dict[str, Any],
    analysis: AstAnalysis,
    semantic_mode: str = "kept-frags",
    semantic_label_source: str = "auto",
    dependency_hops: int = 1,
    context_window: int = 1,
) -> List[List[float]]:
    _, kept_frags = resolve_semantic_seed_lines(
        item,
        analysis.line_count,
        semantic_label_source=semantic_label_source,
    )
    score = clamp01(item.get("score", 0.0))

    semantic_scores = compute_semantic_scores(
        score,
        kept_frags,
        analysis.line_count,
        semantic_mode=semantic_mode,
    )
    syntax_scores = compute_syntax_scores(item["code"], analysis)
    dependency_scores = compute_dependency_scores(
        analysis,
        kept_frags,
        dependency_hops=dependency_hops,
    )
    context_scores = compute_context_scores(
        analysis,
        kept_frags,
        dependency_scores,
        context_window=context_window,
    )

    return [
        make_rubric_vector(semantic, syntax, dependency, context)
        for semantic, syntax, dependency, context in zip(
            semantic_scores,
            syntax_scores,
            dependency_scores,
            context_scores,
        )
    ]


def compact_structural_metadata(analysis: AstAnalysis) -> Dict[str, Any]:
    return {
        "language": analysis.language,
        "line_count": analysis.line_count,
        "parse_ok": analysis.parse_ok,
        "used_tree_sitter": analysis.used_tree_sitter,
        "syntax_error_lines": analysis.syntax_error_lines,
        "dependency_edges": [_model_dump(edge) for edge in analysis.dependency_edges],
        "scope_boundaries": [_model_dump(scope) for scope in analysis.scope_boundaries],
    }


def enrich_item(
    item: Dict[str, Any],
    analyzer: Optional[PythonAstAnalyzer] = None,
    require_tree_sitter: bool = False,
    semantic_mode: str = "kept-frags",
    semantic_label_source: str = "auto",
    dependency_hops: int = 1,
    context_window: int = 1,
    include_structural_metadata: bool = True,
) -> Dict[str, Any]:
    if "code" not in item or not isinstance(item["code"], str):
        raise ValueError("item must contain a string `code` field")

    analyzer = analyzer or PythonAstAnalyzer(require_tree_sitter=require_tree_sitter)
    analysis = analyzer.analyze(item["code"])
    resolved_semantic_source, _ = resolve_semantic_seed_lines(
        item,
        analysis.line_count,
        semantic_label_source=semantic_label_source,
    )
    enriched = {
        **item,
        "rubric_schema": RUBRIC_DIMENSIONS,
        "rubric_scores": compute_rubric_scores(
            item,
            analysis,
            semantic_mode=semantic_mode,
            semantic_label_source=semantic_label_source,
            dependency_hops=dependency_hops,
            context_window=context_window,
        ),
        "line_spans": analysis.line_spans,
        "rubric_parent_dataset": detect_parent_dataset(item),
        "rubric_semantic_source": resolved_semantic_source,
    }
    if include_structural_metadata:
        enriched["structural_metadata"] = compact_structural_metadata(analysis)
    validate_rubric_item(enriched)
    return enriched


@app.command()
def main(
    input_file: Path = typer.Option(
        "swe-pruner-training-dataset-py-v2.jsonl",
        "-i",
        "--input-file",
        help="Input repaired v2 JSONL; accepted rows are preferred",
    ),
    output_jsonl: Path = typer.Option(
        "swe-pruner-training-dataset-py-v2-rubric.jsonl",
        "-o",
        "--output-jsonl",
        help="Output enriched JSONL",
    ),
    require_tree_sitter: bool = typer.Option(
        False,
        "--require-tree-sitter",
        help="Fail if tree-sitter/tree-sitter-python are not installed",
    ),
    semantic_mode: str = typer.Option(
        "kept-frags",
        "--semantic-mode",
        help="Semantic target source: kept-frags or score-only",
    ),
    semantic_label_source: str = typer.Option(
        "auto",
        "--semantic-label-source",
        help=(
            "Which line-mask field seeds the semantic dimension: auto, "
            "final-kept-frags, kept-frags, repaired-kept-frags, or original-kept-frags"
        ),
    ),
    dependency_hops: int = typer.Option(
        1,
        "--dependency-hops",
        help="Transitive dependency hops from semantic seed lines",
    ),
    context_window: int = typer.Option(
        1,
        "--context-window",
        help="Neighboring line window around semantic seed lines",
    ),
    max_items: Optional[int] = typer.Option(
        None,
        "--max-items",
        help="Optional cap for smoke tests",
    ),
    max_code_length: int = typer.Option(
        0,
        "--max-code-length",
        help="Skip code longer than this many characters; 0 disables",
    ),
    include_structural_metadata: bool = typer.Option(
        True,
        "--include-structural-metadata/--no-include-structural-metadata",
        help="Store dependency edges and scope boundaries in each output row",
    ),
    accepted_only: bool = typer.Option(
        True,
        "--accepted-only/--allow-unaccepted",
        help="Process only rows accepted by the repaired v2 judge when that field exists",
    ),
) -> None:
    analyzer = PythonAstAnalyzer(require_tree_sitter=require_tree_sitter)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_row_ids = load_processed_row_ids(output_jsonl)
    console.print(f"Found {len(processed_row_ids)} already processed items")

    total = 0
    written = 0
    skipped = 0
    errors = 0

    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_jsonl, "a", encoding="utf-8") as f_out,
    ):
        for line in tqdm(f_in, desc="Building rubric labels"):
            if max_items is not None and written >= max_items:
                break
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            code = item.get("code")
            if not isinstance(code, str):
                skipped += 1
                continue
            if accepted_only and item.get("accepted") is False:
                skipped += 1
                continue
            row_id = build_row_identity(item)
            if row_id in processed_row_ids:
                skipped += 1
                continue
            if max_code_length and len(code) > max_code_length:
                skipped += 1
                continue

            try:
                enriched = enrich_item(
                    item,
                    analyzer=analyzer,
                    semantic_mode=semantic_mode,
                    semantic_label_source=semantic_label_source,
                    dependency_hops=dependency_hops,
                    context_window=context_window,
                    include_structural_metadata=include_structural_metadata,
                )
            except Exception as exc:
                errors += 1
                console.print(f"[yellow]Skipping row {total}: {exc}[/yellow]")
                continue

            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            written += 1
            processed_row_ids.add(row_id)

    console.rule("Rubric labeling complete")
    console.print(f"Read: {total}")
    console.print(f"Written: {written}")
    console.print(f"Skipped: {skipped}")
    console.print(f"Errors: {errors}")
    console.print(f"Output: [bold]{output_jsonl}[/bold]")


if __name__ == "__main__":
    app()
