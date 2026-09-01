"""
AST-aware line-mask repair for SWE-Pruner v2.

The semantic source of truth remains the original `kept_frags`. This module
adds only the structural lines needed to make that mask more syntactically and
logically closed: enclosing scope headers, control headers, branch companion
headers, and dependency definitions/imports.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import typer
from rich.console import Console
from tqdm import tqdm

from train.core.rubric import AstAnalysis, ControlBoundary, ScopeBoundary, SymbolDefinition
from train.inference.ast_parser import PythonAstAnalyzer


app = typer.Typer(help="Repair kept_frags with AST-aware structural closure")
console = Console()


SPECIAL_BRANCH_PREFIXES = ("else:", "elif ", "finally:", "case ")


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_processed_codes(output_jsonl: Path) -> Set[str]:
    processed_codes: Set[str] = set()
    if not output_jsonl.exists():
        return processed_codes
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = item.get("code")
            if isinstance(code, str):
                processed_codes.add(code)
    return processed_codes


def normalize_kept_frags(raw_kept_frags: Any, line_count: int) -> List[int]:
    kept: Set[int] = set()
    if not isinstance(raw_kept_frags, list):
        return []
    for line_no in raw_kept_frags:
        try:
            line_no = int(line_no)
        except Exception:
            continue
        if 1 <= line_no <= line_count:
            kept.add(line_no)
    return sorted(kept)


def compact_ast_metadata(analysis: AstAnalysis) -> Dict[str, Any]:
    return {
        "language": analysis.language,
        "line_count": analysis.line_count,
        "line_spans": analysis.line_spans,
        "parse_ok": analysis.parse_ok,
        "used_tree_sitter": analysis.used_tree_sitter,
        "syntax_error_lines": analysis.syntax_error_lines,
        "dependency_edges": [_model_dump(edge) for edge in analysis.dependency_edges],
        "scope_boundaries": [_model_dump(scope) for scope in analysis.scope_boundaries],
        "control_boundaries": [_model_dump(ctrl) for ctrl in analysis.control_boundaries],
        "symbol_definitions": [_model_dump(item) for item in analysis.symbol_definitions],
    }


def _indent_width(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _line_range(start_line: int, end_line: int, line_count: int) -> List[int]:
    start_line = max(1, min(start_line, line_count))
    end_line = max(start_line, min(end_line, line_count))
    return list(range(start_line, end_line + 1))


def _find_special_branch_header(code_lines: List[str], line_no: int) -> Optional[int]:
    if not (1 <= line_no <= len(code_lines)):
        return None
    current_indent = _indent_width(code_lines[line_no - 1])

    for idx in range(line_no - 1, 0, -1):
        line = code_lines[idx - 1]
        stripped = line.strip()
        if not stripped:
            continue
        indent = _indent_width(line)
        if indent >= current_indent:
            continue
        if stripped.startswith(SPECIAL_BRANCH_PREFIXES):
            return idx
        if stripped.endswith(":") and indent < current_indent:
            break
    return None


def _containing_scopes(analysis: AstAnalysis, line_no: int) -> List[ScopeBoundary]:
    scopes = [
        scope
        for scope in analysis.scope_boundaries
        if scope.kind != "module" and scope.start_line <= line_no <= scope.end_line
    ]
    return sorted(scopes, key=lambda scope: (scope.end_line - scope.start_line, scope.start_line))


def _containing_controls(
    analysis: AstAnalysis,
    line_no: int,
) -> List[ControlBoundary]:
    controls = [
        ctrl
        for ctrl in analysis.control_boundaries
        if ctrl.start_line <= line_no <= ctrl.end_line
    ]
    return sorted(controls, key=lambda ctrl: (ctrl.end_line - ctrl.start_line, ctrl.start_line))


def _definitions_by_line(
    analysis: AstAnalysis,
) -> Dict[int, List[SymbolDefinition]]:
    out: Dict[int, List[SymbolDefinition]] = defaultdict(list)
    for definition in analysis.symbol_definitions:
        out[definition.line].append(definition)
    return out


def _outgoing_edges(analysis: AstAnalysis):
    out = defaultdict(list)
    for edge in analysis.dependency_edges:
        out[edge.source_line].append(edge)
    return out


def _repair_state(
    code_lines: List[str],
    analysis: AstAnalysis,
    original_kept_frags: Iterable[int],
    dependency_hops: int = 2,
) -> Dict[str, Any]:
    line_count = len(code_lines)
    original = sorted({line for line in original_kept_frags if 1 <= line <= line_count})
    repaired: Set[int] = set(original)
    action_keys: Set[tuple] = set()
    repair_actions: List[Dict[str, Any]] = []

    definitions_by_line = _definitions_by_line(analysis)
    outgoing_edges = _outgoing_edges(analysis)

    def add_lines(
        line_numbers: Iterable[int],
        reason: str,
        **extra: Any,
    ) -> Set[int]:
        new_lines: Set[int] = set()
        for line_no in sorted(set(line_numbers)):
            if not (1 <= line_no <= line_count):
                continue
            if line_no in repaired:
                continue
            repaired.add(line_no)
            action_key = (
                line_no,
                reason,
                extra.get("anchor_line"),
                extra.get("symbol"),
                extra.get("kind"),
            )
            if action_key not in action_keys:
                action_keys.add(action_key)
                action = {"line": line_no, "reason": reason}
                for key, value in extra.items():
                    if value is not None:
                        action[key] = value
                repair_actions.append(action)
            new_lines.add(line_no)
        return new_lines

    def apply_structural_closure(seed_lines: Iterable[int]) -> None:
        queue = list(sorted(set(seed_lines)))
        seen: Set[int] = set()

        while queue:
            line_no = queue.pop(0)
            if line_no in seen or not (1 <= line_no <= line_count):
                continue
            seen.add(line_no)

            for scope in _containing_scopes(analysis, line_no):
                added = add_lines(
                    _line_range(
                        scope.header_start_line,
                        scope.header_end_line,
                        line_count,
                    ),
                    f"enclosing_{scope.kind}_header",
                    anchor_line=line_no,
                    kind=scope.kind,
                    symbol=scope.name or None,
                )
                queue.extend(sorted(added))

            for ctrl in _containing_controls(analysis, line_no):
                added = add_lines(
                    _line_range(
                        ctrl.header_start_line,
                        ctrl.header_end_line,
                        line_count,
                    ),
                    f"enclosing_{ctrl.kind}_header",
                    anchor_line=line_no,
                    kind=ctrl.kind,
                )
                queue.extend(sorted(added))

            branch_line = _find_special_branch_header(code_lines, line_no)
            if branch_line is not None:
                branch_kind = code_lines[branch_line - 1].strip().split()[0].rstrip(":")
                added = add_lines(
                    [branch_line],
                    "branch_companion_header",
                    anchor_line=line_no,
                    kind=branch_kind,
                )
                queue.extend(sorted(added))

    apply_structural_closure(repaired)

    frontier = set(repaired)
    for _ in range(max(0, dependency_hops)):
        next_frontier: Set[int] = set()
        for source_line in sorted(frontier):
            for edge in outgoing_edges.get(source_line, []):
                for definition in definitions_by_line.get(edge.target_line, []):
                    added = add_lines(
                        _line_range(
                            definition.header_start_line,
                            definition.header_end_line,
                            line_count,
                        ),
                        f"dependency_{definition.kind}_definition",
                        anchor_line=source_line,
                        symbol=definition.name,
                        kind=definition.kind,
                    )
                    next_frontier.update(added)
        if not next_frontier:
            break
        apply_structural_closure(next_frontier)
        frontier = set(next_frontier)

    return {
        "original_kept_frags": original,
        "repaired_kept_frags": sorted(repaired),
        "repair_actions": sorted(
            repair_actions,
            key=lambda item: (
                item["line"],
                item["reason"],
                item.get("anchor_line", 0),
                item.get("symbol", ""),
            ),
        ),
    }


def repair_mask(
    code: str,
    kept_frags: List[int],
    analysis: AstAnalysis,
    dependency_hops: int = 2,
) -> Dict[str, Any]:
    code_lines = code.split("\n")
    normalized = normalize_kept_frags(kept_frags, len(code_lines))
    return _repair_state(
        code_lines=code_lines,
        analysis=analysis,
        original_kept_frags=normalized,
        dependency_hops=dependency_hops,
    )


def repair_item(
    item: Dict[str, Any],
    analyzer: Optional[PythonAstAnalyzer] = None,
    dependency_hops: int = 2,
    include_ast_metadata: bool = True,
    require_tree_sitter: bool = False,
) -> Dict[str, Any]:
    code = item.get("code")
    if not isinstance(code, str):
        raise ValueError("item must contain a string `code` field")

    analyzer = analyzer or PythonAstAnalyzer(require_tree_sitter=require_tree_sitter)
    analysis = analyzer.analyze(code)
    repaired = repair_mask(
        code=code,
        kept_frags=item.get("kept_frags", []),
        analysis=analysis,
        dependency_hops=dependency_hops,
    )

    output = {
        **item,
        **repaired,
    }
    if include_ast_metadata:
        output["ast_metadata"] = compact_ast_metadata(analysis)
    return output


@app.command()
def main(
    input_file: Path = typer.Option(..., "-i", "--input-file", help="Input JSONL with code and kept_frags"),
    output_jsonl: Path = typer.Option(..., "-o", "--output-jsonl", help="Output JSONL with repaired masks"),
    dependency_hops: int = typer.Option(2, "--dependency-hops", help="Dependency expansion hops"),
    require_tree_sitter: bool = typer.Option(False, "--require-tree-sitter"),
    include_ast_metadata: bool = typer.Option(
        True,
        "--include-ast-metadata/--no-include-ast-metadata",
    ),
    max_items: Optional[int] = typer.Option(None, "--max-items"),
) -> None:
    analyzer = PythonAstAnalyzer(require_tree_sitter=require_tree_sitter)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed_codes = load_processed_codes(output_jsonl)

    read = 0
    written = 0
    skipped = 0
    errors = 0

    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_jsonl, "a", encoding="utf-8") as f_out,
    ):
        for line in tqdm(f_in, desc="Repairing masks"):
            if max_items is not None and written >= max_items:
                break
            read += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            code = item.get("code")
            if not isinstance(code, str):
                skipped += 1
                continue
            if code in processed_codes:
                skipped += 1
                continue
            try:
                repaired = repair_item(
                    item,
                    analyzer=analyzer,
                    dependency_hops=dependency_hops,
                    include_ast_metadata=include_ast_metadata,
                    require_tree_sitter=require_tree_sitter,
                )
            except Exception as exc:
                console.print(f"[yellow]Skipping row {read}: {exc}[/yellow]")
                errors += 1
                continue
            f_out.write(json.dumps(repaired, ensure_ascii=False) + "\n")
            processed_codes.add(code)
            written += 1

    console.rule("Mask repair complete")
    console.print(f"Read: {read}")
    console.print(f"Written: {written}")
    console.print(f"Skipped: {skipped}")
    console.print(f"Errors: {errors}")
    console.print(f"Output: [bold]{output_jsonl}[/bold]")


if __name__ == "__main__":
    app()
