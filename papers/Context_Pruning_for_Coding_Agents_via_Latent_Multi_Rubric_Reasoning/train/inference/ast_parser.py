"""
AST-aware static analysis for SWE-Pruner v2 data enrichment.

The preferred parser is tree-sitter + tree-sitter-python. When those packages
are unavailable, the analyzer falls back to Python's stdlib ast module so tests
and small local checks can still run. Use require_tree_sitter=True when building
production labels if you want missing tree-sitter dependencies to fail fast.
"""

import ast
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from train.core.rubric import (
    AstAnalysis,
    ControlBoundary,
    DependencyEdge,
    ScopeBoundary,
    SymbolDefinition,
    line_spans_for_code,
)


try:
    from tree_sitter import Language, Parser
    import tree_sitter_python

    TREE_SITTER_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only when deps missing
    Language = None
    Parser = None
    tree_sitter_python = None
    TREE_SITTER_IMPORT_ERROR = exc


SCOPE_NODE_TYPES = {
    "function_definition": "function",
    "class_definition": "class",
    "lambda": "lambda",
}


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _build_python_parser():
    if TREE_SITTER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "tree-sitter support is not installed. Install dependencies with "
            "`pip install tree-sitter tree-sitter-python`."
        ) from TREE_SITTER_IMPORT_ERROR

    raw_language = tree_sitter_python.language()
    try:
        language = Language(raw_language)
    except TypeError:
        language = raw_language

    parser = Parser()
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def _point_row(point) -> int:
    return point.row if hasattr(point, "row") else point[0]


def _point_col(point) -> int:
    return point.column if hasattr(point, "column") else point[1]


def _node_line_range(node, line_count: int) -> Tuple[int, int]:
    start_line = _point_row(node.start_point) + 1
    end_line = _point_row(node.end_point) + 1
    if _point_col(node.end_point) == 0 and end_line > start_line:
        end_line -= 1
    start_line = max(1, min(start_line, max(line_count, 1)))
    end_line = max(start_line, min(end_line, max(line_count, 1)))
    return start_line, end_line


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _walk_tree_sitter(node) -> Iterable:
    yield node
    for child in node.children:
        yield from _walk_tree_sitter(child)


def _find_tree_sitter_error_lines(root, line_count: int) -> List[int]:
    error_lines = set()
    for node in _walk_tree_sitter(root):
        is_missing = bool(getattr(node, "is_missing", False))
        if node.type != "ERROR" and not is_missing:
            continue
        start, end = _node_line_range(node, line_count)
        error_lines.update(range(start, end + 1))
    return sorted(error_lines)


def _extract_scope_boundaries_tree_sitter(
    root,
    source_bytes: bytes,
    line_count: int,
) -> List[ScopeBoundary]:
    scopes = [
        ScopeBoundary(
            start_line=1,
            end_line=max(1, line_count),
            kind="module",
            header_start_line=1,
            header_end_line=1,
        )
    ]

    def emit_scope(node, kind: str, start_node=None) -> None:
        start_src = start_node or node
        start_line, _ = _node_line_range(start_src, line_count)
        _, end_line = _node_line_range(node, line_count)
        name_node = node.child_by_field_name("name")
        name = _node_text(name_node, source_bytes) if name_node is not None else ""
        scopes.append(
            ScopeBoundary(
                start_line=start_line,
                end_line=end_line,
                kind=kind,
                name=name,
                header_start_line=start_line,
                header_end_line=start_line,
            )
        )

    for node in _walk_tree_sitter(root):
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in SCOPE_NODE_TYPES:
                    emit_scope(child, SCOPE_NODE_TYPES[child.type], start_node=node)
                    break
        elif node.type in SCOPE_NODE_TYPES:
            emit_scope(node, SCOPE_NODE_TYPES[node.type])

    deduped: Dict[Tuple[str, str, int], ScopeBoundary] = {}
    for scope in scopes:
        key = (scope.kind, scope.name, scope.end_line)
        previous = deduped.get(key)
        if previous is None or scope.start_line < previous.start_line:
            deduped[key] = scope
    return sorted(
        deduped.values(),
        key=lambda scope: (scope.start_line, scope.end_line, scope.kind, scope.name),
    )


def _target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    if isinstance(target, ast.ExceptHandler) and target.name:
        return [target.name]
    return []


def _parse_python_ast(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _node_end_lineno(node: ast.AST, fallback: int) -> int:
    return max(fallback, getattr(node, "end_lineno", fallback))


def _scope_header_start(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1)
    decorator_list = getattr(node, "decorator_list", None) or []
    if decorator_list:
        start = min(decorator.lineno for decorator in decorator_list)
    return start


def _header_end_from_body(node: ast.AST, fallback: int) -> int:
    body = getattr(node, "body", None) or []
    if body:
        first_body_line = getattr(body[0], "lineno", None)
        if first_body_line is not None:
            return max(fallback, first_body_line - 1)
    return fallback


def _statement_range(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", 1)
    return start, _node_end_lineno(node, start)


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.definitions: Dict[str, List[SymbolDefinition]] = defaultdict(list)
        self.symbol_definitions: List[SymbolDefinition] = []

    def add(
        self,
        name: str,
        line: int,
        kind: str,
        start_line: int,
        end_line: int,
        header_start_line: int,
        header_end_line: int,
    ) -> None:
        if not name:
            return
        definition = SymbolDefinition(
            name=name,
            kind=kind,
            line=line,
            start_line=start_line,
            end_line=end_line,
            header_start_line=header_start_line,
            header_end_line=header_end_line,
        )
        self.symbol_definitions.append(definition)
        self.definitions[name].append(definition)

    def visit_Import(self, node: ast.Import) -> None:
        start_line, end_line = _statement_range(node)
        for alias in node.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            self.add(
                name=name,
                line=node.lineno,
                kind="import",
                start_line=start_line,
                end_line=end_line,
                header_start_line=start_line,
                header_end_line=end_line,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        start_line, end_line = _statement_range(node)
        for alias in node.names:
            if alias.name == "*":
                continue
            self.add(
                name=alias.asname or alias.name,
                line=node.lineno,
                kind="import",
                start_line=start_line,
                end_line=end_line,
                header_start_line=start_line,
                header_end_line=end_line,
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        start_line = _scope_header_start(node)
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        self.add(
            name=node.name,
            line=node.lineno,
            kind="function",
            start_line=start_line,
            end_line=end_line,
            header_start_line=start_line,
            header_end_line=header_end_line,
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        start_line = _scope_header_start(node)
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        self.add(
            name=node.name,
            line=node.lineno,
            kind="function",
            start_line=start_line,
            end_line=end_line,
            header_start_line=start_line,
            header_end_line=header_end_line,
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start_line = _scope_header_start(node)
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        self.add(
            name=node.name,
            line=node.lineno,
            kind="class",
            start_line=start_line,
            end_line=end_line,
            header_start_line=start_line,
            header_end_line=header_end_line,
        )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        start_line, end_line = _statement_range(node)
        for target in node.targets:
            for name in _target_names(target):
                self.add(
                    name=name,
                    line=node.lineno,
                    kind="assignment",
                    start_line=start_line,
                    end_line=end_line,
                    header_start_line=start_line,
                    header_end_line=end_line,
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        start_line, end_line = _statement_range(node)
        for name in _target_names(node.target):
            self.add(
                name=name,
                line=node.lineno,
                kind="assignment",
                start_line=start_line,
                end_line=end_line,
                header_start_line=start_line,
                header_end_line=end_line,
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        start_line, end_line = _statement_range(node)
        for name in _target_names(node.target):
            self.add(
                name=name,
                line=node.lineno,
                kind="assignment",
                start_line=start_line,
                end_line=end_line,
                header_start_line=start_line,
                header_end_line=end_line,
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        for name in _target_names(node.target):
            self.add(
                name=name,
                line=node.lineno,
                kind="assignment",
                start_line=node.lineno,
                end_line=end_line,
                header_start_line=node.lineno,
                header_end_line=header_end_line,
            )
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        for name in _target_names(node.target):
            self.add(
                name=name,
                line=node.lineno,
                kind="assignment",
                start_line=node.lineno,
                end_line=end_line,
                header_start_line=node.lineno,
                header_end_line=header_end_line,
            )
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        start_line, end_line = _statement_range(node)
        for item in node.items:
            if item.optional_vars is not None:
                for name in _target_names(item.optional_vars):
                    self.add(
                        name=name,
                        line=node.lineno,
                        kind="assignment",
                        start_line=start_line,
                        end_line=end_line,
                        header_start_line=start_line,
                        header_end_line=end_line,
                    )
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            line = getattr(node, "lineno", 1)
            header_end_line = _header_end_from_body(node, line)
            end_line = _node_end_lineno(node, header_end_line)
            self.add(
                name=node.name,
                line=line,
                kind="assignment",
                start_line=line,
                end_line=end_line,
                header_start_line=line,
                header_end_line=header_end_line,
            )
        self.generic_visit(node)


def _collect_symbol_definitions_ast(
    tree: Optional[ast.AST],
) -> Tuple[List[SymbolDefinition], Dict[str, List[SymbolDefinition]]]:
    if tree is None:
        return [], defaultdict(list)

    collector = _DefinitionCollector()
    collector.visit(tree)
    return (
        sorted(
            collector.symbol_definitions,
            key=lambda item: (item.start_line, item.end_line, item.kind, item.name),
        ),
        collector.definitions,
    )


def _nearest_definition(
    definitions: Dict[str, List[SymbolDefinition]],
    name: str,
    line: int,
) -> Optional[SymbolDefinition]:
    candidates = [entry for entry in definitions.get(name, []) if entry.line <= line]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.line)


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _extract_dependency_edges_ast(
    tree: Optional[ast.AST],
    definitions: Dict[str, List[SymbolDefinition]],
) -> List[DependencyEdge]:
    if tree is None:
        return []
    edges: Dict[Tuple[int, int, str, str], DependencyEdge] = {}

    def add_edge(source_line: int, target_line: int, kind: str, name: str) -> None:
        if source_line <= 0 or target_line <= 0 or source_line == target_line:
            return
        key = (source_line, target_line, kind, name)
        edges[key] = DependencyEdge(
            source_line=source_line,
            target_line=target_line,
            kind=kind,
            source_name=name,
            target_name=name,
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            found = _nearest_definition(definitions, node.id, node.lineno)
            if found is not None:
                add_edge(node.lineno, found.line, "uses", node.id)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if not name:
                continue
            found = _nearest_definition(definitions, name, node.lineno)
            if found is not None and found.kind in {"function", "class"}:
                add_edge(node.lineno, found.line, "calls", name)

    return sorted(
        edges.values(),
        key=lambda edge: (edge.source_line, edge.target_line, edge.kind, edge.source_name),
    )


def _syntax_status_ast(code: str) -> Tuple[bool, List[int]]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, [exc.lineno] if exc.lineno else []
    return True, []


def _extract_scope_boundaries_ast(
    tree: Optional[ast.AST],
    line_count: int,
) -> List[ScopeBoundary]:
    scopes = [
        ScopeBoundary(
            start_line=1,
            end_line=max(1, line_count),
            kind="module",
            header_start_line=1,
            header_end_line=1,
        )
    ]
    if tree is None:
        return scopes

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        header_start_line = _scope_header_start(node)
        header_end_line = _header_end_from_body(node, node.lineno)
        end_line = _node_end_lineno(node, header_end_line)
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        scopes.append(
            ScopeBoundary(
                start_line=max(1, header_start_line),
                end_line=max(header_start_line, end_line),
                kind=kind,
                name=node.name,
                header_start_line=max(1, header_start_line),
                header_end_line=max(header_start_line, header_end_line),
            )
        )

    return sorted(
        scopes,
        key=lambda scope: (scope.start_line, scope.end_line, scope.kind, scope.name),
    )


def _extract_control_boundaries_ast(
    tree: Optional[ast.AST],
) -> List[ControlBoundary]:
    if tree is None:
        return []

    controls: List[ControlBoundary] = []
    for node in ast.walk(tree):
        kind = None
        if isinstance(node, ast.If):
            kind = "if"
        elif isinstance(node, ast.For):
            kind = "for"
        elif isinstance(node, ast.AsyncFor):
            kind = "async_for"
        elif isinstance(node, ast.While):
            kind = "while"
        elif isinstance(node, ast.With):
            kind = "with"
        elif isinstance(node, ast.AsyncWith):
            kind = "async_with"
        elif isinstance(node, ast.Try):
            kind = "try"
        elif isinstance(node, ast.ExceptHandler):
            kind = "except"
        elif hasattr(ast, "Match") and isinstance(node, ast.Match):
            kind = "match"
        if kind is None:
            continue
        start_line = getattr(node, "lineno", 1)
        header_end_line = _header_end_from_body(node, start_line)
        end_line = _node_end_lineno(node, header_end_line)
        controls.append(
            ControlBoundary(
                start_line=start_line,
                end_line=end_line,
                kind=kind,
                header_start_line=start_line,
                header_end_line=header_end_line,
            )
        )

    return sorted(
        controls,
        key=lambda item: (item.start_line, item.end_line, item.kind),
    )


class PythonAstAnalyzer:
    def __init__(self, require_tree_sitter: bool = False):
        self.require_tree_sitter = require_tree_sitter
        self.parser = None
        if TREE_SITTER_IMPORT_ERROR is None:
            try:
                self.parser = _build_python_parser()
            except Exception as exc:
                if require_tree_sitter:
                    raise RuntimeError(
                        "Failed to initialize tree-sitter Python parser"
                    ) from exc
                self.parser = None
        elif require_tree_sitter:
            _build_python_parser()

    def analyze(self, code: str) -> AstAnalysis:
        line_spans = line_spans_for_code(code)
        line_count = len(line_spans)
        py_tree = _parse_python_ast(code)
        symbol_definitions, definitions_by_name = _collect_symbol_definitions_ast(py_tree)
        dependency_edges = _extract_dependency_edges_ast(py_tree, definitions_by_name)
        scope_boundaries = _extract_scope_boundaries_ast(py_tree, line_count)
        control_boundaries = _extract_control_boundaries_ast(py_tree)

        if self.parser is None:
            parse_ok, syntax_error_lines = _syntax_status_ast(code)
            return AstAnalysis(
                line_count=line_count,
                line_spans=line_spans,
                parse_ok=parse_ok,
                used_tree_sitter=False,
                syntax_error_lines=syntax_error_lines,
                dependency_edges=dependency_edges,
                scope_boundaries=scope_boundaries,
                control_boundaries=control_boundaries,
                symbol_definitions=symbol_definitions,
            )

        source_bytes = code.encode("utf-8")
        tree = self.parser.parse(source_bytes)
        root = tree.root_node
        syntax_error_lines = _find_tree_sitter_error_lines(root, line_count)
        return AstAnalysis(
            line_count=line_count,
            line_spans=line_spans,
            parse_ok=not root.has_error,
            used_tree_sitter=True,
            syntax_error_lines=syntax_error_lines,
            dependency_edges=dependency_edges,
            scope_boundaries=scope_boundaries,
            control_boundaries=control_boundaries,
            symbol_definitions=symbol_definitions,
        )


def analyze_python_code(
    code: str,
    require_tree_sitter: bool = False,
) -> AstAnalysis:
    return PythonAstAnalyzer(require_tree_sitter=require_tree_sitter).analyze(code)


def analysis_to_dict(analysis: AstAnalysis) -> Dict:
    return _model_dump(analysis)
