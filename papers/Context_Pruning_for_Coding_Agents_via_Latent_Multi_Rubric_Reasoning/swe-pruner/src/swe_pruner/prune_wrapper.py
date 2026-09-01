import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from transformers import AutoTokenizer

from .runtime_ast import (
    ControlBoundary,
    RuntimeAstGraph,
    RuntimePythonAstAnalyzer,
    ScopeBoundary,
    SymbolDefinition,
)
from .swepruner import SwePrunerForCodeCompression, SwePrunerOutput
from pydantic import BaseModel


class PruneRequest(BaseModel):
    query: str
    code: str
    threshold: float = 0.5
    always_keep_first_frags: bool = False
    always_keep_last_frags: bool = False
    chunk_overlap_tokens: int = 50


class PruneResponse(BaseModel):
    score: float
    pruned_code: str
    token_scores: List[List[str | float]]  # [[token_str, score], ...]
    kept_frags: List[int]
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    error_msg: Optional[str] = None


def format_instruction(instruction: Optional[str], query: str) -> str:
    """Format instruction and query (LLM style)."""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: "


def estimate_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    """Estimate token count for a text."""
    enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return len(enc["input_ids"])


def split_code_into_chunks(
    code: str,
    tokenizer: AutoTokenizer,
    chunk_max_tokens: int,
    overlap_tokens: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    Split code into chunks with overlap, based on actual token counts.

    Args:
        code: Full code string
        tokenizer: Tokenizer for counting tokens
        chunk_max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if not code:
        return []

    # Tokenize full code with offsets to track character positions
    code_enc = tokenizer(
        code,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    total_tokens = len(code_enc["input_ids"])
    offsets = code_enc["offset_mapping"]

    # If code fits in one chunk, return as-is
    if total_tokens <= chunk_max_tokens:
        return [(code, 0, len(code))]

    # Sanity check: chunk_max_tokens should be large enough for meaningful chunking
    if chunk_max_tokens < overlap_tokens:
        overlap_tokens = 0

    chunks = []
    stride = chunk_max_tokens - overlap_tokens

    if stride <= 0:
        raise ValueError(
            f"Invalid configuration: stride={stride} "
            f"(chunk_max_tokens={chunk_max_tokens}, overlap_tokens={overlap_tokens})"
        )

    start_token_idx = 0

    while start_token_idx < total_tokens:
        end_token_idx = min(start_token_idx + chunk_max_tokens, total_tokens)

        # Get character positions for this chunk
        start_char = (
            offsets[start_token_idx][0] if start_token_idx < len(offsets) else 0
        )
        # For end, use the end of the last token, but ensure we don't go past code length
        if end_token_idx <= len(offsets):
            end_char = offsets[end_token_idx - 1][1]
        else:
            end_char = len(code)

        chunk_text = code[start_char:end_char]
        chunks.append((chunk_text, start_char, end_char))

        # Move to next chunk with overlap
        if end_token_idx >= total_tokens:
            break

        start_token_idx += stride

    return chunks


def build_input_for_llm(
    query: str,
    code: str,
    tokenizer: AutoTokenizer,
    max_length: int = 8192,
    instruction: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Build input tensors for LLM-style model inference."""
    # Format query
    formatted_query = format_instruction(instruction, query)

    # Tokenize query and code
    query_enc = tokenizer(
        formatted_query,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    code_enc = tokenizer(
        code,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    query_ids = query_enc["input_ids"]
    code_ids = code_enc["input_ids"]

    # LLM prefix and suffix (matching train.py)
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # Calculate available length
    available_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    query_len = len(query_ids)
    code_len = len(code_ids)

    # Truncate code if necessary (and corresponding offsets)
    if query_len + code_len > available_length:
        truncate_to = available_length - query_len
        code_ids = code_ids[:truncate_to]
        code_offsets = code_enc["offset_mapping"][:truncate_to]
        code_len = len(code_ids)
    else:
        # Use original offsets
        code_offsets = code_enc["offset_mapping"]

    # Build full sequence. At inference time we keep the sequence tight instead of
    # padding to max_length, otherwise tiny runtime requests pay the full 8k-token
    # attention cost on every call.
    input_ids = prefix_tokens + query_ids + code_ids + suffix_tokens
    attention_mask = [1] * len(input_ids)

    # Calculate code token positions
    # doc_start = pad_len + len(prefix_tokens) + query_len
    # doc_end = doc_start + code_len
    doc_start = len(prefix_tokens) + query_len
    doc_end = doc_start + code_len

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "doc_start": doc_start,
        "doc_end": doc_end,
        "code_offsets": code_offsets,
        "code_len": code_len,
    }


def aggregate_token_scores_to_lines(
    code: str,
    token_scores: List[Tuple[str, float]],
    token_offsets: List[Tuple[int, int]],
) -> Dict[int, float]:
    """Aggregate token scores to line-level scores based on character offsets and code lines.

    Args:
        code: Source code string
        token_scores: List of (token_str, score) tuples
        token_offsets: List of (start_char, end_char) tuples matching token_scores

    Returns:
        Dict mapping line number (1-indexed) to aggregated score.
        Only lines with token coverage are included in the result.
        Lines without tokens will not appear in the dict (treated as low relevance by pruner).
    """
    line_scores_dict: Dict[int, float] = {}

    # Build mapping from character position to score
    char_to_score: Dict[int, float] = {}
    for (token_str, score), (start, end) in zip(token_scores, token_offsets):
        score = max(0.0, min(1.0, float(score)))
        # Clamp to valid range
        start = max(0, min(start, len(code)))
        end = max(0, min(end, len(code)))
        for pos in range(start, end):
            char_to_score[pos] = score

    # Split code into lines and compute line scores
    lines = code.splitlines(keepends=False)
    current_pos = 0

    for line_num, line_text in enumerate(lines, start=1):
        line_start = current_pos
        line_end = current_pos + len(line_text)

        # Collect scores for all characters in this line
        line_scores = []
        for char_pos in range(line_start, line_end):
            if char_pos in char_to_score:
                line_scores.append(char_to_score[char_pos])

        # Only add to dict if this line has token coverage
        # Lines without tokens are treated as having no evidence (not included)
        if line_scores:
            line_scores_dict[line_num] = float(sum(line_scores) / len(line_scores))

        # Move to next line (account for newline character)
        current_pos = line_end + 1  # +1 for the newline character

    return line_scores_dict


def prune_code_lines(
    code: str,
    line_scores: Dict[int, float],
    threshold: float,
    always_keep_first_frags: bool = False,
    always_keep_last_frags: bool = False,
    ast_graph: Optional[RuntimeAstGraph] = None,
) -> Tuple[str, List[int]]:
    """Prune code at line level, similar to model.py lines 249-272.

    Returns:
        Tuple of (pruned_code, kept_frags) where kept_frags is list of kept line numbers
    """
    lines = code.splitlines()
    kept_lines = []
    num_first_frags = 1 if always_keep_first_frags else 0
    num_last_frags = 1 if always_keep_last_frags else 0
    first_last_line_to_keep = (
        max(1, len(lines) - num_last_frags + 1) if num_last_frags else len(lines) + 1
    )

    # Determine which lines to keep
    for line_num in range(1, len(lines) + 1):
        should_keep = False
        if line_num <= num_first_frags:
            should_keep = True
        elif line_num >= first_last_line_to_keep:
            should_keep = True
        elif line_num in line_scores and line_scores[line_num] >= threshold:
            should_keep = True

        if should_keep:
            kept_lines.append(line_num)

    if ast_graph is not None:
        kept_lines = apply_ast_constraints(code, kept_lines, ast_graph)

    # Build pruned code with comment placeholders for omitted spans.
    kept_code_lines = []
    filtered_lines_cnt = 0
    filtered_start_line = 0
    s_format = "# (filtered {} lines)"
    pass_format = "pass  # (filtered {} lines)"
    last_kept_original_line = 0

    def flush_filtered_block(next_kept_line: Optional[int]) -> None:
        nonlocal filtered_lines_cnt, filtered_start_line
        if filtered_lines_cnt <= 0:
            return
        anchor_line = filtered_start_line
        if not (1 <= anchor_line <= len(lines)) and next_kept_line is not None:
            anchor_line = max(1, min(next_kept_line, len(lines)))
        indent = ""
        if 1 <= anchor_line <= len(lines):
            anchor_text = lines[anchor_line - 1]
            indent = _leading_whitespace(anchor_text)

        placeholder = s_format.format(filtered_lines_cnt)
        if 1 <= last_kept_original_line <= len(lines):
            previous_line = lines[last_kept_original_line - 1]
            previous_indent = _indent_width(previous_line)
            next_indent = (
                _indent_width(lines[next_kept_line - 1])
                if next_kept_line is not None and 1 <= next_kept_line <= len(lines)
                else -1
            )
            if previous_line.rstrip().endswith(":") and (
                next_kept_line is None or next_indent <= previous_indent
            ):
                suite_indent = _suite_header_base_indent(
                    ast_graph,
                    lines,
                    last_kept_original_line,
                )
                indent = suite_indent or indent
                if not indent and 1 <= filtered_start_line <= len(lines):
                    indent = _leading_whitespace(lines[filtered_start_line - 1])
                if suite_indent is None and _indent_width(indent) <= previous_indent:
                    indent = _leading_whitespace(previous_line) + "    "
                placeholder = pass_format.format(filtered_lines_cnt)

        kept_code_lines.append(indent + placeholder)
        filtered_lines_cnt = 0
        filtered_start_line = 0

    for line in range(1, len(lines) + 1):
        if lines[line - 1].strip() == "":
            filtered_lines_cnt += 1
            if filtered_start_line == 0:
                filtered_start_line = line
            continue  # Skip empty lines
        if line not in kept_lines:
            filtered_lines_cnt += 1
            if filtered_start_line == 0:
                filtered_start_line = line
        else:
            if filtered_lines_cnt > 0:
                flush_filtered_block(line)
            kept_code_lines.append(lines[line - 1])
            last_kept_original_line = line
    if filtered_lines_cnt > 0:
        flush_filtered_block(None)
    pruned_code = "\n".join(kept_code_lines)

    return pruned_code, kept_lines


SPECIAL_BRANCH_PREFIXES = ("else:", "elif ", "finally:", "case ")


def aggregate_objective_token_logits(
    token_logits: Optional[torch.Tensor],
    rubric_token_logits: Optional[torch.Tensor],
    gating_weights: Optional[torch.Tensor],
    gating_type: str = "softmax",
) -> torch.Tensor:
    if rubric_token_logits is None:
        if token_logits is None:
            raise ValueError("Expected token_logits or rubric_token_logits from model")
        return token_logits.float()

    rubric_token_logits = rubric_token_logits.float()
    if rubric_token_logits.dim() != 3:
        raise ValueError(
            f"Expected rubric_token_logits to have shape [B, L, K], got {tuple(rubric_token_logits.shape)}"
        )

    if rubric_token_logits.size(-1) == 1:
        return rubric_token_logits.squeeze(-1)

    if gating_weights is None:
        gating_weights = torch.full_like(
            rubric_token_logits,
            1.0 / float(rubric_token_logits.size(-1)),
        )
    else:
        gating_weights = gating_weights.float()

    if gating_type == "sigmoid":
        gate_denominator = gating_weights.sum(dim=-1).clamp_min(1e-6)
        return (rubric_token_logits * gating_weights).sum(dim=-1) / gate_denominator

    return (rubric_token_logits * gating_weights).sum(dim=-1)


def decode_code_keep_scores(
    model: SwePrunerForCodeCompression,
    outputs: SwePrunerOutput,
    doc_start: int,
    doc_end: int,
) -> Optional[torch.Tensor]:
    inner_model = getattr(model, "model", None)
    if inner_model is None or getattr(inner_model, "compression_head_type", None) != "crf":
        return None

    compression_emissions = getattr(outputs, "compression_emissions", None)
    fused_emissions = getattr(outputs, "fused_emissions", None)
    if compression_emissions is None:
        return None

    if getattr(inner_model, "num_objectives", 1) == 1:
        emissions = compression_emissions[:, doc_start:doc_end, 0, :]
        crf_layer = inner_model.compression_head.crf_layers[0]
    else:
        if (
            not getattr(inner_model, "use_final_crf", False)
            or getattr(inner_model, "final_crf", None) is None
            or fused_emissions is None
        ):
            return None
        emissions = fused_emissions[:, doc_start:doc_end, :]
        crf_layer = inner_model.final_crf

    if emissions.size(1) == 0:
        return None

    mask = torch.ones(
        emissions.size(0),
        emissions.size(1),
        dtype=torch.bool,
        device=emissions.device,
    )
    decoded = crf_layer._viterbi_decode(emissions, mask)
    return decoded.float()


def _indent_width(line: str) -> int:
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(" "))


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


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


def _containing_scopes(ast_graph: RuntimeAstGraph, line_no: int) -> List[ScopeBoundary]:
    scopes = [
        scope
        for scope in ast_graph.scope_boundaries
        if scope.kind != "module" and scope.start_line <= line_no <= scope.end_line
    ]
    return sorted(
        scopes,
        key=lambda scope: (scope.end_line - scope.start_line, scope.start_line),
    )


def _containing_controls(
    ast_graph: RuntimeAstGraph, line_no: int
) -> List[ControlBoundary]:
    controls = [
        ctrl
        for ctrl in ast_graph.control_boundaries
        if ctrl.start_line <= line_no <= ctrl.end_line
    ]
    return sorted(
        controls,
        key=lambda ctrl: (ctrl.end_line - ctrl.start_line, ctrl.start_line),
    )


def _suite_header_base_indent(
    ast_graph: Optional[RuntimeAstGraph],
    code_lines: List[str],
    header_end_line: int,
) -> Optional[str]:
    if ast_graph is None or not (1 <= header_end_line <= len(code_lines)):
        return None

    candidates: List[Tuple[int, int]] = []
    for scope in ast_graph.scope_boundaries:
        if scope.kind != "module" and scope.header_end_line == header_end_line:
            candidates.append(
                (scope.header_end_line - scope.header_start_line, scope.header_start_line)
            )
    for ctrl in ast_graph.control_boundaries:
        if ctrl.header_end_line == header_end_line:
            candidates.append(
                (ctrl.header_end_line - ctrl.header_start_line, ctrl.header_start_line)
            )

    if not candidates:
        return None

    _, header_start_line = min(candidates)
    return _leading_whitespace(code_lines[header_start_line - 1]) + "    "


def _definitions_by_line(
    ast_graph: RuntimeAstGraph,
) -> Dict[int, List[SymbolDefinition]]:
    out: Dict[int, List[SymbolDefinition]] = defaultdict(list)
    for definition in ast_graph.symbol_definitions:
        out[definition.line].append(definition)
    return out


def _outgoing_edges(ast_graph: RuntimeAstGraph):
    out = defaultdict(list)
    for edge in ast_graph.dependency_edges:
        out[edge.source_line].append(edge)
    return out


def _find_trailing_return_line(
    code_lines: List[str],
    scope: ScopeBoundary,
) -> Optional[int]:
    for line_no in range(scope.end_line, scope.header_end_line, -1):
        stripped = code_lines[line_no - 1].strip()
        if not stripped:
            continue
        if stripped == "return" or stripped.startswith("return "):
            return line_no
        break
    return None


def apply_ast_constraints(
    code: str,
    initial_kept_lines: List[int],
    ast_graph: RuntimeAstGraph,
    dependency_hops: int = 2,
) -> List[int]:
    code_lines = code.split("\n")
    line_count = len(code_lines)
    repaired = {
        int(line_no)
        for line_no in initial_kept_lines
        if isinstance(line_no, int) and 1 <= line_no <= line_count
    }
    if not repaired:
        return []

    definitions_by_line = _definitions_by_line(ast_graph)
    outgoing_edges = _outgoing_edges(ast_graph)

    def add_lines(line_numbers) -> List[int]:
        new_lines: List[int] = []
        for line_no in sorted(set(line_numbers)):
            if not (1 <= line_no <= line_count):
                continue
            if line_no in repaired:
                continue
            repaired.add(line_no)
            new_lines.append(line_no)
        return new_lines

    def apply_structural_closure(seed_lines) -> List[int]:
        queue = list(sorted(set(seed_lines)))
        seen = set()
        added_total: List[int] = []

        while queue:
            line_no = queue.pop(0)
            if line_no in seen or not (1 <= line_no <= line_count):
                continue
            seen.add(line_no)

            for scope in _containing_scopes(ast_graph, line_no):
                added = add_lines(
                    _line_range(
                        scope.header_start_line,
                        scope.header_end_line,
                        line_count,
                    )
                )
                if added:
                    queue.extend(added)
                    added_total.extend(added)

            for ctrl in _containing_controls(ast_graph, line_no):
                added = add_lines(
                    _line_range(
                        ctrl.header_start_line,
                        ctrl.header_end_line,
                        line_count,
                    )
                )
                if added:
                    queue.extend(added)
                    added_total.extend(added)

            branch_line = _find_special_branch_header(code_lines, line_no)
            if branch_line is not None:
                added = add_lines([branch_line])
                if added:
                    queue.extend(added)
                    added_total.extend(added)

        return added_total

    apply_structural_closure(repaired)

    frontier = set(repaired)
    for _ in range(max(0, dependency_hops)):
        next_frontier = set()
        for source_line in sorted(frontier):
            for edge in outgoing_edges.get(source_line, []):
                for definition in definitions_by_line.get(edge.target_line, []):
                    added = add_lines(
                        _line_range(
                            definition.header_start_line,
                            definition.header_end_line,
                            line_count,
                        )
                    )
                    next_frontier.update(added)
        if not next_frontier:
            break
        apply_structural_closure(next_frontier)
        frontier = set(next_frontier)

    changed = True
    while changed:
        changed = False

        string_additions = []
        for start_line, end_line in getattr(ast_graph, "string_spans", []):
            if end_line <= start_line:
                continue
            if any(start_line <= line_no <= end_line for line_no in repaired):
                string_additions.extend(range(start_line, end_line + 1))
        added = add_lines(string_additions)
        if added:
            apply_structural_closure(added)
            changed = True

        bracket_additions = []
        for open_line, close_line in ast_graph.bracket_pairs:
            if close_line <= open_line:
                continue
            if any(open_line <= line_no <= close_line for line_no in repaired):
                bracket_additions.extend([open_line, close_line])
        added = add_lines(bracket_additions)
        if added:
            apply_structural_closure(added)
            changed = True

        return_additions = []
        for scope in ast_graph.scope_boundaries:
            if scope.kind != "function":
                continue
            body_kept = any(
                scope.header_end_line < line_no <= scope.end_line for line_no in repaired
            )
            if not body_kept:
                continue
            return_lines = ast_graph.function_return_lines.get(scope.start_line, [])
            unique_returns = [
                line_no
                for line_no in sorted(set(return_lines))
                if scope.header_end_line < line_no <= scope.end_line
            ]
            if len(unique_returns) == 1:
                return_additions.extend(unique_returns)
                continue
            trailing_return = _find_trailing_return_line(code_lines, scope)
            if trailing_return is not None:
                return_additions.append(trailing_return)
        added = add_lines(return_additions)
        if added:
            apply_structural_closure(added)
            changed = True

    return sorted(repaired)


def merge_token_scores_from_chunks(
    code: str,
    chunk_results: List[
        Tuple[List[Tuple[str, float]], List[Tuple[int, int]], int, int]
    ],
) -> Tuple[List[Tuple[str, float]], List[Tuple[int, int]]]:
    """
    Merge token scores from multiple chunks, averaging overlapping tokens by position.

    Strategy: Group tokens by their (start_pos, end_pos) regardless of token_str,
    since the same position should have the same semantics globally.

    Args:
        code: Full original code string
        chunk_results: List of (token_scores, offsets, start_char, end_char) for each chunk

    Returns:
        Tuple of merged token_scores and their character offsets for the full code
    """

    if not chunk_results:
        return [], []

    # Map from (start_pos, end_pos) to list of scores
    # This groups tokens by position, which is more robust than grouping by position+content
    position_to_scores: Dict[Tuple[int, int], List[Tuple[str, float]]] = defaultdict(
        list
    )

    for token_scores, offsets, start_char, _ in chunk_results:
        max_pairs = min(len(token_scores), len(offsets))
        for idx in range(max_pairs):
            token_str, score = token_scores[idx]
            tok_start, tok_end = offsets[idx]
            abs_start = start_char + tok_start
            abs_end = start_char + tok_end

            # Skip tokens outside code bounds
            if abs_start >= len(code) or abs_end > len(code) or abs_start < 0:
                continue

            position_to_scores[(abs_start, abs_end)].append((token_str, float(score)))

    # Sort by position and build result
    sorted_positions = sorted(position_to_scores.keys())

    merged_token_scores = []
    merged_offsets: List[Tuple[int, int]] = []

    for abs_start, abs_end in sorted_positions:
        token_score_pairs = position_to_scores[(abs_start, abs_end)]

        # Average scores for this position
        avg_score = sum(score for _, score in token_score_pairs) / len(
            token_score_pairs
        )

        # Use the first token_str we saw for this position (they should all be the same)
        token_str = token_score_pairs[0][0]

        merged_token_scores.append((token_str, avg_score))
        merged_offsets.append((abs_start, abs_end))

    return merged_token_scores, merged_offsets


class SwePrunerForCodePruning(SwePrunerForCodeCompression):
    """
    Wrapper around SwePrunerForCodeCompression that provides a simplified prune interface.

    This class can be loaded with trust_remote_code=True from HuggingFace Hub.
    It hides implementation details and always uses line-level aggregation.
    """

    def __init__(self, config):
        # Pass from_pretrained=True if we're being loaded (config._name_or_path will be set)
        # This prevents redundant weight initialization
        from_pretrained = (
            hasattr(config, "_name_or_path") and config._name_or_path is not None
        )
        super().__init__(config, from_pretrained=from_pretrained)
        self.eval()
        # Don't move to device here - let from_pretrained handle it after loading weights
        # Device will be determined when first used, or can be set explicitly
        self._device = None

        # Note: self.tokenizer is already set by parent class (SwePrunerForCodeCompression)
        # Set padding side based on model type (matching online_serving.py)
        # LLM models use left padding, BERT-style models use right padding
        if self.model.is_llm:
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer.padding_side = "right"

        # Match online_serving.py process_single_chunk default instruction
        self.instruction = getattr(
            config,
            "instruction",
            "Given a query, judge if the document(code) is related to query.",
        )
        self.ast_dependency_hops = getattr(config, "runtime_dependency_hops", 2)
        self._ast_analyzer = RuntimePythonAstAnalyzer()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model from pretrained with proper device handling."""
        # The key is to prevent HuggingFace from using device_map or other optimizations
        # that create meta tensors
        kwargs.pop("device_map", None)
        kwargs.pop("low_cpu_mem_usage", None)  # Also disable this optimization

        # Now call parent's from_pretrained without these optimizations
        # This will load weights directly into memory
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        # Determine target device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target_dtype = getattr(model.config, "torch_dtype", None) or getattr(
            model.config, "dtype", None
        )
        if isinstance(target_dtype, str):
            target_dtype = getattr(torch, target_dtype, None)

        # Move to device (and cast if dtype is specified)
        if target_dtype is not None:
            model = model.to(device=device, dtype=target_dtype)
        else:
            model = model.to(device)

        model._device = device
        model.eval()
        return model

    def _ensure_device(self):
        """Ensure model is on the correct device."""
        if self._device is None:
            # Get current device and set it
            try:
                device = next(self.parameters()).device
                self._device = device
            except StopIteration:
                # No parameters, set to cpu as default
                self._device = torch.device("cpu")

    def _process_single_chunk(
        self,
        query: str,
        code_chunk: str,
        tokenizer: AutoTokenizer,
        max_length: int = 8192,
        instruction: Optional[str] = None,
    ) -> Tuple[float, List[Tuple[str, float]], List[Tuple[int, int]]]:
        """Process a single code chunk and return score and token scores."""
        # Ensure model is on the correct device
        self._ensure_device()

        # Build input
        inputs = build_input_for_llm(
            query, code_chunk, tokenizer, max_length=max_length, instruction=instruction
        )

        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        doc_start = inputs["doc_start"]
        doc_end = inputs["doc_end"]
        code_offsets = inputs["code_offsets"]

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
            ):
                outputs: SwePrunerOutput = self(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            token_logits = aggregate_objective_token_logits(
                outputs.token_logits,
                outputs.rubric_token_logits,
                outputs.gating_weights,
                getattr(self.model, "gating_type", "softmax"),
            )
            score_logits = outputs.score_logits.float()  # [1]
            decoded_keep_scores = decode_code_keep_scores(
                self,
                outputs,
                doc_start,
                doc_end,
            )

        # Get score
        score_prob = score_logits.squeeze(0).cpu()
        # Check if this is LLM style model - use is_llm attribute from TokenScorer
        # LLM models output log_softmax (log probabilities), non-LLM output raw logits
        if self.model.is_llm:
            # LLM style: score_logits are log probabilities from log_softmax
            chunk_score = float(torch.exp(score_prob).item())
        else:
            # Non-LLM style: score_logits are raw logits
            chunk_score = float(torch.sigmoid(score_prob).item())

        # Extract code token scores
        token_logits_seq = token_logits.squeeze(0).cpu()  # [L]
        probs = torch.sigmoid(token_logits_seq)  # [L]

        # Get code token positions
        code_token_ids = input_ids[0][doc_start:doc_end].cpu().tolist()
        code_token_scores = []
        decoded_keep_seq = None
        if decoded_keep_scores is not None:
            decoded_keep_seq = decoded_keep_scores.squeeze(0).cpu()

        for idx, pos in enumerate(range(doc_start, doc_end)):
            token_id = code_token_ids[idx]
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            if decoded_keep_seq is not None:
                score = float(decoded_keep_seq[idx].item())
            else:
                score = float(probs[pos].item())
            code_token_scores.append((token_str, score))

        return chunk_score, code_token_scores, code_offsets

    def prune(self, request: PruneRequest, max_length: int = 8192) -> PruneResponse:
        # Check if we need to split into chunks
        formatted_query = format_instruction(None, request.query)
        ast_graph = self._ast_analyzer.analyze(request.code)

        # Estimate total tokens needed
        query_tokens = estimate_token_count(formatted_query, self.tokenizer)
        code_tokens = estimate_token_count(request.code, self.tokenizer)

        # Calculate available length for code (accounting for prefix/suffix)
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        available_length = max_length - len(prefix_tokens) - len(suffix_tokens)
        code_max_tokens = available_length - query_tokens

        # Minimum tokens required to meaningfully process code
        # Match online_serving.py
        MIN_CODE_TOKENS = 100

        # If query is too long and leaves insufficient space for code, skip pruning
        if code_max_tokens < MIN_CODE_TOKENS:
            # Return original code without pruning
            return PruneResponse(
                score=0.0,  # Unknown score since we can't process
                pruned_code=request.code,
                token_scores=[],
                kept_frags=list(
                    range(1, len(request.code.splitlines()) + 1)
                ),  # All lines kept
                origin_token_cnt=code_tokens,
                left_token_cnt=code_tokens,
                model_input_token_cnt=0,
                error_msg=(
                    f"Query too long, insufficient space for code processing. "
                    f"Available code tokens: {code_max_tokens}, "
                    f"minimum required: {MIN_CODE_TOKENS}."
                ),
            )

        # Check if splitting is needed
        if code_tokens > code_max_tokens:
            # Split code into chunks with overlap
            overlap_tokens = request.chunk_overlap_tokens
            chunks = split_code_into_chunks(
                request.code,
                self.tokenizer,
                chunk_max_tokens=code_max_tokens,
                overlap_tokens=overlap_tokens,
            )

            # Process each chunk
            chunk_scores = []
            chunk_results = []

            for chunk_text, start_char, end_char in chunks:
                chunk_score, token_scores, offsets = self._process_single_chunk(
                    request.query,
                    chunk_text,
                    self.tokenizer,
                    max_length=max_length,
                    instruction=self.instruction,
                )
                chunk_scores.append(chunk_score)
                chunk_results.append((token_scores, offsets, start_char, end_char))

            # Average scores across chunks
            # Use max score instead of average to avoid diluting the relevance score
            # when code is split across chunks (each chunk is independent evaluation)
            predicted_score = max(chunk_scores) if chunk_scores else 0.0

            # Merge token scores from chunks (averaging overlaps)
            code_token_scores, code_token_offsets = merge_token_scores_from_chunks(
                request.code, chunk_results
            )
        else:
            # Single chunk processing
            predicted_score, code_token_scores, code_token_offsets = (
                self._process_single_chunk(
                    request.query,
                    request.code,
                    self.tokenizer,
                    max_length=max_length,
                    instruction=self.instruction,
                )
            )

        line_scores = aggregate_token_scores_to_lines(
            request.code,
            code_token_scores,
            code_token_offsets,
        )
        pruned_code, kept_frags = prune_code_lines(
            request.code,
            line_scores,
            request.threshold,
            request.always_keep_first_frags,
            request.always_keep_last_frags,
            ast_graph=ast_graph,
        )
        # Format token_scores for response
        token_scores_response = [[token, score] for token, score in code_token_scores]

        return PruneResponse(
            score=predicted_score,
            pruned_code=pruned_code,
            token_scores=token_scores_response,
            kept_frags=kept_frags,
            origin_token_cnt=code_tokens,
            left_token_cnt=estimate_token_count(pruned_code, self.tokenizer),
            model_input_token_cnt=query_tokens
            + code_tokens
            + len(prefix_tokens)
            + len(suffix_tokens),
        )
