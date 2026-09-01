"""
Judge repaired masks for SWE-Pruner v2.

This module supports two modes:
  - heuristic: offline structural acceptance for smoke tests and CI
  - llm: vLLM-based LLM-as-a-judge for the full paper-aligned filtering step
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field
import typer
from rich.console import Console
from tqdm import tqdm

from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.mask_repair import load_processed_codes, normalize_kept_frags, repair_mask


app = typer.Typer(help="Filter repaired masks with heuristic or LLM judging")
console = Console()


class JudgeEvaluation(BaseModel):
    semantic_preservation: str = Field(..., pattern="^(pass|fail)$")
    syntax_integrity: str = Field(..., pattern="^(pass|fail)$")
    dependency_completeness: str = Field(..., pattern="^(pass|fail)$")
    context_sufficiency: str = Field(..., pattern="^(pass|fail)$")
    redundancy: str = Field(..., pattern="^(acceptable|excessive)$")
    overall_quality: str = Field(..., pattern="^(high|medium|low)$")
    accepted: bool
    reasoning: str = ""


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _brace_extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : idx + 1])
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found in judge output")


def parse_judge_output(text: str) -> JudgeEvaluation:
    payload = _brace_extract_json(text)
    return JudgeEvaluation(**payload)


def _render_numbered_code(
    code: str,
    original_kept_frags: List[int],
    repaired_kept_frags: List[int],
) -> str:
    original_set = set(original_kept_frags)
    repaired_set = set(repaired_kept_frags)
    rendered_lines = []
    for idx, line in enumerate(code.split("\n"), start=1):
        markers: List[str] = []
        if idx in original_set:
            markers.append("O")
        if idx in repaired_set:
            markers.append("R")
        marker_text = "".join(markers) or "-"
        rendered_lines.append(f"{idx:>4} [{marker_text:<2}] {line}")
    return "\n".join(rendered_lines)


def build_judge_prompt(
    query: str,
    code: str,
    original_kept_frags: List[int],
    repaired_kept_frags: List[int],
) -> str:
    numbered_code = _render_numbered_code(code, original_kept_frags, repaired_kept_frags)
    return f"""
You are validating an AST-aware repair of line-retention labels for code-context pruning.

Assess whether the repaired mask preserves the semantic usefulness of the original kept lines
while adding only the structural context needed for syntax, dependency closure, and local context.

Return JSON only with this schema:
{{
  "semantic_preservation": "pass|fail",
  "syntax_integrity": "pass|fail",
  "dependency_completeness": "pass|fail",
  "context_sufficiency": "pass|fail",
  "redundancy": "acceptable|excessive",
  "overall_quality": "high|medium|low",
  "accepted": true,
  "reasoning": "short explanation"
}}

Query:
{query}

Original semantic lines:
{original_kept_frags}

Repaired lines:
{repaired_kept_frags}

Numbered code with markers:
- O means original semantic line
- R means repaired line

{numbered_code}
""".strip()


def heuristic_judge_repair(
    query: str,
    code: str,
    original_kept_frags: List[int],
    repaired_kept_frags: List[int],
    analysis,
    dependency_hops: int = 2,
    max_expansion_factor: float = 4.0,
    max_extra_lines: int = 40,
) -> JudgeEvaluation:
    del query  # structural heuristic does not currently inspect natural language

    line_count = len(code.split("\n"))
    original = normalize_kept_frags(original_kept_frags, line_count)
    repaired = normalize_kept_frags(repaired_kept_frags, line_count)

    closure = repair_mask(
        code=code,
        kept_frags=repaired,
        analysis=analysis,
        dependency_hops=dependency_hops,
    )
    closure_actions = closure["repair_actions"]
    missing_repair_lines = sorted(set(closure["repaired_kept_frags"]) - set(repaired))

    syntax_missing = [
        action
        for action in closure_actions
        if action["line"] in missing_repair_lines
        and (
            action["reason"].startswith("enclosing_")
            or action["reason"] == "branch_companion_header"
        )
    ]
    dependency_missing = [
        action
        for action in closure_actions
        if action["line"] in missing_repair_lines
        and action["reason"].startswith("dependency_")
    ]

    semantic_preservation = "pass" if set(original).issubset(repaired) else "fail"
    syntax_integrity = "pass" if analysis.parse_ok and not syntax_missing else "fail"
    dependency_completeness = "pass" if not dependency_missing else "fail"
    context_sufficiency = (
        "pass"
        if semantic_preservation == "pass" and syntax_integrity == "pass"
        else "fail"
    )

    max_allowed_lines = max(
        len(original) + max_extra_lines,
        int(math.ceil(max(1, len(original)) * max_expansion_factor)),
    )
    redundancy = "acceptable" if len(repaired) <= max_allowed_lines else "excessive"

    accepted = all(
        [
            semantic_preservation == "pass",
            syntax_integrity == "pass",
            dependency_completeness == "pass",
            context_sufficiency == "pass",
            redundancy == "acceptable",
        ]
    )

    if accepted:
        overall_quality = "high"
    elif (
        semantic_preservation == "pass"
        and syntax_integrity == "pass"
        and dependency_completeness == "pass"
    ):
        overall_quality = "medium"
    else:
        overall_quality = "low"

    reasons: List[str] = []
    if semantic_preservation == "fail":
        reasons.append("repaired mask dropped original semantic lines")
    if syntax_missing:
        reasons.append(
            f"missing structural headers on lines {[action['line'] for action in syntax_missing[:5]]}"
        )
    if dependency_missing:
        reasons.append(
            f"missing dependency lines {[action['line'] for action in dependency_missing[:5]]}"
        )
    if redundancy == "excessive":
        reasons.append(
            f"repair expanded from {len(original)} to {len(repaired)} lines"
        )
    if not reasons:
        reasons.append("repaired mask is structurally closed and within expansion budget")

    return JudgeEvaluation(
        semantic_preservation=semantic_preservation,
        syntax_integrity=syntax_integrity,
        dependency_completeness=dependency_completeness,
        context_sufficiency=context_sufficiency,
        redundancy=redundancy,
        overall_quality=overall_quality,
        accepted=accepted,
        reasoning="; ".join(reasons),
    )


class VllmRepairJudge:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        try:
            import torch
            from vllm import LLM, SamplingParams
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "LLM judge mode requires vllm and torch to be installed"
            ) from exc

        tp = max(1, min(tensor_parallel_size, torch.cuda.device_count() or 1))
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def judge(
        self,
        query: str,
        code: str,
        original_kept_frags: List[int],
        repaired_kept_frags: List[int],
    ) -> JudgeEvaluation:
        prompt = build_judge_prompt(
            query=query,
            code=code,
            original_kept_frags=original_kept_frags,
            repaired_kept_frags=repaired_kept_frags,
        )
        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a strict code-dataset quality judge. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
        ]
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
        return parse_judge_output(outputs[0].outputs[0].text)


def judge_repair(
    query: str,
    code: str,
    original_kept_frags: List[int],
    repaired_kept_frags: List[int],
    analysis,
    judge_mode: str = "heuristic",
    llm_judge: Optional[VllmRepairJudge] = None,
    dependency_hops: int = 2,
    max_expansion_factor: float = 4.0,
    max_extra_lines: int = 40,
) -> JudgeEvaluation:
    if judge_mode == "heuristic":
        return heuristic_judge_repair(
            query=query,
            code=code,
            original_kept_frags=original_kept_frags,
            repaired_kept_frags=repaired_kept_frags,
            analysis=analysis,
            dependency_hops=dependency_hops,
            max_expansion_factor=max_expansion_factor,
            max_extra_lines=max_extra_lines,
        )
    if judge_mode != "llm":
        raise ValueError(f"Unsupported judge_mode={judge_mode!r}")
    if llm_judge is None:
        raise ValueError("judge_mode='llm' requires an initialized llm_judge")
    return llm_judge.judge(
        query=query,
        code=code,
        original_kept_frags=original_kept_frags,
        repaired_kept_frags=repaired_kept_frags,
    )


@app.command()
def main(
    input_file: Path = typer.Option(..., "-i", "--input-file", help="Input JSONL with repaired masks"),
    output_jsonl: Path = typer.Option(..., "-o", "--output-jsonl", help="Output JSONL with judge results"),
    judge_mode: str = typer.Option("heuristic", "--judge-mode", help="heuristic or llm"),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="vLLM judge model path"),
    tensor_parallel_size: int = typer.Option(1, "--tensor-parallel-size"),
    max_model_len: int = typer.Option(16384, "--max-model-len"),
    dependency_hops: int = typer.Option(2, "--dependency-hops"),
    max_expansion_factor: float = typer.Option(4.0, "--max-expansion-factor"),
    max_extra_lines: int = typer.Option(40, "--max-extra-lines"),
    require_tree_sitter: bool = typer.Option(False, "--require-tree-sitter"),
    keep_rejected: bool = typer.Option(False, "--keep-rejected"),
    max_items: Optional[int] = typer.Option(None, "--max-items"),
) -> None:
    analyzer = PythonAstAnalyzer(require_tree_sitter=require_tree_sitter)
    llm_judge = None
    if judge_mode == "llm":
        if not model_name:
            raise typer.BadParameter("--model-name is required when --judge-mode llm")
        llm_judge = VllmRepairJudge(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

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
        for line in tqdm(f_in, desc="Judging repairs"):
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

            original_kept = item.get("original_kept_frags", item.get("kept_frags", []))
            repaired_kept = item.get("repaired_kept_frags", item.get("kept_frags", []))

            try:
                analysis = analyzer.analyze(code)
                evaluation = judge_repair(
                    query=item.get("query", ""),
                    code=code,
                    original_kept_frags=original_kept,
                    repaired_kept_frags=repaired_kept,
                    analysis=analysis,
                    judge_mode=judge_mode,
                    llm_judge=llm_judge,
                    dependency_hops=dependency_hops,
                    max_expansion_factor=max_expansion_factor,
                    max_extra_lines=max_extra_lines,
                )
            except Exception as exc:
                console.print(f"[yellow]Skipping row {read}: {exc}[/yellow]")
                errors += 1
                continue

            output_item = {
                **item,
                "judge_evaluation": _model_dump(evaluation),
                "accepted": evaluation.accepted,
            }
            if evaluation.accepted or keep_rejected:
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                processed_codes.add(code)
                written += 1

    console.rule("Judge filtering complete")
    console.print(f"Read: {read}")
    console.print(f"Written: {written}")
    console.print(f"Skipped: {skipped}")
    console.print(f"Errors: {errors}")
    console.print(f"Output: [bold]{output_jsonl}[/bold]")


if __name__ == "__main__":
    app()
