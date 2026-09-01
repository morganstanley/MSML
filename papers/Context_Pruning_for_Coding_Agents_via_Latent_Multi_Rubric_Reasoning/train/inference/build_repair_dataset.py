"""
Build the corrected SWE-Pruner v2 dataset:

semantic source of truth:
  original kept_frags

structural repair:
  AST-aware mask closure

quality filter:
  heuristic or LLM judge
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from tqdm import tqdm

from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.judge_filter import VllmRepairJudge, _model_dump, judge_repair
from train.inference.mask_repair import repair_item
from train.inference.row_identity import build_row_identity, load_processed_row_ids


app = typer.Typer(help="Build the repaired-and-judged SWE-Pruner v2 dataset")
console = Console()


@app.command()
def main(
    input_file: Path = typer.Option(
        "swe-pruner-training-dataset-py.jsonl",
        "-i",
        "--input-file",
        help="Input v1 JSONL with query, code, score, kept_frags",
    ),
    output_jsonl: Path = typer.Option(
        "swe-pruner-training-dataset-py-v2.jsonl",
        "-o",
        "--output-jsonl",
        help="Output repaired-and-judged JSONL",
    ),
    dependency_hops: int = typer.Option(2, "--dependency-hops"),
    judge_mode: str = typer.Option(
        "heuristic",
        "--judge-mode",
        help="heuristic or llm",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="vLLM judge model path when --judge-mode llm",
    ),
    tensor_parallel_size: int = typer.Option(1, "--tensor-parallel-size"),
    max_model_len: int = typer.Option(16384, "--max-model-len"),
    require_tree_sitter: bool = typer.Option(False, "--require-tree-sitter"),
    include_ast_metadata: bool = typer.Option(
        True,
        "--include-ast-metadata/--no-include-ast-metadata",
    ),
    max_expansion_factor: float = typer.Option(4.0, "--max-expansion-factor"),
    max_extra_lines: int = typer.Option(40, "--max-extra-lines"),
    keep_rejected: bool = typer.Option(
        False,
        "--keep-rejected",
        help="Write rejected rows too for auditing",
    ),
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
    processed_row_ids = load_processed_row_ids(output_jsonl)

    read = 0
    written = 0
    skipped = 0
    errors = 0
    accepted_count = 0
    rejected_count = 0

    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_jsonl, "a", encoding="utf-8") as f_out,
    ):
        for line in tqdm(f_in, desc="Building repaired dataset"):
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
            row_id = build_row_identity(item)
            if row_id in processed_row_ids:
                skipped += 1
                continue

            try:
                repaired = repair_item(
                    item=item,
                    analyzer=analyzer,
                    dependency_hops=dependency_hops,
                    include_ast_metadata=include_ast_metadata,
                    require_tree_sitter=require_tree_sitter,
                )
                analysis = analyzer.analyze(code)
                evaluation = judge_repair(
                    query=repaired.get("query", ""),
                    code=code,
                    original_kept_frags=repaired["original_kept_frags"],
                    repaired_kept_frags=repaired["repaired_kept_frags"],
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

            accepted = evaluation.accepted
            if accepted:
                accepted_count += 1
                final_kept_frags = repaired["repaired_kept_frags"]
            else:
                rejected_count += 1
                final_kept_frags = repaired["original_kept_frags"]

            output_item: Dict[str, Any] = {
                **repaired,
                "final_kept_frags": final_kept_frags,
                # Keep current trainer compatibility: train.py still reads `kept_frags`.
                "kept_frags": final_kept_frags,
                "judge_evaluation": _model_dump(evaluation),
                "accepted": accepted,
            }

            if accepted or keep_rejected:
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                processed_row_ids.add(row_id)
                written += 1

    console.rule("Repaired dataset build complete")
    console.print(f"Read: {read}")
    console.print(f"Written: {written}")
    console.print(f"Skipped: {skipped}")
    console.print(f"Errors: {errors}")
    console.print(f"Accepted: {accepted_count}")
    console.print(f"Rejected: {rejected_count}")
    console.print(f"Output: [bold]{output_jsonl}[/bold]")


if __name__ == "__main__":
    app()
