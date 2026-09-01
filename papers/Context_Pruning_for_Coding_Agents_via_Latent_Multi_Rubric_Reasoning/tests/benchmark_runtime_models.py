import argparse
import ast
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "swe-pruner" / "src"))

from swe_pruner.prune_wrapper import PruneRequest, SwePrunerForCodePruning


def parse_model_arg(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Model spec must be name=path, got: {raw!r}"
        )
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"Model spec must be name=path, got: {raw!r}"
        )
    return name, path


def load_rows(
    dataset_path: Path,
    sample_size: int,
    seed: int,
    max_code_chars: int,
    parseable_only: bool,
) -> List[dict]:
    accepted_rows: List[dict] = []
    rejected_rows: List[dict] = []
    fallback_rows: List[dict] = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            code = row.get("code")
            query = row.get("query")
            if not isinstance(code, str) or not isinstance(query, str):
                continue
            if not code.strip() or len(code) > max_code_chars:
                continue
            if parseable_only and not is_python_parseable(code):
                continue
            row["_row_idx"] = idx
            fallback_rows.append(row)
            accepted = row.get("accepted")
            if accepted is True:
                accepted_rows.append(row)
            elif accepted is False:
                rejected_rows.append(row)

    rng = random.Random(seed)
    if accepted_rows and rejected_rows and sample_size >= 2:
        half = sample_size // 2
        accepted_pick = min(len(accepted_rows), half + (sample_size % 2))
        rejected_pick = min(len(rejected_rows), half)
        selected = rng.sample(accepted_rows, accepted_pick)
        selected.extend(rng.sample(rejected_rows, rejected_pick))
        if len(selected) < sample_size:
            used = {row["_row_idx"] for row in selected}
            remaining = [row for row in fallback_rows if row["_row_idx"] not in used]
            if remaining:
                selected.extend(
                    rng.sample(remaining, min(sample_size - len(selected), len(remaining)))
                )
    else:
        selected = rng.sample(fallback_rows, min(sample_size, len(fallback_rows)))

    selected.sort(key=lambda row: row["_row_idx"])
    return selected


def is_python_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def percentile_ms(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_model(
    model_path: str,
    rows: List[dict],
    threshold: float,
    failure_examples: int = 0,
) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwePrunerForCodePruning.from_pretrained(model_path)
    latencies_ms: List[float] = []
    syntax_ok = 0
    input_parseable = 0
    syntax_ok_given_parseable = 0
    kept_line_counts: List[int] = []
    token_retention_rates: List[float] = []
    score_values: List[float] = []
    failures: List[Dict[str, object]] = []

    warmup_rows = rows[: min(2, len(rows))]
    with torch.inference_mode():
        for row in warmup_rows:
            _ = model.prune(
                PruneRequest(
                    query=row["query"],
                    code=row["code"],
                    threshold=threshold,
                )
            )
        if device == "cuda":
            torch.cuda.synchronize()

        for row in rows:
            request = PruneRequest(
                query=row["query"],
                code=row["code"],
                threshold=threshold,
            )
            row_input_parseable = is_python_parseable(row["code"])
            input_parseable += int(row_input_parseable)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            response = model.prune(request)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            row_output_parseable = is_python_parseable(response.pruned_code)
            syntax_ok += int(row_output_parseable)
            if row_input_parseable:
                syntax_ok_given_parseable += int(row_output_parseable)
                if (not row_output_parseable) and len(failures) < failure_examples:
                    failures.append(
                        {
                            "row_idx": row.get("_row_idx"),
                            "query": row["query"],
                            "code": row["code"],
                            "pruned_code": response.pruned_code,
                            "kept_frags": response.kept_frags,
                        }
                    )
            kept_line_counts.append(len(response.kept_frags))
            score_values.append(float(response.score))
            if response.origin_token_cnt > 0:
                token_retention_rates.append(
                    float(response.left_token_cnt) / float(response.origin_token_cnt)
                )

    result = {
        "requests": len(rows),
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_p95_ms": percentile_ms(latencies_ms, 0.95),
        "input_parseable_rate": input_parseable / float(len(rows) or 1),
        "syntax_valid_rate": syntax_ok / float(len(rows) or 1),
        "syntax_valid_given_input_parseable_rate": (
            syntax_ok_given_parseable / float(input_parseable or 1)
        ),
        "kept_lines_mean": statistics.mean(kept_line_counts) if kept_line_counts else 0.0,
        "token_retention_mean": (
            statistics.mean(token_retention_rates) if token_retention_rates else 0.0
        ),
        "score_mean": statistics.mean(score_values) if score_values else 0.0,
    }
    if failures:
        result["failure_examples"] = failures

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SWE-Pruner runtime models.")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL to sample requests from")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model spec: name=path",
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-code-chars", type=int, default=6000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--failure-examples", type=int, default=0)
    parser.add_argument(
        "--parseable-only",
        action="store_true",
        help="Restrict the sample to inputs that parse as standalone Python.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_rows(
        dataset_path=dataset_path,
        sample_size=args.sample_size,
        seed=args.seed,
        max_code_chars=args.max_code_chars,
        parseable_only=args.parseable_only,
    )
    if not rows:
        raise SystemExit("No usable rows selected for benchmark.")

    accepted_count = sum(1 for row in rows if row.get("accepted") is True)
    rejected_count = sum(1 for row in rows if row.get("accepted") is False)

    results = {
        "dataset": str(dataset_path),
        "sample_size": len(rows),
        "seed": args.seed,
        "max_code_chars": args.max_code_chars,
        "threshold": args.threshold,
        "parseable_only": args.parseable_only,
        "sample_breakdown": {
            "accepted": accepted_count,
            "rejected": rejected_count,
            "other": len(rows) - accepted_count - rejected_count,
            "row_indices": [row["_row_idx"] for row in rows],
        },
        "models": {},
    }

    for raw in args.models:
        name, path = parse_model_arg(raw)
        results["models"][name] = benchmark_model(
            path,
            rows,
            args.threshold,
            failure_examples=args.failure_examples,
        )

    payload = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
