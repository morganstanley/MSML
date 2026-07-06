"""Configuration-driven Phase 0 domain-knowledge evaluation.

Combines LLM-as-judge rubrics (per-criterion + semantic similarity) with
mechanical metrics from JSONL logs into a single composite score suitable
as an optimization target for DSPy/GEPA-style prompt tuning.

Entry point: ``alpha-lab-evaluate`` (registered in pyproject.toml).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from alpha_lab.adapter import DomainAdapter
from alpha_lab.adapter_loader import (
    load_adapter,
    resolve_reference_adapter,
)
from alpha_lab.client import get_pydantic_ai_model

logger = logging.getLogger("alpha_lab.evaluations")

RESERVED_YAML_KEYS = frozenset(
    {"semantic_similarity_rubric", "_template_source"}
)
DEFAULT_FILE_WEIGHTS = {"criteria": 0.0, "similarity": 1.0}
DEFAULT_SECTION_WEIGHTS = {"files": 0.8, "mechanical": 0.2}


# ---------------------------------------------------------------------------
# Step 3 — YAML config loader
# ---------------------------------------------------------------------------

def load_eval_config(config_path: Path) -> dict[str, Any]:
    """Read and validate evaluation YAML config.

    Raises ``ValueError`` when required keys are missing.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping, got {type(raw).__name__}")

    # A valid eval section must satisfy all four conditions:
    #   1. Not a reserved key (e.g. "semantic_similarity_rubric", "_template_source")
    #   2. Not underscore-prefixed (used for YAML anchors/shared defaults)
    #   3. A mapping (not a scalar) — filters out orphan rubric strings such as
    #      ``semantic_similarity_rubric_1: "..."`` that would otherwise slip through
    #   4. Contains a "judge" sub-key — the minimum contract for a runnable section
    eval_sections = {
        k: v for k, v in raw.items()
        if k not in RESERVED_YAML_KEYS
        and not k.startswith("_")
        and isinstance(v, dict)
        and "judge" in v
    }
    if not eval_sections:
        raise ValueError("YAML must contain at least one evaluation section")

    return raw


# ---------------------------------------------------------------------------
# Step 4 — Domain resolution
# ---------------------------------------------------------------------------

def resolve_eval_domain(
    workspace: Path,
) -> tuple[str, DomainAdapter, DomainAdapter]:
    """Resolve domain name, workspace adapter, and reference adapter.

    Returns ``(domain_name, workspace_adapter, reference_adapter)``.
    Falls back to ``time_series`` reference when the domain is not a built-in.
    """
    adapter_dir = workspace / "adapter"
    manifest_path = adapter_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {adapter_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    domain_name: str = manifest["domain_name"]

    workspace_adapter = load_adapter(adapter_dir)
    reference_adapter = resolve_reference_adapter(domain_name)

    return domain_name, workspace_adapter, reference_adapter


# ---------------------------------------------------------------------------
# Template variable loading
# ---------------------------------------------------------------------------

def load_template_variables(
    workspace: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load template variables from a JSON file in the workspace.

    Reads the file at ``workspace / config["_template_source"]`` and
    returns the full parsed dict.  Rubric authors use Jinja2 dot-access
    to reach nested values (e.g. ``{{ generator.n_features }}``).

    Returns an empty dict when ``_template_source`` is not in *config*.

    Raises:
        FileNotFoundError: if the source file does not exist.
    """
    source = config.get("_template_source")
    if source is None:
        return {}

    source_path = workspace / source
    with open(source_path) as f:
        return json.load(f)


def render_criteria(
    criteria: dict[str, str],
    variables: dict[str, Any],
) -> dict[str, str]:
    """Apply template variables to criteria rubric strings.

    Uses Jinja2 with ``StrictUndefined`` so that missing variables
    raise ``ValueError`` rather than rendering silently.
    """
    env = Environment(undefined=StrictUndefined)
    rendered: dict[str, str] = {}
    for name, rubric in criteria.items():
        try:
            rendered[name] = env.from_string(rubric).render(variables)
        except UndefinedError as exc:
            raise ValueError(
                f"Criteria '{name}': {exc}"
            ) from exc
    return rendered


def resolve_file_content(
    workspace_adapter: DomainAdapter,
    reference_adapter: DomainAdapter,
    file_key: str,
) -> tuple[str, str]:
    """Return ``(candidate, reference)`` content for an adapter file key.

    *file_key* is ``domain_knowledge`` or a prompt key (e.g. ``phase1``).
    """
    if file_key == "domain_knowledge":
        return workspace_adapter.domain_knowledge, reference_adapter.domain_knowledge
    return workspace_adapter.prompts[file_key], reference_adapter.prompts[file_key]


# ---------------------------------------------------------------------------
# Step 5 — LLM judge factory
# ---------------------------------------------------------------------------

def _rubric_hash(rubric: str) -> str:
    return hashlib.md5(rubric.encode()).hexdigest()[:8]


def _build_similarity_judge(model: Any, sim_rubric: str) -> LLMJudge:
    """Build the shared semantic similarity judge (dataset-level)."""
    return LLMJudge(
        model=model,
        rubric=sim_rubric,
        include_input=False,
        include_expected_output=True,
        score={
            "evaluation_name": f"semantic_similarity [{_rubric_hash(sim_rubric)}]",
            "include_reason": True,
        },
        assertion={"include_reason": True},
    )


def build_criteria_judges(
    model: Any,
    file_config: dict[str, Any],
) -> list[LLMJudge]:
    """Build per-file criteria LLMJudge instances (case-level)."""
    judges: list[LLMJudge] = []
    # NOTE: The presence of the `assertion` named argument is what drives the
    # `Assertions` column in the final output report.  We could drop the named argument
    # here for a slightly cleaner report but it would increase the complexity of the
    # code that judges pass/fail.
    for name, rubric in file_config.get("criteria", {}).items():
        judges.append(
            LLMJudge(
                model=model,
                rubric=rubric,
                include_input=False,
                include_expected_output=False,
                score={
                    "evaluation_name": f"{name} [{_rubric_hash(rubric)}]",
                    "include_reason": True,
                },
                assertion={"include_reason": True},
            )
        )
    return judges


# ---------------------------------------------------------------------------
# Step 6 — JSONL metrics parser
# ---------------------------------------------------------------------------

def parse_phase0_metrics(log_path: Path) -> dict[str, float]:
    """Extract mechanical metrics from a Phase 0 JSONL log (path-agnostic)."""
    input_tokens = 0
    output_tokens = 0
    tool_calls = 0
    first_ts: float | None = None
    last_ts: float | None = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)

            ts = event.get("timestamp")
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            event_type = event.get("type", "")
            if event_type == "api_response":
                usage = event.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
            elif event_type == "tool_call":
                tool_calls += 1

    duration = 0.0
    if first_ts is not None and last_ts is not None:
        duration = last_ts - first_ts

    return {
        "input_tokens": float(input_tokens),
        "output_tokens": float(output_tokens),
        "tool_calls": float(tool_calls),
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------------
# Step 7 — Mechanical metrics evaluator
# ---------------------------------------------------------------------------

class ThresholdEvaluator(Evaluator):
    """Check a metric value against floor/ceiling thresholds.

    Reports the raw value and a band label (``below_floor``, ``ok``,
    ``above_ceiling``).  The proportional in-band count feeds the
    composite score.
    """

    def __init__(self, metric_name: str, value: float, floor: float, ceiling: float) -> None:
        self.metric_name = metric_name
        self.value = value
        self.floor = floor
        self.ceiling = ceiling

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float]:
        if self.value < self.floor:
            band = "below_floor"
        elif self.value > self.ceiling:
            band = "above_ceiling"
        else:
            band = "ok"

        logger.debug(
            "threshold_%s: value=%.1f band=%s [floor=%.1f, ceiling=%.1f]",
            self.metric_name, self.value, band, self.floor, self.ceiling,
        )
        # Return raw value under a stable metric-specific name.
        # Band is encoded separately in the key suffix so the table stays readable.
        return {f"threshold_{self.metric_name} ({band})": self.value}


def build_threshold_evaluators(
    metrics: dict[str, float],
    thresholds: dict[str, dict[str, float]],
) -> tuple[list[ThresholdEvaluator], float]:
    """Build ThresholdEvaluator instances and compute in-band ratio.

    Returns ``(evaluators, mechanical_penalty)`` where penalty is
    ``in_band_count / total_metrics`` (0.0-1.0).
    """
    evaluators: list[ThresholdEvaluator] = []
    in_band = 0
    total = 0

    for metric_name, bounds in thresholds.items():
        value = metrics.get(metric_name, 0.0)
        floor = bounds.get("floor", 0.0)
        ceiling = bounds.get("ceiling", float("inf"))
        evaluators.append(
            ThresholdEvaluator(
                metric_name=metric_name,
                value=value,
                floor=floor,
                ceiling=ceiling,
            )
        )
        total += 1
        if floor <= value <= ceiling:
            in_band += 1

    penalty = in_band / total if total > 0 else 1.0
    return evaluators, penalty


# ---------------------------------------------------------------------------
# Step 8 — Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset(
    eval_name: str,
    cases: list[Case],
    similarity_judge: LLMJudge | None,
) -> Dataset:
    """Assemble a single pydantic_evals Dataset for all adapter files.

    The *similarity_judge*, when provided, is applied at the dataset level
    (runs on every case).  Per-file criteria judges are attached to
    individual ``Case`` objects via their ``evaluators`` field.
    """
    evaluators = [similarity_judge] if similarity_judge is not None else []
    return Dataset(
        name=eval_name,
        cases=cases,
        evaluators=evaluators,
    )


# ---------------------------------------------------------------------------
# Step 9 — Async main
# ---------------------------------------------------------------------------

def _extract_file_composite(
    case_result: Any,
    criteria_count: int,
    weights: dict[str, float],
) -> float:
    """Compute per-file composite from a single ``ReportCase``.

    ``file_composite = w_criteria * criteria_pass_rate
                     + w_similarity * normalized_similarity``
    """
    criteria_passed = 0
    similarity_score = 0.0

    for eval_name, result in case_result.assertions.items():
        if "semantic_similarity" in eval_name:
            continue
        if result.value:
            criteria_passed += 1

    for eval_name, score_result in case_result.scores.items():
        if "semantic_similarity" in eval_name:
            raw = score_result.value if hasattr(score_result, "value") else score_result
            if raw is not None:
                similarity_score = float(raw)

    criteria_pass_rate = criteria_passed / criteria_count if criteria_count > 0 else 0.0
    normalized_similarity = similarity_score / 5.0

    w_criteria = weights.get("criteria", DEFAULT_FILE_WEIGHTS["criteria"])
    w_similarity = weights.get("similarity", DEFAULT_FILE_WEIGHTS["similarity"])

    composite = w_criteria * criteria_pass_rate + w_similarity * normalized_similarity

    logger.info(
        "  file composite: %.4f  (criteria=%.2f * %.2f, similarity=%.2f * %.2f)",
        composite,
        criteria_pass_rate, w_criteria,
        normalized_similarity, w_similarity,
    )
    return composite


async def async_main(args: argparse.Namespace) -> float:
    """Run the full evaluation pipeline. Returns section-level composite."""
    config = load_eval_config(Path(args.eval_config))
    eval_name: str = args.eval_name
    workspace = Path(args.workspace)

    if args.run_phase0:
        _run_phase0_before_eval(args, workspace)

    domain_name, ws_adapter, ref_adapter = resolve_eval_domain(workspace)
    logger.info("Evaluating domain '%s' with eval section '%s'", domain_name, eval_name)

    section = config[eval_name]
    judge_cfg = section["judge"]
    model = get_pydantic_ai_model(judge_cfg["provider"], judge_cfg["model"])
    sim_rubric: str | None = config.get("semantic_similarity_rubric")
    files_config = section.get("files", {})
    template_vars = load_template_variables(workspace, config)

    # --- Mechanical metrics (separate from file evaluation) ---
    # Phase 0 writes phase0_customize.jsonl for built-in adapter customization
    # and phase0.jsonl for novel-domain generation. Check both; use the first found.
    _logs_dir = workspace / "logs"
    _phase0_log_names = ("phase0_customize.jsonl", "phase0.jsonl")
    log_path = next(
        (_logs_dir / name for name in _phase0_log_names if (_logs_dir / name).exists()),
        None,
    )
    mechanical_penalty = 1.0
    if log_path is not None and "mechanical_metrics" in section:
        metrics = parse_phase0_metrics(log_path)
        threshold_evaluators, mechanical_penalty = build_threshold_evaluators(
            metrics, section["mechanical_metrics"],
        )
        print(f"\nMechanical metrics (penalty={mechanical_penalty:.2f}):")
        for ev in threshold_evaluators:
            if ev.value < ev.floor:
                band = "below_floor"
            elif ev.value > ev.ceiling:
                band = "above_ceiling"
            else:
                band = "ok"
            print(
                f"  {ev.metric_name}: {ev.value:.1f}  "
                f"[floor={ev.floor:.1f}, ceiling={ev.ceiling:.1f}]  ({band})"
            )
        logger.info("Mechanical metrics: %s  penalty=%.2f", metrics, mechanical_penalty)
    elif log_path is None:
        logger.warning(
            "No phase0 JSONL log found in %s — skipping mechanical metrics", _logs_dir
        )

    # --- Build cases (one per adapter file) ---
    cases: list[Case] = []
    # Track per-file metadata for composite extraction after evaluation.
    file_meta: dict[str, dict[str, Any]] = {}

    for file_key, file_cfg in files_config.items():
        if file_cfg is None:
            file_cfg = {}

        sut_content, reference_content = resolve_file_content(
            ws_adapter, ref_adapter, file_key,
        )

        if template_vars and "criteria" in file_cfg:
            file_cfg = {
                **file_cfg,
                "criteria": render_criteria(file_cfg["criteria"], template_vars),
            }

        criteria_judges = build_criteria_judges(model, file_cfg)
        case_name = f"{eval_name}/{file_key}"

        # Case.evaluators runs these judges only on this case, in addition
        # to the dataset-level similarity judge shared across all cases.
        cases.append(Case(
            name=case_name,
            inputs=sut_content,
            expected_output=reference_content,
            evaluators=tuple(criteria_judges),
        ))

        file_meta[case_name] = {
            "criteria_count": len(file_cfg.get("criteria", {})),
            "weights": file_cfg.get("weights", DEFAULT_FILE_WEIGHTS),
        }

    # --- Evaluate all files in a single Dataset ---
    similarity_judge = (
        _build_similarity_judge(model, sim_rubric) if sim_rubric is not None else None
    )
    dataset = build_dataset(eval_name, cases, similarity_judge)

    async def passthrough(input_text: str) -> str:
        return input_text

    report = await dataset.evaluate(passthrough)
    if args.show_table:
        report.print(
            include_input=False,
            include_output=False,
            include_durations=False, # This would measure our `passthrough` function above
            include_reasons=True,
            include_averages=False, # With dataset being a collection of cases, this can be misleading
        )

    # --- Per-file composites from the unified report ---
    file_composites: dict[str, float] = {}
    for case_result in report.cases:
        meta = file_meta[case_result.name]
        fc = _extract_file_composite(
            case_result, meta["criteria_count"], meta["weights"],
        )
        file_key = case_result.name.split("/", 1)[1]
        file_composites[file_key] = fc
        print(f"  {file_key} composite: {fc:.4f}")

    # --- Section-level composite ---
    section_weights = section.get("weights", DEFAULT_SECTION_WEIGHTS)
    w_files = section_weights.get("files", DEFAULT_SECTION_WEIGHTS["files"])
    w_mechanical = section_weights.get("mechanical", DEFAULT_SECTION_WEIGHTS["mechanical"])

    files_aggregate = (
        sum(file_composites.values()) / len(file_composites)
        if file_composites else 0.0
    )
    section_composite = w_files * files_aggregate + w_mechanical * mechanical_penalty

    logger.info(
        "Section composite: %.4f  (files_agg=%.4f * %.2f, mechanical=%.2f * %.2f)",
        section_composite, files_aggregate, w_files, mechanical_penalty, w_mechanical,
    )
    print(f"\nSection composite: {section_composite:.4f}")
    return section_composite


# ---------------------------------------------------------------------------
# Step 11 — Optional Phase 0 execution
# ---------------------------------------------------------------------------

def _run_phase0_before_eval(args: argparse.Namespace, workspace: Path) -> None:
    """Run Phase 0 before evaluation when --run-phase0 is set."""
    from alpha_lab.client import get_provider
    from alpha_lab.config import load_config
    from alpha_lab.events import AgentEvent
    from alpha_lab.phase0 import run_phase0

    task_config = load_config(args.config)
    provider = get_provider(task_config.provider)

    workspace.mkdir(parents=True, exist_ok=True)
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "phase0_customize.jsonl"

    def event_callback(event: AgentEvent) -> None:
        with open(log_path, "a") as f:
            # Match event callback logging present in `run.py`
            d = event.to_dict()
            d["datetime"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            f.write(json.dumps(d) + "\n")

    run_phase0(provider, task_config, str(workspace), event_callback)


# ---------------------------------------------------------------------------
# Step 10 — CLI argument parser
# ---------------------------------------------------------------------------

DEFAULT_EVAL_CONFIG = "tests/fixtures/evaluations/adapter_evaluation.yaml"


def main() -> None:
    """CLI entry point for alpha-lab-evaluate."""
    parser = argparse.ArgumentParser(
        prog="alpha-lab-evaluate",
        description="Evaluate Phase 0 domain-knowledge customization quality.",
    )
    parser.add_argument(
        "--workspace", required=True,
        help="Path to workspace directory",
    )
    parser.add_argument(
        "--eval-name", required=True,
        help="Name of the evaluation section in the YAML config",
    )
    parser.add_argument(
        "--eval-config", default=DEFAULT_EVAL_CONFIG,
        help=f"Path to evaluation YAML config (default: {DEFAULT_EVAL_CONFIG})",
    )
    parser.add_argument(
        "--run-phase0", action="store_true",
        help="Run Phase 0 before evaluating",
    )
    parser.add_argument(
        "--config",
        help="Task config JSON path (required when --run-phase0 is set)",
    )
    parser.add_argument(
        "--show-table", action=argparse.BooleanOptionalAction, default=True,
        help="Print a detailed table of scores and assertions"
    )
    args = parser.parse_args()

    # Validate mutual dependency
    if args.run_phase0 and not args.config:
        parser.error("--run-phase0 requires --config")

    # Validate eval-name exists in config
    config = load_eval_config(Path(args.eval_config))
    if args.eval_name not in config or args.eval_name in RESERVED_YAML_KEYS or args.eval_name.startswith("_"):
        available = [k for k in config if k not in RESERVED_YAML_KEYS and not k.startswith("_")]
        parser.error(
            f"--eval-name '{args.eval_name}' not found in config. "
            f"Available: {available}"
        )

    asyncio.run(async_main(args))

