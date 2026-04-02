"""Load, save, and resolve domain adapters for alpha-lab.

An adapter is a directory containing:
  - manifest.json — MetricConfig, ExperimentStructure, metadata
  - 9 prompt .md files (one per PROMPT_REGISTRY key)
  - domain_knowledge.md (optional)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from alpha_lab.adapter import (
    ADAPTER_FILES,
    PROMPT_KEYS,
    DomainAdapter,
    ExperimentStructure,
    MetricConfig,
)

logger = logging.getLogger("alpha_lab.adapter_loader")

# Built-in adapters live alongside this module
_BUILTINS_DIR = Path(__file__).parent / "adapters"

# Known built-in adapter names
BUILTIN_ADAPTERS = ["time_series", "cuda_kernel", "nanogpt", "llm_speedrun"]


def load_adapter(adapter_dir: str | Path) -> DomainAdapter:
    """Load a DomainAdapter from a directory on disk.

    Reads manifest.json for metric/experiment config, then loads
    each prompt .md file and domain_knowledge.md.
    """
    adapter_dir = Path(adapter_dir)
    manifest_path = adapter_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {adapter_dir}")

    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)

    # Parse metric config
    metric_raw = manifest.get("metric", {})
    metric = MetricConfig(
        primary_metric=metric_raw.get("primary_metric", "sharpe"),
        direction=metric_raw.get("direction", "maximize"),
        extract_key=metric_raw.get("extract_key", ""),
        display_name=metric_raw.get("display_name", ""),
        secondary_metrics=metric_raw.get("secondary_metrics", []),
    )

    # Parse experiment structure
    exp_raw = manifest.get("experiment", {})
    experiment = ExperimentStructure(
        required_files=exp_raw.get("required_files", ["strategy.py", "run_experiment.py"]),
        entry_point=exp_raw.get("entry_point", "run_experiment.py"),
        results_dir=exp_raw.get("results_dir", "results"),
        results_file=exp_raw.get("results_file", "metrics.json"),
        framework_dir=exp_raw.get("framework_dir", "backtest"),
        framework_files=exp_raw.get("framework_files", []),
    )

    # Load prompts from .md files
    prompts: dict[str, str] = {}
    for key in PROMPT_KEYS:
        prompt_file = adapter_dir / f"{key}.md"
        if prompt_file.exists():
            prompts[key] = prompt_file.read_text()

    # Validate that all required prompts are present
    missing_keys = [key for key in PROMPT_KEYS if key not in prompts]
    if missing_keys:
        missing_files = ", ".join(f"{key}.md" for key in missing_keys)
        raise FileNotFoundError(
            f"Missing prompt files in adapter directory {adapter_dir}: {missing_files}"
        )
    # Load domain knowledge
    domain_knowledge = ""
    dk_path = adapter_dir / "domain_knowledge.md"
    if dk_path.exists():
        domain_knowledge = dk_path.read_text()

    return DomainAdapter(
        domain_name=manifest.get("domain_name", adapter_dir.name),
        domain_description=manifest.get("domain_description", ""),
        prompts=prompts,
        metric=metric,
        experiment=experiment,
        phase2_framework_description=manifest.get(
            "phase2_framework_description", "framework"
        ),
        domain_knowledge=domain_knowledge,
        phase2_review_file=manifest.get("phase2_review_file", "review.md"),
    )


def save_adapter(adapter: DomainAdapter, adapter_dir: str | Path) -> None:
    """Write a DomainAdapter to disk as a directory of files."""
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest.json
    manifest = {
        "domain_name": adapter.domain_name,
        "domain_description": adapter.domain_description,
        "phase2_framework_description": adapter.phase2_framework_description,
        "phase2_review_file": adapter.phase2_review_file,
        "metric": {
            "primary_metric": adapter.metric.primary_metric,
            "direction": adapter.metric.direction,
            "extract_key": adapter.metric.extract_key,
            "display_name": adapter.metric.display_name,
            "secondary_metrics": adapter.metric.secondary_metrics,
        },
        "experiment": {
            "required_files": adapter.experiment.required_files,
            "entry_point": adapter.experiment.entry_point,
            "results_dir": adapter.experiment.results_dir,
            "results_file": adapter.experiment.results_file,
            "framework_dir": adapter.experiment.framework_dir,
            "framework_files": adapter.experiment.framework_files,
        },
    }
    with open(adapter_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Write prompt files
    for key, content in adapter.prompts.items():
        (adapter_dir / f"{key}.md").write_text(content)

    # Write domain knowledge
    if adapter.domain_knowledge:
        (adapter_dir / "domain_knowledge.md").write_text(adapter.domain_knowledge)


def load_builtin_adapter(name: str) -> DomainAdapter:
    """Load a built-in reference adapter by name."""
    adapter_dir = _BUILTINS_DIR / name
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Built-in adapter '{name}' not found. "
            f"Available: {BUILTIN_ADAPTERS}"
        )
    return load_adapter(adapter_dir)


def copy_builtin_to_workspace(name: str, workspace_adapter_dir: str | Path) -> None:
    """Copy a built-in adapter directory to the workspace."""
    src = _BUILTINS_DIR / name
    if not src.is_dir():
        raise FileNotFoundError(f"Built-in adapter '{name}' not found")
    dest = Path(workspace_adapter_dir)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    logger.info("Copied built-in adapter '%s' to %s", name, dest)


def resolve_adapter(
    workspace: str | Path,
    domain: str = "",
) -> DomainAdapter:
    """Resolve which adapter to use. Priority:

    1. Workspace adapter ({workspace}/adapter/) — always wins if present
    2. Built-in adapter matching domain name exactly
    3. Built-in time_series as fallback
    """
    workspace = Path(workspace)
    ws_adapter = workspace / "adapter"

    # 1. Workspace adapter
    if (ws_adapter / "manifest.json").exists():
        logger.info("Loading workspace adapter from %s", ws_adapter)
        return load_adapter(ws_adapter)

    # 2. Built-in matching domain name
    if domain and domain in BUILTIN_ADAPTERS:
        logger.info("Loading built-in adapter: %s", domain)
        return load_builtin_adapter(domain)

    # 3. Fallback to time_series
    logger.info("No adapter found, falling back to built-in time_series")
    return load_builtin_adapter("time_series")
