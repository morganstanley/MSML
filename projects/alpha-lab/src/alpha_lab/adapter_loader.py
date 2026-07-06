"""Load, save, and resolve domain adapters for alpha-lab.

An adapter is a directory containing:
  - manifest.json -- MetricConfig, ExperimentStructure, metadata
  - 9 prompt .md files (one per PROMPT_REGISTRY key)
  - domain_knowledge.md (optional)
"""

from __future__ import annotations

import json
import logging
import shutil
from importlib.resources import files
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


def _resolve_adapter_path(name_or_path: str) -> Path:
    """Resolve an adapter name or absolute path to a directory on disk."""
    p = Path(name_or_path)
    if p.is_absolute():
        if (p / "manifest.json").exists():
            return p
        raise FileNotFoundError(f"No manifest.json in {p}")
    candidate = Path(str(files("alpha_lab.adapters") / name_or_path))
    if (candidate / "manifest.json").is_file():
        return candidate
    available = [
        d.name for d in files("alpha_lab.adapters").iterdir()
        if (d / "manifest.json").is_file()
    ]
    raise FileNotFoundError(
        f"Adapter '{name_or_path}' not found. Available: {available}"
    )


def load_adapter(adapter_dir: str | Path) -> DomainAdapter:
    """Load a DomainAdapter from a directory on disk."""
    adapter_dir = Path(adapter_dir)
    manifest_path = adapter_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {adapter_dir}")

    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)

    metric_raw = manifest.get("metric", {})
    metric = MetricConfig(
        primary_metric=metric_raw.get("primary_metric", "sharpe"),
        direction=metric_raw.get("direction", "maximize"),
        extract_key=metric_raw.get("extract_key", ""),
        display_name=metric_raw.get("display_name", ""),
        secondary_metrics=metric_raw.get("secondary_metrics", []),
    )

    exp_raw = manifest.get("experiment", {})
    raw_required = exp_raw.get("required_files", [])
    if not isinstance(raw_required, list) or not all(isinstance(f, str) for f in raw_required):
        raise ValueError(
            f"manifest.json: experiment.required_files must be a list of strings, got {raw_required!r}"
        )
    experiment = ExperimentStructure(
        required_files=raw_required,
        entry_point=exp_raw.get("entry_point", "run_experiment.py"),
        results_dir=exp_raw.get("results_dir", "results"),
        results_file=exp_raw.get("results_file", "metrics.json"),
        framework_dir=exp_raw.get("framework_dir", "backtest"),
        framework_files=exp_raw.get("framework_files", []),
    )

    prompts: dict[str, str] = {}
    for key in PROMPT_KEYS:
        prompt_file = adapter_dir / f"{key}.md"
        if prompt_file.exists():
            prompts[key] = prompt_file.read_text()

    missing = [k for k in PROMPT_KEYS if k not in prompts]
    if missing:
        raise ValueError(
            f"Adapter {adapter_dir.name} is missing prompt files: "
            f"{', '.join(f'{k}.md' for k in missing)}"
        )

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

    for key, content in adapter.prompts.items():
        (adapter_dir / f"{key}.md").write_text(content)

    if adapter.domain_knowledge:
        (adapter_dir / "domain_knowledge.md").write_text(adapter.domain_knowledge)


def copy_adapter_to_workspace(src: Path, dest: Path) -> None:
    """Copy an adapter directory to the workspace."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    logger.info("Copied adapter %s to %s", src, dest)


def resolve_reference_adapter(domain_name: str) -> DomainAdapter:
    """Return the reference adapter for *domain_name*.

    Resolves via :func:`_resolve_adapter_path`; raises
    :class:`FileNotFoundError` if *domain_name* does not match a packaged
    adapter (or an absolute path). No silent fallback.
    """
    return load_adapter(_resolve_adapter_path(domain_name))


def resolve_adapter(
    workspace: str | Path,
    domain: str = "",
) -> DomainAdapter:
    """Resolve which adapter to use. Priority:

    1. Workspace adapter ({workspace}/adapter/) -- always wins if present
    2. Adapter by name or absolute path
    """
    workspace = Path(workspace)
    ws_adapter = workspace / "adapter"

    if (ws_adapter / "manifest.json").exists():
        logger.info("Loading workspace adapter from %s", ws_adapter)
        return load_adapter(ws_adapter)

    if not domain:
        raise ValueError("No workspace adapter found and no domain specified")

    adapter_path = _resolve_adapter_path(domain)
    logger.info("Loading adapter: %s", adapter_path)
    return load_adapter(adapter_path)
