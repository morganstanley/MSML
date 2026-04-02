"""Domain adapter contract for alpha-lab.

A DomainAdapter parameterizes every domain-specific aspect of the pipeline:
prompts, metrics, experiment structure, and domain knowledge. The stable
kernel (agent loop, providers, GPU executor, experiment DB, dispatcher logic)
stays untouched.

Adapters live as directories of files in {workspace}/adapter/ or as
built-in reference adapters under src/alpha_lab/adapters/{name}/.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricConfig:
    """Configuration for the primary optimization metric."""

    primary_metric: str = "sharpe"
    direction: str = "maximize"  # "maximize" or "minimize"
    extract_key: str = ""  # key in results JSON (defaults to primary_metric)
    display_name: str = ""  # human-readable name (defaults to primary_metric)
    secondary_metrics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.extract_key:
            self.extract_key = self.primary_metric
        if not self.display_name:
            self.display_name = self.primary_metric.replace("_", " ").title()


@dataclass
class ExperimentStructure:
    """Describes the expected file layout for experiments."""

    required_files: list[str] = field(
        default_factory=lambda: ["strategy.py", "run_experiment.py"]
    )
    entry_point: str = "run_experiment.py"
    results_dir: str = "results"
    results_file: str = "metrics.json"
    framework_dir: str = "backtest"
    framework_files: list[str] = field(
        default_factory=lambda: [
            "strategy.py", "engine.py", "metrics.py",
            "baselines.py", "run_backtest.py",
        ]
    )


@dataclass
class DomainAdapter:
    """Complete domain adapter — prompts, metrics, experiment structure."""

    domain_name: str = "time_series"
    domain_description: str = "Time series prediction and forecasting"

    # Keyed by phase/role: phase1, phase2_builder, phase2_critic, phase2_tester,
    # phase3_strategist, phase3_worker_implement, phase3_worker_analyze,
    # phase3_reporter, phase3_fixer
    prompts: dict[str, str] = field(default_factory=dict)

    metric: MetricConfig = field(default_factory=MetricConfig)
    experiment: ExperimentStructure = field(default_factory=ExperimentStructure)

    # Description of the Phase 2 framework for the builder prompt
    phase2_framework_description: str = "walk-forward backtesting framework"

    # Optional domain knowledge injected into all prompts
    domain_knowledge: str = ""

    # Review file location within framework_dir
    phase2_review_file: str = "review.md"


# All valid prompt keys (must match PROMPT_REGISTRY keys in prompts.py)
PROMPT_KEYS = [
    "phase1",
    "phase2_builder",
    "phase2_critic",
    "phase2_tester",
    "phase3_strategist",
    "phase3_worker_implement",
    "phase3_worker_analyze",
    "phase3_reporter",
    "phase3_fixer",
]

# Files allowed in an adapter directory
ADAPTER_FILES = (
    ["manifest.json", "domain_knowledge.md"]
    + [f"{key}.md" for key in PROMPT_KEYS]
)
