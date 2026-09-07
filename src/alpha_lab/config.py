"""Task configuration for alpha-lab."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("alpha_lab.config")

# Try yaml, fall back to json-only
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class Phase3Config:
    """Configuration for Phase 3 experiment orchestration."""

    # Executor type: "slurm" or "local"
    executor: str = "local"

    max_experiments: int = 50
    strategist_interval: int = 300  # seconds between strategist turns
    worker_count: int = 4
    report_interval: int = 10  # generate milestone report every N done experiments

    # SLURM settings (used when executor="slurm")
    max_concurrent_gpus: int = 8
    slurm_partitions: list[str] = field(default_factory=lambda: ["hpc-mid"])
    gpu_per_job: int = 1
    slurm_time_limit: str = "02:00:00"

    # Local GPU settings (used when executor="local")
    gpu_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    max_per_gpu: int = 1  # experiments per GPU (increase for packing)
    time_limit_seconds: int = 7200  # 2 hours default

    # CPU executor settings (for tree-based models)
    cpu_enabled: bool = True  # Run CPU experiments in parallel with GPU
    cpu_max_parallel: int = 4  # Max concurrent CPU experiments
    cpu_time_limit_seconds: int = 3600  # 1 hour default for CPU jobs

    # Python executable for experiment subprocesses
    # Falls back to ALPHALAB_PYTHON env var, then sys.executable
    python_executable: str = ""

    def __post_init__(self) -> None:
        if not self.python_executable:
            self.python_executable = os.environ.get("ALPHALAB_PYTHON", "")

    # Convergence detection
    convergence_threshold: int = 20  # Stop if no improvement for N experiments
    convergence_metric: str = ""  # Metric to track (empty = use adapter's primary_metric)

    # Ablation flags
    no_strategist: bool = False  # Replace strategist with random experiment proposals
    no_playbook: bool = False  # Disable playbook accumulation

    # Conductor scheduling. Setting no_conductor=True is the NOOP fallback —
    # the dispatcher skips the conductor hook entirely, no meta/ writes happen,
    # other agents tolerate empty/missing meta/directives.md gracefully, and
    # the system reverts to its pre-Conductor behavior with no other moving
    # part disturbed. provider/model/reasoning_effort for the Conductor are
    # top-level on TaskConfig (parallel to the main `provider` / `model` /
    # `reasoning_effort` knobs), not here.
    no_conductor: bool = False
    conductor_interval: int = 1800  # seconds between timer-triggered conductor turns

    # Supervisor short-circuit. Setting no_supervisor=True is the NOOP fallback:
    # every Supervisor entry point (validate_adapter, review_phase1,
    # review_phase2, phase3_health_check) returns immediately with an empty
    # summary. Mirrors no_conductor — use when you want to disable the
    # supervisor health-check loop (e.g., to reproduce a run without
    # mid-flight adapter patches) while keeping the rest of the pipeline
    # intact.
    no_supervisor: bool = False

    # Sliding cap on the strategist's pending-proposal queue. The strategist
    # context surfaces this number each turn; the strategist proposes up to
    # (cap - current_pending) per turn. Replaces the previous "fill the entire
    # max_experiments budget in the first session" pattern that left the
    # strategist informed-but-inactive for ~95% of the run. max_experiments is
    # still honored as a lifetime safety ceiling, but the sliding cap is the
    # primary feedback signal for the strategist's turn-to-turn decisions.
    max_pending_proposals: int = 12

    # Cap on the number of variant experiments that can be spawned from one
    # base experiment via the `propose_variant` strategist tool. Variants share
    # the base's strategy.py / run_experiment.py code (only config.yaml or
    # similar should differ); without a cap the strategist can spam variants
    # and crowd out genuine exploration.
    max_variants_per_base: int = 5

    # Iteration budget for one strategist session (0 = unbounded). One
    # audited strategist log carried 3,606 calls and 40% of its run's
    # tokens; at the budget the session is told to consolidate and report,
    # and a small grace window later it is force-ended. The next dispatcher
    # tick starts a fresh session — bounded sessions ARE the rotation.
    strategist_max_iterations: int = 80


@dataclass
class PipelineConfig:
    """Configuration for the multi-phase pipeline."""

    phases: list[str] = field(default_factory=lambda: ["phase1"])
    max_fix_iterations: int = 3
    phase3: Phase3Config = field(default_factory=Phase3Config)


@dataclass
class TaskConfig:
    """Configuration for an analysis task."""

    data_path: str
    description: str
    target: str = ""
    reasoning_effort: str = "low"
    model: str = "gpt-5.2"
    provider: str = "openai"  # "openai", "bedrock", "grok", "kimi", or "glm"
    domain: str = ""  # "time_series", "cuda_kernel", "nanogpt", or free-text for Phase 0
    shell_timeout: int = 300  # seconds for shell_exec commands (agent can request less)
    tool_output_max_chars: int = 8000  # per-tool-result char cap applied in the agent loop
    # Hard cap applied inside tool implementations (shell_exec, grep_file) BEFORE
    # the per-tool-result agent-loop compaction kicks in. Default matches the
    # previous hardcoded `MAX_OUTPUT_CHARS` in tools.py. Must be >= tool_output_max_chars.
    tool_output_hard_cap_chars: int = 30_000
    # Max consecutive tool calls before the agent loop forces a nudge. Previous
    # hardcoded value in agent.py was 50.
    max_consecutive_tool_calls: int = 50
    # When learnings.md exceeds this many tokens, the context manager
    # summarizes it. Previous hardcoded value in context.py was 20_000.
    learnings_summary_threshold_tokens: int = 20_000
    # When cumulative conversation tokens exceed this, ContextManager triggers
    # summarize_and_fork. Previous hardcoded value in context.py was 150_000.
    context_summarization_threshold_tokens: int = 150_000
    # Dispatcher's "stuck worker" watchdog: a worker that has been busy
    # without emitting any event for this many seconds gets a stuck_worker
    # warning. Default 1200 (20 min) — the previous 10-min default was
    # below the upper tail of slow LLM API calls (especially gpt-5.5 xhigh
    # / opus max), producing false alarms. Raise to 1800 for long-thinking
    # runs.
    stuck_worker_threshold_seconds: int = 1200
    # Max consecutive nudges (no-tool-call agent turns) before the agent
    # loop forces termination. Previous hardcoded value in agent.py was 5.
    # Opus runs occasionally output deliberation prose that triggers
    # nudges; raise this to 10 if you see frequent nudge-limit kills.
    max_consecutive_nudges: int = 5
    # Conductor-specific provider/model/reasoning_effort. These parallel
    # the top-level `provider` / `model` / `reasoning_effort` knobs above
    # but apply only to Conductor turns. By default the Conductor routes
    # to Bedrock + opus at max thinking budget regardless of what the
    # rest of the pipeline uses — its job (deep cross-experiment audit,
    # retrospective evaluation, structured judgment) benefits from the
    # strongest available reasoning. Set provider/model to "" to inherit
    # the main pipeline's values.
    conductor_provider: str = "bedrock"
    conductor_model: str = "claude-opus-4-7"
    conductor_reasoning_effort: str = "high"

    # Conductor end-of-run governance. The Conductor may request that the run
    # terminate via the `request_run_end` tool when it has Python-verified
    # evidence (diminishing returns, broad mechanism-class coverage, no
    # progress for N analyzed experiments). These floors prevent premature
    # termination:
    #   - min_runtime_hours: cannot end before this many hours of dispatcher
    #     wall-clock have elapsed (counted from the first dispatcher tick).
    #   - min_analyzed_before_end: cannot end before this many experiments
    #     have reached the `analyzed` or `done` state.
    # Set allow_conductor_end_run=False to disable the tool entirely (the
    # request_run_end tool refuses regardless of floors). All three live at
    # the top level, parallel to the conductor_provider / conductor_model /
    # conductor_reasoning_effort knobs above.
    min_runtime_hours: float = 6.0
    min_analyzed_before_end: int = 100
    allow_conductor_end_run: bool = True

    # Verifier (finding-verifier) config. ONE engine in two modes: a standalone module
    # (scripts/verify_workspace.py) AND, integrated, invoked by the Conductor mid-run. Its three
    # agent roles can each use a different model; "" inherits the main `model`. verifier_provider /
    # verifier_reasoning_effort parallel the main knobs ("" inherits).
    verifier_provider: str = ""
    verifier_reasoning_effort: str = ""
    verifier_worker_model: str = ""
    verifier_critic_model: str = ""
    verifier_userrep_model: str = ""   # covers select + arbitrate + watchdog + final reports
    verifier_max_candidates: int = 4
    verifier_max_rounds: int = 2
    verifier_notebook_timeout: int = 1800
    # Run-end drain: how long dispatcher.stop() waits for an in-flight
    # verification to reach its verdict before the process exits. The run is
    # over and nothing acts on the verdict, but the decision must still be
    # recorded (verify/<candidate>/STATE.json + ARBITER_VERDICT.md). Default
    # covers a full round (2x 1800s notebook executions + agent turns);
    # 0 = no drain (verification dies with the process, as before).
    verifier_drain_seconds: int = 7200
    verifier_watchdog_interval: int = 0   # 0 = User-Rep watchdog off (mirrors no_conductor)
    # Conductor auto-invokes the verifier once this many experiments reach `analyzed` and the
    # Conductor has not itself requested one (0 = never auto-invoke).
    conductor_verify_after_n_strategies: int = 10

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        # tool_output_max_chars is user-settable via top-level config.json; reject
        # values that would make compact_tool_output silently misbehave. bool is a
        # subclass of int in Python, so exclude it explicitly.
        v = self.tool_output_max_chars
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"tool_output_max_chars must be an int, got "
                f"{type(v).__name__}={v!r}"
            )
        if v < 100:
            raise ValueError(
                f"tool_output_max_chars must be >= 100 so head+tail slicing "
                f"leaves room for content, got {v}"
            )
        # Validate the new int knobs the same way: catch bool-as-int, negative
        # or too-small values that would silently make the loop misbehave.
        for fname, fmin in (
            ("tool_output_hard_cap_chars", 100),
            ("max_consecutive_tool_calls", 1),
            ("learnings_summary_threshold_tokens", 100),
            ("context_summarization_threshold_tokens", 100),
            ("stuck_worker_threshold_seconds", 60),
            ("max_consecutive_nudges", 1),
            ("min_analyzed_before_end", 0),
        ):
            fv = getattr(self, fname)
            if isinstance(fv, bool) or not isinstance(fv, int):
                raise ValueError(
                    f"{fname} must be an int, got {type(fv).__name__}={fv!r}"
                )
            if fv < fmin:
                raise ValueError(f"{fname} must be >= {fmin}, got {fv}")
        # min_runtime_hours is the only float knob: same defensive checks but
        # allow either int or float (an int from a JSON config is fine here).
        mrh = self.min_runtime_hours
        if isinstance(mrh, bool) or not isinstance(mrh, (int, float)):
            raise ValueError(
                f"min_runtime_hours must be a number, got "
                f"{type(mrh).__name__}={mrh!r}"
            )
        if mrh < 0:
            raise ValueError(f"min_runtime_hours must be >= 0, got {mrh}")
        if not isinstance(self.allow_conductor_end_run, bool):
            raise ValueError(
                "allow_conductor_end_run must be bool, got "
                f"{type(self.allow_conductor_end_run).__name__}"
            )
        # Sliding pending cap. Must be at least 1 so the strategist can always
        # propose something; without that the system deadlocks at first turn.
        # Validate the phase3 cap fields only when pipeline.phase3 is a
        # proper Phase3Config — tests sometimes pass `pipeline={"phases": ...}`
        # as a dict for brevity and we should not crash on that path.
        phase3_obj = getattr(getattr(self, "pipeline", None), "phase3", None)
        if isinstance(phase3_obj, Phase3Config):
            for fname, fmin in (
                ("max_pending_proposals", 1),
                ("max_variants_per_base", 0),
            ):
                fv = getattr(phase3_obj, fname, None)
                if fv is not None:
                    if isinstance(fv, bool) or not isinstance(fv, int):
                        raise ValueError(
                            f"phase3.{fname} must be an int, got "
                            f"{type(fv).__name__}={fv!r}"
                        )
                    if fv < fmin:
                        raise ValueError(
                            f"phase3.{fname} must be >= {fmin}, got {fv}"
                        )
        # The agent-loop compaction (tool_output_max_chars) runs AFTER the
        # tool-impl hard cap. Allowing hard_cap < per_call_cap would mean the
        # per-call cap can never bite — fail loudly instead of silently.
        if self.tool_output_hard_cap_chars < self.tool_output_max_chars:
            raise ValueError(
                f"tool_output_hard_cap_chars ({self.tool_output_hard_cap_chars}) "
                f"must be >= tool_output_max_chars ({self.tool_output_max_chars})"
            )

    def resolve_data_path(self, base_dir: str | Path) -> str:
        """Resolve data_path relative to base_dir if not absolute."""
        p = Path(self.data_path)
        if not p.is_absolute():
            p = Path(base_dir) / p
        return str(p.resolve())


def load_config(path: str | Path) -> TaskConfig:
    """Load a TaskConfig from a YAML or JSON file.

    Required fields: data_path, description.
    Optional: target, reasoning_effort, model, provider, domain,
    shell_timeout (seconds; max wall-clock for shell_exec commands),
    tool_output_max_chars (per-tool-result char cap in agent loop; default 8000),
    pipeline (nested PipelineConfig/Phase3Config).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        content = f.read()

    # Try JSON first, then YAML
    if path.suffix == ".json" or content.strip().startswith("{"):
        raw: dict[str, Any] = json.loads(content)
    elif YAML_AVAILABLE:
        raw = yaml.safe_load(content)
    else:
        raise ImportError(
            "YAML config requires pyyaml. Either install it or use a .json config file."
        )

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a mapping, got {type(raw).__name__}")

    # Validate required fields
    for key in ("data_path", "description"):
        if key not in raw:
            raise ValueError(f"Missing required config field: {key}")

    # Strip whitespace from string values
    cleaned: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            cleaned[k] = v.strip()
        else:
            cleaned[k] = v

    # Handle nested pipeline config
    if "pipeline" in cleaned and isinstance(cleaned["pipeline"], dict):
        pipeline_raw = dict(cleaned["pipeline"])
        # Handle nested phase3 config inside pipeline
        if "phase3" in pipeline_raw and isinstance(pipeline_raw["phase3"], dict):
            p3_known = {f.name for f in Phase3Config.__dataclass_fields__.values()}
            p3_unknown = [k for k in pipeline_raw["phase3"] if k not in p3_known]
            if p3_unknown:
                logger.warning(
                    "load_config: unknown phase3 field(s) %s in config — "
                    "ignored. (Common typos: 'cpu_executor_enabled' should be "
                    "'cpu_enabled'.)",
                    p3_unknown,
                )
            p3_data = {k: v for k, v in pipeline_raw["phase3"].items() if k in p3_known}
            pipeline_raw["phase3"] = Phase3Config(**p3_data)
        pipeline_known = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
        pipeline_unknown = [k for k in pipeline_raw if k not in pipeline_known]
        if pipeline_unknown:
            logger.warning(
                "load_config: unknown pipeline field(s) %s in config — ignored.",
                pipeline_unknown,
            )
        pipeline_data = {k: v for k, v in pipeline_raw.items() if k in pipeline_known}
        cleaned["pipeline"] = PipelineConfig(**pipeline_data)

    # Only pass known fields to TaskConfig
    known_fields = {f.name for f in TaskConfig.__dataclass_fields__.values()}
    top_unknown = [k for k in cleaned if k not in known_fields]
    if top_unknown:
        logger.warning(
            "load_config: unknown top-level field(s) %s in config — ignored.",
            top_unknown,
        )
    filtered = {k: v for k, v in cleaned.items() if k in known_fields}

    return TaskConfig(**filtered)
