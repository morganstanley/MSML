# Adapters

A **domain adapter** is the resource that customizes a run's domain knowledge,
metrics, prompts, and file requirements. Every run uses exactly one, and Phase 0
resolves it before anything else runs — it's what lets the same system tackle
exchange-rate forecasting, CUDA kernels, or LLM pretraining without code changes.

An adapter parameterizes three things:

- **Metric** — the primary metric, its direction (maximize / minimize), and a
  display name. This is what Phase 3 optimizes and tracks convergence against.
- **Experiment structure** — the files an experiment must produce, its entry
  point, and the framework directory Phase 2 builds into.
- **Prompts** — a markdown prompt per phase/role, plus a `domain_knowledge.md`
  that is injected into every agent's prompt.

## Built-in adapters

| Domain | Metric | Direction | Framework |
|--------|--------|-----------|-----------|
| `time_series` | Sharpe ratio | maximize | Walk-forward backtesting |
| `cuda_kernel` | Speedup vs PyTorch native | maximize | CUDA kernel harness (`torch load_inline`) |
| `nanogpt` | Wall-clock seconds | minimize | Training / evaluation framework |
| `llm_speedrun` | Validation bits-per-byte | minimize | `train.py` harness with time budgets |
| `tabular_classification` | Accuracy | maximize | Tabular classification evaluation |
| `tabular_regression` | MSE | minimize | Tabular regression evaluation |
| `blackbox` | Response value | minimize | Black-box optimization harness |

## Resolution

Phase 0 picks the adapter from the `domain` config field (see
[Configuration](02_configuration.md)):

| Situation | What Phase 0 does |
|-----------|-------------------|
| Workspace already has `adapter/manifest.json` | Load it and return (resume — no LLM call) |
| `domain` names a built-in, or is a path to an adapter | Copy that template, then run the **customization** agent |
| `domain` is omitted / `null` | Run the **generation** agent to build an adapter from scratch |
| `domain` is set but resolves to neither | Error — a set `domain` that doesn't resolve is treated as a config mistake, not a generation trigger |

**The customization agent** examines your actual data and patches the generic
adapter template to be task-specific. It reads the installed adapter, explores the
dataset (columns, dtypes, distributions, patterns), and patches files — especially
`domain_knowledge.md`, which gets injected into every phase's prompt. This means
even built-in domains produce adapters tailored to the specific dataset. **The
generation agent** does the same job with no template, building a full adapter
from the data and the task `description`/`target` (it reads the `time_series`
adapter as a format reference). Both agents are described in [Agents](04_agents.md).

This is deliberate: we'd rather a misspelled `domain` fail loudly than silently
generate an unintended adapter.

> Note: an empty-string `domain` is rejected by config validation — omit the
> field (or set `null`) to generate. See [Configuration](02_configuration.md).

## Authoring a new adapter

Phase 0 can generate an adapter for you (omit `domain`), but to write one by hand,
create a directory containing the manifest and **all nine** prompt files — the
loader rejects an adapter missing any of them. `domain_knowledge.md` is optional
and, when present, is injected into every prompt.

```
<adapter>/
├── manifest.json
├── domain_knowledge.md          # optional
├── phase1.md
├── phase2_builder.md
├── phase2_critic.md
├── phase2_tester.md
├── phase3_strategist.md
├── phase3_worker_implement.md
├── phase3_worker_analyze.md
├── phase3_reporter.md
└── phase3_fixer.md
```

Reference it by directory path in `domain`, or place it under
`src/alpha_lab/adapters/<name>/` to reference it by name.

### manifest.json

Every field has a default, so a manifest only needs what differs from them:

```json
{
  "domain_name": "my_domain",
  "domain_description": "One line describing the task.",
  "phase2_framework_description": "evaluation framework for my_domain",
  "metric": {
    "primary_metric": "accuracy",
    "direction": "maximize",
    "display_name": "Accuracy"
  },
  "experiment": {
    "required_files": ["run_experiment.py"],
    "entry_point": "run_experiment.py",
    "framework_dir": "harness",
    "framework_files": ["run_harness.py"]
  }
}
```

| Field | Default | Meaning |
|-------|---------|---------|
| `metric.primary_metric` | `sharpe` | Key read from the results file; what Phase 3 optimizes |
| `metric.direction` | `maximize` | `maximize` or `minimize` |
| `metric.extract_key` | = `primary_metric` | Results-file key, if it differs from the metric name |
| `metric.display_name` | title-cased metric | Human-readable metric label |
| `experiment.required_files` | `[]` | Files each experiment must produce |
| `experiment.entry_point` | `run_experiment.py` | Declared entry file (feeds the routing scan; the local executors launch `run_experiment.py` regardless — see below) |
| `experiment.results_dir` / `results_file` | `results` / `metrics.json` | Declared results location (the local executors use `results/metrics.json` — see below) |
| `experiment.framework_dir` | `backtest` | Directory Phase 2 builds the framework into |
| `experiment.framework_files` | `[]` | Framework files; a `run_*.py` here is used as the baseline runner |

### The hardcoded execution contract

Both local executors (`local_gpu.py` and `local_cpu.py`) launch **every** experiment
by running `run_experiment.py`, and mark it done only if `results/metrics.json`
exists — regardless of the manifest's `entry_point`/`results_dir`/`results_file`.
So an experiment must provide a `run_experiment.py` that writes its metrics as JSON
to `results/metrics.json`; the `primary_metric` (or `extract_key`) value is read
from there.

The manifest's `entry_point`/`results_*` are read into the adapter and feed the
GPU/CPU routing scan, **not** the launch — so keep the entry file named
`run_experiment.py` regardless of what `entry_point` says.
