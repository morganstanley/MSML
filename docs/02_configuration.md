# Configuration

A run is driven by a JSON config file (`--config`); YAML also works if `pyyaml`
is installed. `data_path` and `description` are required; everything else has a
default. Unknown fields produce an error.

```json
{
  "data_path": "data/exchange_rates.csv",
  "description": "Daily FX rates for several currency pairs.",
  "target": "Next-day return of each pair.",
  "provider": "openai",
  "model": "gpt-5.2",
  "reasoning_effort": "low",
  "domain": "time_series",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "max_fix_iterations": 3,
    "phase3": { "executor": "local", "max_experiments": 50, "worker_count": 4 }
  }
}
```

## Config resolution

The run materializes a canonical config at `{workspace}/.alpha_lab/config.json` and
treats it as the single source of truth (intake may edit it in place). `--config` **seeds**
that canonical on the first run, so it is **required only when the workspace has no config
yet** — on a resume you can omit `--config` and the workspace's canonical is used. Passing
`--config` when a canonical already exists is an error **unless** it matches the existing
config (compared as materialized `TaskConfig`s, so optional/defaulted fields don't cause
false mismatches); pass **`--overwrite-config`** to deliberately replace it.

## Task

| Field | Default | Description |
|-------|---------|-------------|
| `data_path` | *required* | Dataset path; resolved relative to the config file if not absolute. |
| `description` | *required* | What the dataset is. |
| `target` | `""` | What to predict or optimize. |
| `domain` | `null` | Adapter selection. A built-in name (or an adapter path) copies and customizes that template; omitting it (or `null`) generates one from scratch; an empty string is rejected. See [Adapters](03_adapters.md). |
| `workspace_includes` | `[]` | Extra workspace-relative files/dirs to carry alongside `data/` (e.g. `["private"]`). Each must exist; no absolute paths or `..`. |
| `memory_spec` | *(none)* | A local path or git source to seed shared memory from. See [Memory](06_memory.md). |
| `shell_timeout` | `300` | Max seconds for a `shell_exec` command. |
| `tool_output_max_chars` | `8000` | Per-tool-result character cap in the agent loop (minimum 100). |
| `web_search_model` | `"gpt-4.1-mini"` | OpenAI model performing the `web_search` proxy for non-OpenAI providers (`bedrock`/`anthropic`/`grok`/`local`). |

## Agent settings

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"openai"` | `"openai"`, `"anthropic"`, `"grok"`, `"bedrock"`, or `"local"`. See [Agents › Providers](04_agents.md). |
| `model` | `"gpt-5.2"` | Model id (provider-specific; e.g. `"claude-opus-4-8"` for `anthropic`/`bedrock`). |
| `reasoning_effort` | `"low"` | `"none"`, `"low"`, `"medium"`, or `"high"`. |

## Intake

Off by default; enable with the `--enable-intake` CLI flag (not a config field).
It runs a short interactive session before the run, where the agent asks
clarifying questions about the task and records the outcome to `agenda.md` for
the run to use.

## Pipeline

`pipeline.phases` selects which of `phase1`/`phase2`/`phase3` run (Phase 0 always
runs first). `max_fix_iterations` (default 3) caps the Phase 2 Builder ↔ Critic /
Tester loop. Phase 3 options live under `pipeline.phase3`.

### Phase 3 — resources

| Field | Default | Description |
|-------|---------|-------------|
| `executor` | `"local"` | `"local"` or `"slurm"`. See [Overview › Executors](01_overview.md). |
| `gpu_ids` | `"auto"` | GPU indices, `"auto"` to detect, or `[]` for CPU-only. |
| `max_per_gpu` | `1` | Experiments packed per GPU. |
| `time_limit_seconds` | `7200` | Kill a GPU experiment after this many seconds. |
| `worker_count` | `4` | Parallel worker agents. |
| `cpu_enabled` | `true` | Run CPU-suitable models in parallel with GPU work. |
| `cpu_max_parallel` | `4` | Max concurrent CPU experiments. |
| `cpu_time_limit_seconds` | `3600` | Kill a CPU experiment after this many seconds. |
| `python_executable` | `""` | Python for experiment subprocesses (empty → `ALPHALAB_PYTHON`, then `sys.executable`). |

SLURM-only (when `executor: "slurm"`): `max_concurrent_gpus`, `slurm_partitions`,
`gpu_per_job`, `slurm_time_limit`.

### Phase 3 — run control

| Field | Default | Description |
|-------|---------|-------------|
| `max_experiments` | `50` | Experiment cap; the strategist may end earlier only through a validated `complete_research` decision. |
| `strategist_interval` | `300` | Seconds between strategist turns. |
| `report_interval` | `10` | Milestone report every N finished experiments. |
| `convergence_threshold` | `20` | Log convergence after N experiments without improvement; this signal does not end the run by itself. |
| `convergence_metric` | `""` | Metric tracked for convergence (empty → the adapter's primary metric). |

### Phase 3 — features & ablations

| Field | Default | Description |
|-------|---------|-------------|
| `jit` | `true` | Just-in-time proposals: the strategist proposes against free capacity as slots open, rather than in fixed batches. |
| `handoff` | `true` | When `true` (the default), each analyzed experiment gets a user-proxy handoff turn that writes directional feedback to `agenda.md`. |
| `no_strategist` | `false` | Replace the strategist with random proposals (ablation). |
| `no_playbook` | `false` | Disable playbook accumulation (ablation). |
| `no_strategist_completion` | `false` | Remove `complete_research` from the strategist, so Phase 3 never receives an explicit completion decision and runs to `max_experiments` (`convergence_threshold` only logs). **Caveat:** this restores the failure mode the completion contract exists to fix — runs that cannot conclude cleanly and stop only at the experiment budget. Leave off outside ablation studies. |
