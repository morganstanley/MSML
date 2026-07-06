# AlphaLab — Detailed Documentation

This document covers the full architecture, configuration options, and internals. For a quick start, see [README.md](README.md).

---

## Local GPU Executor

Alpha Lab was designed for SLURM clusters, but includes a **LocalGPUManager** for running on a single multi-GPU box (like a 4x H100 workstation).

### How it works

Instead of submitting jobs via `sbatch`, LocalGPUManager:
- Spawns experiments as subprocesses directly
- Pins each experiment to a specific GPU via `CUDA_VISIBLE_DEVICES`
- Tracks job status by polling `proc.poll()`
- Enforces time limits by killing long-running jobs
- Supports GPU packing (multiple experiments per GPU) via `max_per_gpu`

### Configuration

```json
{
  "pipeline": {
    "phase3": {
      "executor": "local",
      "gpu_ids": [0, 1, 2, 3],
      "max_per_gpu": 1,
      "time_limit_seconds": 21600
    }
  }
}
```

| Setting | Description |
|---------|-------------|
| `executor` | `"local"` for LocalGPUManager, `"slurm"` for SLURM clusters |
| `gpu_ids` | List of GPU indices to use. Omit or set to `[]` to auto-detect via `nvidia-smi` |
| `max_per_gpu` | Experiments per GPU (1 = exclusive, 2-3 = packing if models fit) |
| `time_limit_seconds` | Kill experiments exceeding this (like SLURM `--time`) |

### Same interface as SLURM

Both executors implement the same 5-method interface:
```python
submit_experiment(exp, workspace) -> job_id
poll_jobs(job_ids) -> {job_id: "RUNNING" | "COMPLETED" | "FAILED" | "TIMEOUT"}
cancel(job_id)
can_submit() -> bool
running_gpu_count() -> int
```

The dispatcher doesn't know or care which executor is running — just swap `executor: local` to `executor: slurm` and it works on a cluster.

### CPU Executor (Parallel)

A **LocalCPUManager** runs tree-based and linear models on CPU in parallel with GPU experiments. When enabled, the dispatcher automatically routes experiments to CPU or GPU based on model type.

**Auto-detected CPU models:** XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree, Gradient Boosting, Linear/Lasso/Ridge/ElasticNet, and any sklearn model. Experiments can also set `resource: "cpu"` or `resource: "gpu"` explicitly in their config.

```json
{
  "pipeline": {
    "phase3": {
      "cpu_enabled": true,
      "cpu_max_parallel": 4,
      "cpu_time_limit_seconds": 3600
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `cpu_enabled` | `true` | Enable parallel CPU execution |
| `cpu_max_parallel` | `4` | Max concurrent CPU experiments |
| `cpu_time_limit_seconds` | `3600` | Timeout for CPU jobs (1 hour) |

---

## LLM Provider Support

Alpha Lab uses the **OpenAI** API (the `provider` field is currently `"openai"`).

### OpenAI

Uses the OpenAI Responses API (gpt-5.2 by default). Built-in web search via `web_search_preview`.

```json
{
  "provider": "openai",
  "model": "gpt-5.2"
}
```

**How the provider abstraction works:** All LLM calls go through a `Provider` protocol (`provider.py`). `OpenAIProvider` wraps the Responses API. The agent loop, context manager, pipeline, and all other components are provider-agnostic — they call `provider.stream_response()`, `provider.complete()`, etc. without knowing which backend is running. Adding another backend means implementing the `Provider` protocol and wiring it into `get_provider()` in `client.py`.

### API Configuration

- **Auth**: standard OpenAI API key via the `OPENAI_API_KEY` env var
- **Custom endpoint**: set `OPENAI_BASE_URL` to target an OpenAI-compatible gateway
- **ZDR Compatible**: uses `store=False`, local conversation history tracking

---

## Agent Definitions

Every agent in the pipeline is declared by a markdown file under `src/alpha_lab/agents/registry/`, grouped by phase or role:

```
agents/registry/
├── cli/        # interactive (REPL)
├── phase0/     # customization, generation
├── phase1/     # explorer
├── phase2/     # builder, critic, tester
├── phase3/     # strategist, reporter, worker_{analyze,fixer,implement}
└── supervisor/ # adapter_validator, phase{1,2}_reviewer, phase3_health_check
```

Agent definitions follow the Agents.md and Skills.md open standards. For more information, see https://agents.md/ and https://agentskills.io/home.

Each file contains a YAML frontmatter (tools list, log name, model knobs, prompt source) followed by an optional prompt body. `alpha_lab.agents.load_agent("phase3/worker_implement")` reads the file and returns an `AgentDefinition` dataclass (`alpha_lab.agents.agent_definition`):

`prompt_source` controls where the runtime prompt text comes from:

- **`inline`** — the prompt body of the .md file is the prompt verbatim. Used by Phase 0 (`customization`, `generation`) and the supervisor agents, whose prompts are fixed and not parameterized by the domain.
- **`adapter:<key>`** — the body is discarded; the prompt is rendered each turn from the active `DomainAdapter` by `prompts.build_step_prompt(<key>, …)`. All Phase 1/2/3 agents use this form, so the prompt swaps automatically with the workspace adapter (and picks up live `learnings.md`, `domain_knowledge.md`, supervisor patches, and per-call `extra_context`).

Pipeline call sites (`run.py`, `phase0.py`, `pipeline.py`, `strategist.py`, `worker.py`, `supervisor.py`) run an agent through `sandboxing.sandbox.run_agent`, which assembles it via the single `build_agent` construction site (see "Agent Sandboxing" below); only the cli REPL (`cli.py`) and the `spawn_sub_agent` child (`agent.py`) build an `AgentLoop` directly, in-process. `build_agent` builds the `prompt_builder` closure off `prompt_source`. `d.tools` is a tuple of `ToolDefinition` objects (resolved from the `allowed-tools` names at load time — see "Tool Definitions" below); `get_tool_schemas(d.tools, include_web_search=d.include_web_search)` renders those into provider tool schemas when the `AgentLoop` is built.

### Validating agent definitions

`scripts/validate_agents_frontmatter.py` recursively scans a directory and checks every `.md` file's YAML frontmatter against the Skills.md contract — only `name`, `description`, `allowed-tools`, `license`, and `metadata` are permitted at the top level; anything else must be nested under `metadata:`. It also enforces that `name` and `description` are non-empty strings and that `allowed-tools` / `license` / `metadata` have the right shape.

```bash
venv/bin/python scripts/validate_agents_frontmatter.py src/alpha_lab/agents/registry/
```

Files without any frontmatter (plain markdown READMEs, adapter prompt bodies, etc.) are bucketed separately by default; pass `--strict` to count them as invalid. Exit code is `0` when every file with frontmatter passes, `1` otherwise — suitable for use in pre-commit hooks or CI.

## Tool Definitions

The `tools/` package mirrors the `agents/` package, and separates **reading/building** a tool from its **implementation**:

- **Definition** — each tool is one flat markdown file at `src/alpha_lab/tools/registry/<tool>.md` (no group subdirectories), carrying the same Skills.md frontmatter contract as agents: top-level `name`/`description`, with the JSON-Schema under `metadata.parameters`.
- **Reading/building** — `alpha_lab.tools.load_tool("shell_exec")` reads one file and returns a `ToolDefinition` dataclass (`alpha_lab.tools.tool_definition`), exactly as `load_agent` returns an `AgentDefinition`. `load_tools(names)` resolves an agent's `allowed-tools` into a tuple of `ToolDefinition`s (names without a registry file — e.g. `web_search` — are skipped, since web search is a provider built-in driven by `include_web_search`). `AgentDefinition.tools` holds these `ToolDefinition`s, and `get_tool_schemas(...)` turns them into provider schema dicts at `AgentLoop` build time. Tools are read on demand per agent build — there is no import-time registry.
- **Implementation** — `alpha_lab.tools.execution` holds `execute_tool` (the name-keyed dispatcher), `parse_tool_args`, and every tool's Python implementation (plus the web-search proxy). The definition files carry no implementation reference; the dispatcher maps a tool name to its function.

### Workspace footprint (least-privilege path derivation)

Each tool `.md` also declares its **workspace footprint** under `metadata.workspace_access` — a mapping of workspace-relative path to access effect:

```yaml
metadata:
  workspace_access:
    playbook.md: rw     # <workspace-relative path>: ro | rw
  parameters: {...}
```

Each value is parsed into the `ToolEffect` enum (`ro`/`rw`); keys are workspace-relative paths (`.` is the workspace root; e.g. `experiments.db`, `.memory`, `adapter`, `playbook.md`). A tool that touches no workspace path simply omits `workspace_access`. From this, `alpha_lab.tools.access.build_minimal_workspace_access_schema_for_tools(tools, workspace)` derives the **minimal** read-only / read-write path set an agent needs: it unions the granted tools' footprints (each path joined under `workspace`), expands every footprint to its **mount closure** (the path, its resolved real target, and every symlink reachable by walking it — so a symlinked path is mounted at both its link and real locations, all the way down), then collapses nesting — an ancestor subsumes its descendants, and a read-write path subsumes a read-only request for the same subtree (so an agent with `shell_exec`, which declares `.: rw`, collapses to `rw={workspace root}`, while a shell-less agent stays narrow: e.g. `rw={experiments.db, .memory, playbook.md}`). This schema is what the sandbox layer (below) maps to bwrap mounts; `data_path` (the external dataset) is supplied separately from `TaskConfig`, not derived here.

Tool definitions are validated with the same frontmatter checker as agents: `venv/bin/python scripts/validate_agents_frontmatter.py src/alpha_lab/tools/registry/`.

---

## Agent Sandboxing

When [`bwrap`](https://github.com/containers/bubblewrap) is available, every pipeline agent runs in a subprocess confined to the minimal filesystem its tools need; otherwise it runs in-process exactly as before. The cli REPL always runs in-process.

**Single construction site.** `alpha_lab.agents.factory.build_agent(agent_definition, **runtime)` is the one place an `AgentLoop` is assembled — used identically by in-process callers, the no-bwrap fallback, and the bwrap child, so a sandboxed agent is byte-identical to an in-process one. Static properties (tools, log name, reasoning effort, min report attempts) come from the `AgentDefinition`; the run-level constants `config`, `workspace`, and `api_key` come from the active `RunDeps` (`deps.get()`), so `build_agent`/`run_agent` must run inside a `with RunDeps(...)` scope (run.py wraps the whole pipeline); only genuinely per-call values (`initial_message`, `extra_context`, `log_name`, `tools_include`, …) are passed in. There is **no** separate "agent spec" class — the agent is reconstructed in the child from its `agent_id` via `load_agent`, and the per-call values travel as a plain JSON dict. `config`/`workspace`/`api_key` still cross the bwrap boundary inside that dict (config as a mapping, workspace as a path string) so the child can open its **own** `RunDeps`; carrying the resolved `TaskConfig` lets the child see run-resolved values like `gpu_ids`.

**Flow.** Callers invoke `sandboxing.sandbox.run_agent(agent_id, event_callback, *, initial_message, …, provider, db, adapter, metrics, owner=None, handle_attr="_run_handle")` (with `config`/`workspace`/`api_key` read from the active `RunDeps`), which allocates an `AgentRunHandle`, publishes it on `owner.<handle_attr>` for the run's duration (default `_run_handle`; the strategist and pipeline override to `_agent` / `_current_agent`) so the orchestrator's `stop()` reaches it, and returns it:
- **No bwrap** (or `ALPHALAB_AGENT_NOSANDBOX` set) → `build_agent(...)` and `agent.run(...)` in-process; the live `MetricsCollector` is passed straight through.
- **bwrap** → derive the minimal RO/RW mounts from the agent's tools via `tools.access.build_minimal_workspace_access_schema_for_tools`, spawn `python -m alpha_lab.sandboxing.runner` under bwrap, write the JSON payload to stdin (before taking the handle lock, so a large `initial_message`/`extra_context` write can't stall a concurrent `stop()`), and relay the child's stdout event JSONL back through `events.event_from_dict` to `event_callback`. `ask_user`/`stop` are relayed to the child's stdin via the `AgentRunHandle` (control messages issued before the child exists are buffered and flushed on attach, never dropped). `stop()` **escalates** stdin → `SIGTERM` → `SIGKILL` via the shared `process_control.escalate_termination` (a generous cooperative grace first, so a slow-but-correct shutdown — final metric flush, `run_deps.close()`, MLflow finalization — isn't killed).
  - **Metrics** can't share the live collector across the boundary, so the child records into its own and emits **incremental `MetricsEvent` deltas**, flushed from the recording site via `MetricsCollector(on_record=…)` (robust to any counter, not just API/error ones); the relay merges each delta additively into the parent's collector, so a hard child crash (segfault/OOM/SIGKILL) loses at most the last recorded counter rather than the whole run's.
  - **Tracing.** The payload carries a W3C `traceparent` carrier (`tracing.inject_context`); the child re-initializes MLflow/OTel from the inherited env — mirroring run.py's mutually-exclusive backend selection (`--mlflow` ⇒ MLflow only) — and attaches that context (`tracing.context_from_carrier`) before building the agent, so its spans/traces nest under the parent run instead of starting an orphan trace.
  - **Stores (board + memory).** The child never opens `experiments.db` or `memory.db` itself — two processes on one SQLite file is unsafe (corruption on NFS, where WAL/`fcntl` locking don't work). Its board tools (`read_board`/`propose`/`update`/`cancel`) and memory tools (`memory_store`/`search`/`read`) reach the parent's stores through a single `ProxyChannel` (`sandboxing/db_proxy.py`): each call emits a `backend_request` line tagged with its `target` (`experiments`/`memory`), and the parent's relay applies it against the **single** `ExperimentDB`/`MemoryStore` it owns, replying with `backend_response` only after the write commits — so the child's blocked tool sees the real result. `Experiment`/`MemoryEntry` rows cross the boundary via `encode_result`/`decode_result`. A dead parent (`--die-with-parent`) kills the child and `fail_all` unblocks any in-flight call, so the one ambiguous window (committed-but-un-acked) collapses into a full teardown rather than an inconsistency.

**Mounts.** The child gets: the tool-derived workspace paths (RW overlay an RO ancestor); an always-RW **agent-infrastructure** set the agent writes regardless of tools — the `logs/` dir (JSONL log) and `learnings.md` / `.memory/learnings_archive` (rewritten by context summarization); the repo (RO); the venv (`sys.prefix`/`sys.base_prefix`, RO); `data_path` (RO); and `/usr`, `/etc`, `/proc`, `/dev`, `/tmp`. Each RW bind **source** is created if absent so bwrap can bind it (`_create_rw_sources`): an existing path is left as-is, a missing path with a file extension (e.g. `experiments.db`, `playbook.md`) is touched as a file, and anything else is created as a directory. A tool's declared DB/memory paths **are** mounted (the access schema no longer excludes them), but the child never opens those files directly — its board (`read_board`/`propose`/`update`/`cancel`) and memory (`memory_store`/`search`/`read`) tools reach the parent's single connection through the DB proxy (above), so the on-disk mount is unused. (A shell-bearing worker also mounts the workspace root RW for `shell_exec`; LLM-authored experiment code writes `metrics.json`, not the board.) When an agent's `metadata.needs_gpu: true` (phase 1 explorer, phase 2 builder/critic/tester, phase 3 workers), the NVIDIA devices are bound in (probed once per process).

**Namespace policy.** Only the PID namespace is unshared (`--unshare-pid --die-with-parent --new-session`). The network and user namespaces are **inherited** so the agent keeps its host identity and can reach the LLM API — egress is the host's concern, not the sandbox's. (This is why the generic repo `sandbox` script's `--unshare-all --unshare-net` is *not* used for agents.)

**Escape hatch.** Set `ALPHALAB_AGENT_NOSANDBOX=1` to force in-process execution (the test suite does this, since fake providers can't cross the process boundary). In-process, the agent uses the real `ExperimentDB`/`MemoryStore` directly; only under bwrap do the store proxies stand in, so the parent process stays the sole connection owner either way.

**Tuning.** The parent's child-supervision timeouts can be overridden via environment variables (each defaults to the value shown, in seconds except where noted):

| Variable | Default | Controls |
| --- | --- | --- |
| `ALPHALAB_SANDBOX_STOP_GRACE_SECONDS` | `30` | `stop()` cooperative grace before escalating to `SIGTERM` — must exceed a normal shutdown (final metric flush, `run_deps.close()`, MLflow finalization). |
| `ALPHALAB_SANDBOX_STOP_KILL_GRACE_SECONDS` | `5` | Grace for the child to honor `SIGTERM` before `SIGKILL`. |
| `ALPHALAB_SANDBOX_MAX_RELAYED_STDERR` | `4000` | Max trailing characters of the child's stderr relayed on failure (chars, not seconds). |

---

## Phase Details

### Phase 0: Domain Adapter Resolution & Customization

Phase 0 runs before anything else to set up the domain adapter — the configuration layer that tells the pipeline what metrics to optimize, what experiment files to expect, and what domain-specific knowledge to inject into every agent prompt.

**Four paths:**

| Scenario | What happens |
|----------|-------------|
| Workspace already has `adapter/manifest.json` | Load and return (resume — no LLM call) |
| Domain matches a built-in (`time_series`, `cuda_kernel`, `nanogpt`) | Copy template → run customization agent |
| No domain specified | Copy `time_series` template → run customization agent |
| Free-text domain description | Run full generation agent to create adapter from scratch |

**The customization agent** (paths 2 and 3) examines your actual data and patches the generic adapter template to be task-specific. It reads the installed adapter, explores the dataset (columns, dtypes, distributions, patterns), and patches files — especially `domain_knowledge.md`, which gets injected into every phase's prompt. This means even built-in domains produce adapters tailored to the specific dataset.

**Built-in adapters:**

| Domain | Primary Metric | Direction | Framework |
|--------|---------------|-----------|-----------|
| `time_series` | Sharpe ratio | maximize | Walk-forward backtesting |
| `cuda_kernel` | throughput (GFLOPS) | maximize | Benchmark framework |
| `nanogpt` | wall clock seconds | minimize | Training framework |
| `llm_speedrun` | val BPB (bits per byte) | minimize | Training harness with time/param limits |

### Phase 1: Autonomous Data Exploration

**Duration:** 30-90 minutes depending on dataset complexity

A single LLM agent explores your dataset from scratch with no human guidance. It operates in a continuous loop: think → write code → execute → observe results → think again. The agent has access to shell commands, can write and execute Python scripts, view generated plots, and search the web for domain context.

**What it does:**

1. **Planning** — Creates `plan.md` with a detailed checklist of everything it intends to investigate
2. **Schema discovery** — Loads the data, inspects dtypes, identifies date columns, categorical vs numeric, etc.
3. **Statistical profiling** — For every column: distributions, missing values, outliers, cardinality
4. **Target analysis** — If you specified a target variable: distribution, autocorrelation, stationarity tests, seasonality decomposition
5. **Temporal analysis** — Time series structure, gaps, frequency, trends
6. **Correlation analysis** — Feature relationships, multicollinearity, lagged correlations
7. **Data quality** — Duplicates, inconsistencies, suspicious patterns
8. **Domain research** — Web searches for context about the data domain
9. **Report assembly** — Compiles everything into a structured report

**Output structure:**

```
workspace/
├── plan.md                 # Checklist the agent works through
├── learnings.md            # Accumulated knowledge (updated continuously)
├── scripts/                # All Python analysis scripts
├── plots/                  # All generated visualizations
├── notes/                  # Per-topic findings as markdown
└── data_report/            # Final structured deliverable
    ├── schema.md
    ├── statistics.md
    └── findings.md
```

### Phase 2: Evaluation Framework Construction

**Duration:** 20-60 minutes

A multi-agent pipeline builds a domain-appropriate evaluation framework specifically designed for your dataset.

**The agents:**

| Agent | Role |
|-------|------|
| **Builder** | Writes the framework code: Strategy base class, walk-forward engine, performance metrics, baseline strategies |
| **Critic** | Reviews for lookahead bias, data leakage, incorrect metric calculations, edge cases |
| **Tester** | Writes pytest tests with known-output assertions, runs them, reports failures |

**The loop:**

```
Builder writes code
       ↓
Critic reviews → Issues found? → Builder fixes → Critic reviews again
       ↓ (no issues)
Tester writes tests
       ↓
Tests pass? → No → Builder fixes → Tests run again
       ↓ (yes)
Phase 2 complete
```

This repeats up to `max_fix_iterations` times (default 3) until the code passes both review and tests.

### Phase 3: GPU-Scale Experiment Orchestration

**Duration:** Hours to days (runs until `max_experiments` reached)

An ever-running system that blasts your GPUs with diverse model experiments. Multiple LLM agents work in parallel, coordinated by a pure-Python dispatcher. Experiments are tracked on a SQLite kanban board.

**Architecture:**

```
                    ┌──────────────┐
                    │  Strategist  │  Reviews results, proposes new experiments
                    │   (LLM)      │  Maintains playbook.md of what works
                    └──────┬───────┘
                           │ proposes experiments
                    ┌──────▼───────┐
                    │   SQLite DB  │  Kanban board with experiment states:
                    │              │  proposed → to_implement → implemented →
                    │              │  checked → queued → running → finished →
                    │              │  analyzed → done
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Dispatcher  │  Pure Python orchestration:
                    │  (no LLM)    │  - Assigns workers to tasks
                    │              │  - Submits jobs to GPU executor
                    │              │  - Polls status, handles failures
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │Worker 0 │       │Worker 1 │       │Worker 2 │
    │  (LLM)  │       │  (LLM)  │       │  (LLM)  │
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
    implementing      analyzing         reporting
    experiment_07     experiment_04     milestone_02
```

**Experiment lifecycle:**

| State | Description |
|-------|-------------|
| `proposed` | Strategist proposed it, not yet picked up |
| `to_implement` | Worker assigned to write the code |
| `implemented` | Code written, needs review |
| `checked` | Passed smoke test, ready for GPU |
| `queued` | Submitted to GPU executor, waiting |
| `running` | GPU job in progress |
| `finished` | GPU job done, needs analysis |
| `analyzed` | Worker wrote debrief with results |
| `done` | Fully complete |

**Strategist behavior:**

The Strategist agent runs periodically (every `strategist_interval` seconds or after N experiments complete). It:

1. Reads the current experiment board (what's done, what's in progress)
2. Reviews debriefs from recently completed experiments
3. Identifies patterns (which architectures work, which hyperparameters matter)
4. Proposes 2-5 new experiments that explore promising directions
5. Updates `playbook.md` with accumulated wisdom

**Milestone reports:**

Every `report_interval` experiments (default 10), a worker generates a milestone report summarizing: best performing models, what's been tried, emerging patterns, recommended next steps.

---

## Full Configuration Reference

```json
{
  "data_path": "data/exchange_rates.csv",
  "description": "Description of your dataset...",
  "target": "What you want to predict/analyze...",
  "provider": "openai",
  "model": "gpt-5.2",
  "reasoning_effort": "low",
  "domain": "",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "max_fix_iterations": 3,
    "phase3": {
      "executor": "local",
      "max_experiments": 50,
      "worker_count": 4,
      "max_per_gpu": 1,
      "time_limit_seconds": 21600,
      "python_executable": "/path/to/your/python",
      "convergence_metric": "",
      "convergence_threshold": 20,
      "strategist_interval": 300,
      "report_interval": 10,
      "cpu_enabled": true,
      "cpu_max_parallel": 4,
      "cpu_time_limit_seconds": 3600,
      "no_strategist": false,
      "no_playbook": false
    }
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"openai"` | `"openai"` |
| `model` | `"gpt-5.2"` | Model identifier (provider-specific) |
| `reasoning_effort` | `"low"` | `"none"`, `"low"`, `"medium"`, `"high"` |
| `domain` | `""` | Domain adapter: `""` or `"time_series"` (default), `"cuda_kernel"`, `"nanogpt"`, `"llm_speedrun"`, or free-text description |
| `python_executable` | `""` | Full path to Python binary for experiment subprocesses (empty = `sys.executable`) |
| `convergence_metric` | `""` | Override adapter's primary metric for convergence tracking (empty = use adapter default) |
| `convergence_threshold` | `20` | Stop if no improvement for N experiments |
| `no_strategist` | `false` | Replace strategist with random experiment proposals (ablation) |
| `no_playbook` | `false` | Disable playbook accumulation (ablation) |

---

## Event System & Metrics

### Real-Time Events

The pipeline emits structured events for real-time monitoring (used by both the CLI and web dashboard):

| Event Type | Description |
|-----------|-------------|
| `StatusEvent` | Agent status transitions (starting, thinking, tool_executing, done, error) |
| `PhaseEvent` | Phase transitions with iteration tracking |
| `ExperimentEvent` | Experiment state changes with metrics |
| `BoardSummaryEvent` | Periodic kanban board snapshots |
| `ToolCallEvent` / `ToolResultEvent` | Tool execution tracing |
| `FileChangedEvent` | Workspace file watcher events |
| `ErrorEvent` | Error logging |

### Metrics Collection

`MetricsCollector` provides thread-safe, in-memory tracking with no external dependencies:

- Token accounting (input/output tokens per API call)
- API call counts and error rates
- Experiment throughput (count, average duration, experiments/hour)
- Session uptime

Call `metrics.snapshot()` for a JSON-serializable summary at any point.

### Output Generation

After each phase, `OutputGenerator` produces polished markdown documents in `{workspace}/output/` — no LLM calls, purely deterministic extraction from workspace artifacts:

| Document | Source |
|----------|--------|
| `01_data_exploration.md` | Phase 1 findings, schema, learnings, plots |
| `02_backtest_methodology.md` | Phase 2 framework design, baseline strategies |
| `03_baseline_results.md` | Baseline metric tables by strategy/location |
| `04_milestone_NNN.md` | Phase 3 milestone reports |
| `index.md` | Table of contents for all generated docs |

### Distributed Tracing (OpenTelemetry)

The `tracing.py` module provides opt-in OpenTelemetry instrumentation. When an OTLP exporter is configured (via `OTEL_EXPORTER_OTLP_ENDPOINT`), spans are emitted for the full pipeline lifecycle. When unconfigured, the OTel API is a no-op with zero overhead.

**Span hierarchy:**

```
alpha_lab.run                              (root — entire pipeline)
├── phase0.resolve                         (adapter resolution)
├── invoke_agent conversation              (Phase 1 agent)
│   ├── chat gpt-5.2                       (LLM API call with token counts)
│   ├── execute_tool shell_exec            (tool execution)
│   └── ...
├── invoke_agent builder / critic          (Phase 2 agents)
└── phase3.dispatcher                      (Phase 3 orchestration)
    ├── submit_experiment exp_name         (job submission)
    ├── invoke_agent strategist            (strategist thread)
    └── invoke_agent worker_0_implement_*  (worker threads)
```

**Run identification:** Each run gets a `run.id` attribute (auto-generated or user-supplied via `--run-id` / `--run-id-prefix` CLI args or `ALPHALAB_RUN_ID` env var). The run ID and trace ID are persisted to `{workspace}/trace_info.json` for later lookup.

**Restart semantics:** When a run resumes (same workspace, same `--run-id`), a new trace is created with an OTel span link pointing to the previous trace. The `attempt` counter increments. Query by `run.id` to find all attempts; span links make them navigable in trace backends like Grafana Tempo.

**`trace_info.json` format:**

```json
{
  "run_id": "my-experiment-20260506-a1b2c3",
  "attempt": 2,
  "trace_id": "abcdef1234567890abcdef1234567890",
  "span_id": "1234567890abcdef",
  "previous_trace_id": "99887766554433221100ffeeddccbbaa",
  "user": "corcole",
  "started_at": "2026-05-06T14:30:00Z",
  "config": "data/exchange_config.json"
}
```

**Querying traces after a run:**

```bash
# Find trace ID for a completed run
cat workspace/trace_info.json | jq .trace_id

# Query all attempts for a logical run (in Grafana Tempo)
# TraceQL: { .run.id = "my-experiment-20260506-a1b2c3" }
```

---

## Project Structure

```
alpha-lab/
├── scripts/
│   └── validate_agents_frontmatter.py    # Skills.md frontmatter validator for agent definitions
├── examples/
│   ├── run_traffic_gpt.sh      # Paper reproduction: traffic forecasting
│   └── run_llm_speedrun_gpt.sh # Paper reproduction: LLM speedrun
├── run.py                  # Simple runner (no PYTHONPATH needed)
├── serve.py                # Simple server runner
├── src/alpha_lab/
│   ├── adapter.py          # DomainAdapter dataclass and file constants
│   ├── adapter_loader.py   # Load/copy/resolve adapters
│   ├── adapters/           # Built-in adapter templates
│   │   ├── time_series/    # Sharpe ratio, walk-forward backtesting
│   │   ├── cuda_kernel/    # Throughput GFLOPS, benchmark framework
│   │   ├── nanogpt/        # Wall clock seconds, training framework
│   │   └── llm_speedrun/   # Val BPB, LLM pretraining quality optimization
│   ├── agent.py            # Core agent loop (provider-agnostic)
│   ├── agents/             # Markdown-driven agent registry (see "Agent Definitions" above)
│   │   ├── __init__.py     # Exports AgentDefinition + AGENTS_DIR (importlib.resources handle)
│   │   ├── agent_definition.py  # AgentDefinition frozen dataclass
│   │   ├── factory.py      # build_agent() — single AgentLoop construction site
│   │   └── registry/       # Markdown agent definitions, grouped by phase/role
│   │       ├── cli/        # interactive
│   │       ├── phase0/     # customization, generation
│   │       ├── phase1/     # explorer
│   │       ├── phase2/     # builder, critic, tester
│   │       ├── phase3/     # strategist, reporter, worker_{analyze,fixer,implement}
│   │       └── supervisor/ # adapter_validator, phase{1,2}_reviewer, phase3_health_check
│   ├── benchmarks/         # Benchmark suite generation, registration, runners, and reporting
│   ├── client.py           # Provider factory (standard OpenAI client)
│   ├── config.py           # YAML/JSON task config loading
│   ├── context.py          # Token counting and conversation management
│   ├── provider.py         # Provider protocol and normalized types
│   ├── provider_openai.py  # OpenAI Responses API provider
│   ├── phase0.py           # Phase 0: adapter resolution and customization
│   ├── supervisor.py       # Supervisory agent (validates adapter, reviews phases)
│   ├── dispatcher.py       # Phase 3 orchestration loop
│   ├── experiment_db.py    # SQLite kanban board
│   ├── local_gpu.py        # LocalGPUManager (SLURM replacement)
│   ├── local_cpu.py        # LocalCPUManager (parallel CPU experiments)
│   ├── memory.py           # Persistent {workspace}/.memory/ store (memory_store/search/read tools)
│   ├── metrics.py          # Thread-safe token/experiment metrics
│   ├── events.py           # Structured event types for real-time monitoring
│   ├── output_generator.py # Deterministic markdown report generation
│   ├── pipeline.py         # Phase 2 multi-agent pipeline
│   ├── prompts.py          # System prompt registry + build_step_prompt() for adapter-driven agents
│   ├── run.py              # Headless CLI entry point
│   ├── sandboxing/         # Agent sandboxing (see "Agent Sandboxing")
│   │   ├── sandbox.py      # bwrap confinement: is_available/run_agent/popen/AgentRunHandle
│   │   ├── runner.py       # `python -m alpha_lab.sandboxing.runner` — sandboxed child entrypoint
│   │   └── db_proxy.py     # Child-side store proxies (experiments.db, memory.db) → parent
│   ├── server.py           # FastAPI + WebSocket server (passive dashboard)
│   ├── slurm.py            # SlurmManager (for clusters)
│   ├── strategist.py       # Phase 3 strategist agent
│   ├── tools/              # Tool registry + execution (mirrors the agents package)
│   │   ├── __init__.py     # TOOLS_DIR + load_tool()/load_tools() + get_tool_schemas() (reading/building)
│   │   ├── tool_definition.py  # ToolDefinition dataclass (name, description, parameters, workspace_access mapping)
│   │   ├── access.py       # build_minimal_workspace_access_schema_for_tools() — minimal RO/RW path set per agent
│   │   ├── execution.py    # execute_tool() dispatcher + tool implementations (+ web search proxy)
│   │   └── registry/       # One flat <tool>.md per tool (shell_exec, read_file, grep_file,
│   │                       #   view_image, ask_user, report_to_user, memory_{store,search,read},
│   │                       #   spawn_sub_agent, update_playbook, {read_board, propose_experiment,
│   │                       #   update_experiment, cancel_experiments, reality_check},
│   │                       #   {read,write,patch}_adapter*, read_reference_adapter)
│   ├── tracing.py          # OpenTelemetry instrumentation (no-op when unconfigured)
│   ├── validation.py       # System-side reality_check for experiments (called by the reality_check tool)
│   ├── experiment_validation.py  # Worker-side validation helpers called inside run_experiment.py
│   └── worker.py           # Phase 3 worker agents
├── frontend/               # React web dashboard
├── data/                   # Config files and test data
└── requirements.txt        # Python dependencies
```
