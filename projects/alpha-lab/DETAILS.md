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
| `gpu_ids` | List of GPU indices to use (from `nvidia-smi`) |
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

Alpha Lab supports two LLM providers: **OpenAI** (default) and **Anthropic Claude**. Set the `provider` field in your config to switch.

### OpenAI (default)

Uses the OpenAI Responses API (gpt-5.2 by default). Built-in web search via `web_search_preview`.

```json
{
  "provider": "openai",
  "model": "gpt-5.2"
}
```

### Anthropic Claude

Uses the Anthropic Messages API. Requires ANTHROPIC_API_KEY.

```json
{
  "provider": "anthropic",
  "model": "us.anthropic.claude-sonnet-4-20250514",
  "reasoning_effort": "medium"
}
```

**Reasoning effort** controls Claude's extended thinking budget:

| Setting | Thinking budget |
|---------|----------------|
| `"none"` | Disabled |
| `"low"` | 5,000 tokens |
| `"medium"` | 16,000 tokens |
| `"high"` | 32,000 tokens |

**How web search works with Claude:** The Anthropic API does not have built-in web search, so it is search is handled through a proxy. When the agent needs web search, the Anthropic provider presents it to Claude as a regular function tool called `web_search(query)`. When Claude calls it, the tool dispatcher routes the query to GPT-4.1-mini with OpenAI's built-in `web_search_preview`. The search results flow back to Claude as a normal tool result. Claude decides *when* to search; GPT performs the actual web lookup.

```
Claude calls web_search("exchange rate forecasting methods")
        │
        ▼
Tool dispatcher routes to _proxy_web_search()
        │
        ▼
GPT-4.1-mini + web_search_preview does the search
        │
        ▼
Results returned to Claude as tool output
```

**How the provider abstraction works:** All LLM calls go through a `Provider` protocol (`provider.py`). `OpenAIProvider` wraps the Responses API; `AnthropicProvider wraps the Anthropic Messages API. The agent loop, context manager, pipeline, and all other components are provider-agnostic — they call `provider.stream_response()`, `provider.complete()`, etc. without knowing which backend is running. Tool schemas are written in OpenAI format and the Anthropic provider translates them to Anthropic's tool format automatically.

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
      "gpu_ids": [0, 1, 2, 3],
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
| `provider` | `"openai"` | `"openai"` or `"anthropic"` |
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

---

## Project Structure

```
alpha-lab/
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
│   ├── client.py           # Provider factory + 3-tier auth (cache → disk → live SCV)
│   ├── config.py           # YAML/JSON config loading
│   ├── context.py          # Token counting and conversation management
│   ├── provider.py         # Provider protocol and normalized types
│   ├── provider_openai.py  # OpenAI Responses API provider
│   ├── provider_anthropic.py # Anthropic Messages API provider (Claude)
│   ├── phase0.py           # Phase 0: adapter resolution and customization
│   ├── supervisor.py       # Supervisory agent (validates adapter, reviews phases)
│   ├── dispatcher.py       # Phase 3 orchestration loop
│   ├── experiment_db.py    # SQLite kanban board
│   ├── local_gpu.py        # LocalGPUManager (SLURM replacement)
│   ├── local_cpu.py        # LocalCPUManager (parallel CPU experiments)
│   ├── metrics.py          # Thread-safe token/experiment metrics
│   ├── events.py           # Structured event types for real-time monitoring
│   ├── output_generator.py # Deterministic markdown report generation
│   ├── pipeline.py         # Phase 2 multi-agent pipeline
│   ├── prompts.py          # System prompts for all agents
│   ├── run.py              # Headless CLI entry point
│   ├── server.py           # FastAPI + WebSocket server (passive dashboard)
│   ├── slurm.py            # SlurmManager (for clusters)
│   ├── strategist.py       # Phase 3 strategist agent
│   ├── tools.py            # Tool definitions and execution (+ web search proxy)
│   └── worker.py           # Phase 3 worker agents
├── frontend/               # React web dashboard
├── data/                   # Config files and test data
└── requirements.txt        # Python dependencies
```
