<div align="center">

<img src="header.png" alt="AlphaLab" width="600">

**Autonomous research agent.** Give it a dataset and a task, and it will explore the data end-to-end, build an evaluation framework, then run dozens of experiments on GPUs — all without human intervention.

</div>

---

## 🚨🚨🚨 SAFETY WARNING 🚨🚨🚨

> **⚠️ READ THIS BEFORE RUNNING ⚠️**
>
> AlphaLab runs an LLM **in a loop** as **you** on your Unix machine.
>
> Anything you can do from your shell, AlphaLab can do:
> - **Delete files and directories** — anything you have permission to `rm`, it can `rm`
> - **Overwrite files** — anything you can write to, it can overwrite
> - **Execute arbitrary code** — it writes and runs scripts autonomously
> - **Install packages, modify environments, make network calls**
>
> It is **not malicious**, but it is **autonomous** — and autonomous agents make mistakes. Assume that anything you have permission to destroy *could* be destroyed.
>
> **Before running:**
> - Run in an **isolated workspace** — not in your home directory
> - **Back up anything that matters**
> - Understand that you are giving an AI agent **the same access as your user account**

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/your-org/msml.alpha-lab
cd msml.alpha-lab

# Set up Python (3.11+ required)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env — set ALPHALAB_PYTHON to your Python with torch/numpy/pandas:
#   ALPHALAB_PYTHON=/path/to/your/python
source .env
```

The `ALPHALAB_PYTHON` variable tells AlphaLab which Python to use for running GPU experiments (Phase 3). This should point to a Python environment that has PyTorch, NumPy, Pandas, and other ML dependencies installed. All shell scripts and configs read from this variable automatically.

### 2. Authenticate (on-prem only)

Create the token cache file as your own account, then start the refresh daemon as proid in a **separate terminal**:

```bash
# In your normal terminal — create the cache file
cd /your/full/path/to/msml.alpha-lab
touch .token_cache.json
chmod 666 .token_cache.json

# In a separate tmux session — start the token refresh daemon
tmux new -s token
suu -tr "<ticket>" <proid>
cd /your/full/path/to/msml.alpha-lab   # use your full path, not ~
export ALPHALAB_PYTHON=<your python with httpx>   # see note below
./scripts/token_refresh.sh
# Detach with Ctrl+b d — leave it running
```

> **Note on `ALPHALAB_PYTHON` for the proid session:** `token_refresh.sh` calls `prefetch_token.py`, which needs `httpx`. The proid's default `python3` usually doesn't have it, so you must `export ALPHALAB_PYTHON=<your-own-python-with-httpx>` in the proid shell before running the script — otherwise you'll see `Failed to fetch token: No module named 'httpx'`. The python you point at must also be readable by the proid.

For off-prem, just set `export USE_OFFPREM=1` and `export OPENAI_API_KEY=your-key` instead.

### 3. Run the demo

A working demo is included out of the box using synthetic exchange rate data:

```bash
# Generate the synthetic dataset (no internet needed)
python data/generate_synthetic.py

# Run the full pipeline
python run.py --config data/demo_exchange_config.json --workspace ./workspace_demo
```

This will run all four phases autonomously:
- **Phase 0**: Customize the domain adapter for your data
- **Phase 1**: Explore the dataset, write scripts, generate plots, build a report (~30-90 min)
- **Phase 2**: Build an evaluation framework with tests (~20-60 min)
- **Phase 3**: Run 10 GPU experiments with different ML models (~1-3 hours)

### 4. Watch it work — Web Dashboard

The dashboard lets you watch the pipeline in real-time. **In a separate terminal:**

```bash
# First time only — build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the dashboard (point at the same workspace)
python serve.py --workspace ./workspace_demo --port 8000
# Open http://localhost:8000
```

The dashboard is a passive viewer — it doesn't control the pipeline. You can start it before, during, or after a run. It will:
- **Stream live events** — see the LLM thinking, writing code, running experiments
- **Browse workspace files** — scripts, plots, reports, experiment code
- **Show the kanban board** — experiment lifecycle from proposed → running → done
- **Display the leaderboard** — experiments ranked by metric
- **Chat** — ask questions about system state ("What's the best model?", "Any errors?")

---

## Running Your Own Experiments

### Use an agent to set up your config

This codebase was largely built by Claude Code, and while it aims to be plug-and-play, it may need some tweaking for your setup. We recommend using an AI coding agent:

1. Open [Claude Code](https://claude.ai/code) (or your preferred agent) in this repo
2. Prompt it to **explore the repository and become an expert in it**
3. Tell it: **where your data is**, **what you want to do**, **which model** (gpt-5.2, claude-opus-4-6-v1), **what GPUs you have**
4. Ask it to **write a config and run script** for you
5. If there are errors, paste them back and let it fix things

### Config format

```json
{
  "data_path": "path/to/your/data.csv",
  "description": "What this dataset is...",
  "target": "What to predict/optimize...",
  "provider": "openai",
  "model": "gpt-5.2",
  "reasoning_effort": "low",
  "domain": "",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "phase3": {
      "executor": "local",
      "max_experiments": 50,
      "gpu_ids": [0, 1, 2, 3],
      "max_per_gpu": 1,
      "time_limit_seconds": 21600,
      "python_executable": "/path/to/your/python"
    }
  }
}
```

**`python_executable`:** If left empty (recommended), this reads from the `ALPHALAB_PYTHON` env var you set in `.env`. You can also hardcode a path here per-config.

**Domain options:** Leave `""` for time series (default), or set to `"cuda_kernel"`, `"nanogpt"`, `"llm_speedrun"`, or any free-text description to generate a custom adapter from scratch.

**Provider options:** `"openai"` (default, gpt-5.2) or `"bedrock"` (Claude via AWS Bedrock, model `"claude-opus-4-6-v1"`).

### Included examples

| Config | What it does |
|--------|-------------|
| `data/demo_exchange_config.json` | Synthetic FX rates, 10 experiments — quick demo |
| `data/llm_speedrun_config.json` | LLM pretraining speed/quality optimization |
| `data/paper_llm_speedrun_gpt.json` | Paper reproduction — LLM speedrun with GPT-5.2 |
| `data/paper_traffic_gpt.json` | Paper reproduction — traffic forecasting with GPT-5.2 |

---

## How It Works

AlphaLab runs in four phases (0 → 1 → 2 → 3), each building on the last. A supervisory agent reviews output between phases and monitors health during Phase 3.

| Phase | What happens | Duration |
|-------|-------------|----------|
| **Phase 0** | Resolve and customize the domain adapter for your data | ~5 min |
| **Phase 1** | Explore dataset, write analysis scripts, generate plots, build research report | 30-90 min |
| **Phase 2** | Multi-agent pipeline (Builder/Critic/Tester) creates evaluation framework with tests | 20-60 min |
| **Phase 3** | Strategist + Workers run dozens of GPU experiments, tracked on a kanban board | Hours |

A **Conductor** runs in parallel: it represents you across the whole pipeline, reads everything (proposals, code, debriefs, milestones, its own past decisions, your written instructions), and steers with a light hand.

### The Conductor

The Conductor is a meta-agent that sits above the Strategist/Workers/Reporter/Supervisor and represents your research goal across long runs. It runs after each phase boundary, after each Phase 3 milestone, and on a slow timer (default every 30 min) between milestones — so it's continuously checking whether the system is still serving your goal.

What it can do:

- Park unpromising experiments (soft-cancel; reversible) and reorder priorities so the queue serves the goal
- Annotate the leaderboard with `champion` / `control` / `quarantined` / `exploration` / `exploitation` / `ensemble-candidate` / `home-run-attempt` labels that the Strategist sees
- Issue directives to other agents by writing to `meta/directives.md` (every other agent reads this at the top of its next turn)
- Translate your instructions: write to `meta/instructions/from_user.md` at any time and the Conductor picks it up on its next turn — never blocks, never waits for you
- Set system throttles when GPU/CPU oversubscription is hurting throughput
- Audit its own past decisions retrospectively for whether they paid off
- Request phase rewinds (Python-verified evidence required) when a structural defect demands it
- Keep a running record at `meta/notes_to_user.md` that you can read whenever you want (system never alerts you)

What it does **not** do: propose experiments, write framework code, edit prompts, pause the pipeline, wait for human approval, or alert you. The user reads what the Conductor writes when they choose.

The Conductor's filesystem layout under `<workspace>/meta/`:

```
meta/
├── directives.md              # Conductor → other agents (read every turn)
├── annotations.json           # Conductor-applied leaderboard labels
├── instructions/
│   ├── from_user.md           # YOU write here to instruct the Conductor
│   ├── ack.md                 # Conductor's acknowledgements of your input
│   └── .last_seen             # Conductor bookkeeping
├── notes_to_user.md           # Conductor → you (no expectation you read it)
├── notes_inbox.md             # Strategist/workers → Conductor
├── meta_log.jsonl             # Append-only audit trail of every decision
├── meta_log.md                # Human-readable digest of meta_log.jsonl
├── throttle.json              # {"gpu": "none|slow|halt-new", "cpu": ...}
├── scratch/                   # Conductor's Python analysis scripts
└── backups/<ts>/              # Pre-overwrite snapshots
```

Configuring the Conductor:

```json
{
  "conductor_provider": "bedrock",
  "conductor_model": "claude-opus-4-7",
  "conductor_reasoning_effort": "high",
  "pipeline": {
    "phase3": {
      "no_conductor": false,
      "conductor_interval": 1800
    }
  }
}
```

By default the Conductor runs on Bedrock + opus regardless of what the main pipeline uses — deep cross-experiment audit and retrospective evaluation benefit from the strongest available model. Set `conductor_provider` and `conductor_model` to `""` to inherit the main pipeline's provider/model. If the secondary provider can't be built (auth, network), the dispatcher falls back to the main one with a warning rather than crashing.

**Giving the Conductor instructions:** the only place is `<workspace>/meta/instructions/from_user.md`. The dispatcher creates that file on first run with a comment block explaining what to put there. Replace the comment with your guidance — exploration / exploitation policy, mechanism classes to prefer or avoid, evaluation slices to prioritize, leakage rules, mid-run nudges. The Conductor reads it at the top of every turn and treats whatever's there as authoritative. An empty file means no instructions; the Conductor falls back to its built-in defaults plus the run's `description` and `target` fields.

Set `phase3.no_conductor=true` to disable the Conductor entirely. The dispatcher reverts to its pre-Conductor behavior, the strategist's `cancel_experiments` tool is restored, and no `meta/` writes happen. Useful as an A/B test or fallback.

### Observability

Alpha Lab supports two mutually exclusive observability backends. Both are strictly opt-in — the pipeline runs without either.

#### MLflow (recommended)

MLflow records the full pipeline run as a hierarchy of Runs and Spans in any MLflow-compatible tracking server. Install and enable:

```bash
pip install "mlflow>=2.20.0"
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_EXPERIMENT_NAME=my-alpha-lab-runs
alpha-lab-run --config data/demo.json --workspace ./ws --mlflow
```

What gets logged:
- **Pipeline Run** — one per `alpha-lab-run` invocation. Params: task description, provider, model, config path. Artifacts: `config.json` at start; `phase0/adapter/`, `phase1/learnings.md`, `phase1/data_report/`, `phase2/framework/`, `phase3/leaderboard.md`, `phase3/playbook.md`, `phase3/reports/` at each phase boundary.
- **Experiment Sub-Runs** — one per Phase 3 experiment, nested under the Pipeline Run. Metrics: all numeric results from `update_experiment`; artifacts: debrief file, experiment outputs.
- **MLflow native traces** — per-agent LLM call traces (input/output, tool calls, token counts) via `mlflow.start_span`.

**Benchmark suites:** Use `MLflowRunner` instead of `LocalRunner` to group all benchmark pipeline runs under a single Suite Run:

```python
from alpha_lab.benchmarks.runners import MLflowRunner
runner = MLflowRunner(output_root="./results", suite_run_name="my-benchmark-suite")
```

Or via `run_benchmarks.py`:
```bash
ALPHALAB_MLFLOW=1 python -m alpha_lab.benchmarks.run_benchmarks \
  --generator ... --runner alpha_lab.benchmarks.runners:MLflowRunner \
  --runner-kwargs '{"output_root":"./results","suite_run_name":"my-suite"}'
```

See [MLFLOW.md](MLFLOW.md) for the complete guide.

#### OTel / Tempo (legacy)

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set and `--mlflow` is **not** passed, traces are exported via OpenTelemetry to a Tempo/Jaeger backend. This is the older path; MLflow is preferred for new setups.

For detailed architecture docs, see [DETAILS.md](DETAILS.md).
