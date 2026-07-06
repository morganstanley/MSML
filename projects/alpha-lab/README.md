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

### 1. Clone

```bash
git clone <your-fork-url> alpha-lab
cd alpha-lab
```

### 2. Setup a virtual environment

```bash
# create a venv
source scripts/setup-venv
# pass in any existing venv (source scripts/setup-venv --help)
source scripts/setup-venv --venv-path <venv-path>
```

> The `ALPHALAB_PYTHON` variable tells AlphaLab which Python to use for running GPU experiments (Phase 3). This should point to a Python environment that has PyTorch, NumPy, Pandas, and other ML dependencies installed. All shell scripts and configs read from this variable automatically.

### 3. Run tests

```bash
python -m pytest tests
```

### 4. Set your API key

AlphaLab uses the standard OpenAI API:

```bash
export OPENAI_API_KEY=your-key-here
# Optional: point at an OpenAI-compatible endpoint
# export OPENAI_BASE_URL=https://your-endpoint/v1
```

### 5. Run the demo

A working demo is included out of the box using synthetic exchange rate data:

```bash
# 1. Generate the synthetic dataset and config (no internet needed).
#    Writes exchange_rates.csv + config.json into the given directory.
./scripts/generate-exchange-test-data --output-dir /var/tmp/${USER}/alpha-lab/workspace

# 2. Run the full pipeline using the generated config.
python run.py --config /var/tmp/${USER}/alpha-lab/workspace/config.json --workspace /var/tmp/${USER}/alpha-lab/workspace
```

This will run all four phases autonomously:
- **Phase 0**: Customize the domain adapter for your data
- **Phase 1**: Explore the dataset, write scripts, generate plots, build a report (~30-90 min)
- **Phase 2**: Build an evaluation framework with tests (~20-60 min)
- **Phase 3**: Run 10 GPU experiments with different ML models (~1-3 hours)

### 6. Watch it work — Web Dashboard

The dashboard lets you watch the pipeline in real-time. **In a separate terminal:**

```bash
# First time only — build the frontend (requires Node.js)
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
3. Tell it: **where your data is**, **what you want to do**, **which model** (e.g. gpt-5.2), **what GPUs you have**
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
      "max_per_gpu": 1,
      "time_limit_seconds": 21600,
      "python_executable": "/path/to/your/python"
    }
  }
}
```

**`python_executable`:** If left empty (recommended), this reads from the `ALPHALAB_PYTHON` env var you set in `.env`. You can also hardcode a path here per-config.

**Domain options:** Leave `""` for time series (default), or set to `"cuda_kernel"`, `"nanogpt"`, `"llm_speedrun"`, or any free-text description to generate a custom adapter from scratch.

**Provider options:** `"openai"` (default, gpt-5.2).

### Run identification & tracing

Two tracing backends are supported, mutually exclusive at runtime:

- **OpenTelemetry / Tempo**  — enabled by setting `OTEL_EXPORTER_OTLP_ENDPOINT`
  (e.g., to a Grafana Tempo or Jaeger instance). When configured, spans are
  emitted for each pipeline phase, LLM call, and tool execution. When no
  exporter is configured, tracing is a no-op with zero overhead.

  ```bash
  export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # your Tempo/Jaeger collector
  python run.py --config data/my_config.json --workspace ./workspace --run-id my-experiment-v2
  python run.py --config data/my_config.json --workspace ./workspace --run-id-prefix forex
  ```

- **MLflow** *(opt-in via `--mlflow`)* — Run / metric / artifact logging on a
   MLflow tracking server, plus native MLflow tracing (with
  auto-instrumentation of OpenAI provider calls). Requires
  `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`. 

  ```bash
  export MLFLOW_TRACKING_URI="http://localhost:8081"
  export MLFLOW_EXPERIMENT_NAME="alpha-lab-demo"
  python run.py --config data/my_config.json --workspace ./workspace --mlflow
  ```


### Included examples

| Config | What it does |
|--------|-------------|
| `data/demo_exchange_config.json` | Synthetic FX rates, 10 experiments — quick demo |
| `data/llm_speedrun_config.json` | LLM pretraining speed/quality optimization |
| `data/paper_llm_speedrun_gpt.json` | Paper reproduction — LLM speedrun with GPT-5.2 |
| `data/paper_traffic_gpt.json` | Paper reproduction — traffic forecasting with GPT-5.2 |

---

## Evaluations

`alpha-lab-evaluate` allows for mechanical and model-as-a-judge evaluation of adapter customization results.  Evaluations are configured via YAML with examples present in
`tests/fixtures/evaluations/`.

To execute an evaluation, the minimum requirements are a workspace with customized adapter content and logs and a named set of evaluation criteria from the YAML file.

```bash
# Uses default tests/fixtures/evaluations/adapter_evaluation.yaml
alpha-lab-evaluate --workspace workspace_llm_speedrun_phase0 --eval-name llm_speedrun_pleias
```

Output includes detailed pass/fail explanations for each criterion in the evaluation.  The table output can be disabled which will limit output to numeric metrics.
```bash
alpha-lab-evaluate --workspace workspace_llm_speedrun_phase0 --eval-name llm_speedrun_pleias --no-show-table
```
Output of the above:
```
Mechanical metrics (penalty=0.25):
  input_tokens: 387524.0  [floor=500.0, ceiling=20000.0]  (above_ceiling)
  output_tokens: 11901.0  [floor=200.0, ceiling=15000.0]  (ok)
  tool_calls: 31.0  [floor=3.0, ceiling=25.0]  (above_ceiling)
  duration_seconds: 313.7  [floor=30.0, ceiling=300.0]  (above_ceiling)
Evaluating passthrough ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
  domain_knowledge composite: 0.6743
  phase1 composite: 0.2000
  phase2_builder composite: 0.8000
  phase2_critic composite: 0.2000
  phase2_tester composite: 0.2000
  phase3_fixer composite: 1.0000
  phase3_reporter composite: 1.0000
  phase3_strategist composite: 0.2000
  phase3_worker_analyze composite: 0.2000
  phase3_worker_implement composite: 0.2000

Section composite: 0.4239
```

---

## How It Works

### Orchestration

AlphaLab runs in four phases (0 → 1 → 2 → 3), each building on the last. A supervisory agent reviews output between phases and monitors health during Phase 3.

| Phase | What happens | Duration |
|-------|-------------|----------|
| **Phase 0** | Resolve and customize the domain adapter for your data | ~5 min |
| **Phase 1** | Explore dataset, write analysis scripts, generate plots, build research report | 30-90 min |
| **Phase 2** | Multi-agent pipeline (Builder/Critic/Tester) creates evaluation framework with tests | 20-60 min |
| **Phase 3** | Strategist + Workers run dozens of GPU experiments, tracked on a kanban board | Hours |

### Agents

Every LLM-driven step in the pipeline is performed by one of the agents below. Each has a focused responsibility: exploring data, building or critiquing code, proposing or running experiments, or supervising the work between phases.

| Agent | Phase | Role |
|-------|-------|------|
| Customization | 0 | Patches a built-in adapter template to fit the actual dataset or benchmark |
| Generation | 0 | Builds a full domain adapter from scratch from a free-text domain description |
| Adapter validator | 0 (supervisor) | Reviews the resolved adapter for completeness, manifest validity, and prompt quality |
| Explorer | 1 | Autonomously profiles the dataset; produces scripts, plots, `learnings.md`, and `data_report/` |
| Phase 1 reviewer | 1 (supervisor) | Audits exploration artifacts and patches the adapter when prompts look misaligned with the data |
| Builder | 2 | Constructs the domain-specific evaluation framework (e.g. backtesting code) |
| Critic | 2 | Reviews the framework for lookahead bias, data leakage, and other pitfalls; writes a verdict |
| Tester | 2 | Writes pytest tests for the framework and fixes failures until they pass |
| Phase 2 reviewer | 2 (supervisor) | Validates the framework, tests, and verdict; patches adapter framework config if wrong |
| Strategist | 3 | Periodic meta-agent: reviews the board, prunes the queue, proposes experiments, maintains the playbook |
| Worker (implement) | 3 | Writes a proposed experiment's strategy/config/run files, smoke-tests, advances to `checked` |
| Worker (analyze) | 3 | Analyzes a completed experiment against baselines and writes a debrief |
| Worker (fixer) | 3 | Diagnoses and fixes a failed experiment so it can be resubmitted |
| Reporter | 3 | Generates milestone reports with leaderboards and publication-quality comparison plots |
| Phase 3 health check | 3 (supervisor) | Diagnoses systemic failures (triggered when error rate > 40%) and patches the adapter |
| Interactive | CLI (out-of-pipeline) | REPL agent driven by `alpha-lab`; shares the Explorer's tool surface with `ask_user` enabled |

For detailed architecture docs, see [DETAILS.md](DETAILS.md).
