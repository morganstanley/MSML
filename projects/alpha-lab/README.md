<div align="center">

<img src="header.png" alt="AlphaLab" width="600">

**Autonomous research agent.** Give it a dataset and a task, and it will explore the data end-to-end, build an evaluation framework, then run dozens of experiments on GPUs — all without human intervention.

[DETAILS.md](DETAILS.md)

*Released by the Morgan Stanley Machine Learning Research Team*

</div>

---

## What is AlphaLab?

AlphaLab automates quantitative, verifiable research end-to-end. You give it a dataset and a natural-language objective — "optimize these CUDA kernels", "forecast this time series", "train the best small language model you can in 20 minutes" — and it goes off for hours or days, autonomously exploring the data, building its own evaluation framework, running dozens of GPU experiments, and evolving its own methodology as it learns what works. No human in the loop.

The system is **domain-agnostic**: the same pipeline handles CUDA kernel optimization, LLM pretraining, time series forecasting, and any other domain with objective, easy-to-evaluate metrics. All domain-specific behavior lives in *adapters* that the model generates itself from the data — AlphaLab writes its own prompts, defines its own metrics, and builds its own evaluation harness. It is **self-evolving**: a persistent *playbook* accumulates knowledge experiment-by-experiment, functioning as online prompt optimization — by the end of a campaign, the system has discovered and encoded domain methodology that didn't exist anywhere in its code or prompts at launch. And it is **recursively LLM-powered**: agents can spawn sub-agents as needed, delegating subtasks without polluting the parent's context.

This repository accompanies the paper *AlphaLab: Autonomous Multi-Agent Research Across Optimization Domains with Frontier LLMs*. See the paper for full experimental details, ablations, and analysis.

<div align="center">
<img src="docs/pipeline_overview.png" alt="AlphaLab Pipeline" width="800">
</div>

### The Pipeline

AlphaLab is a *harness*: a combination of tools and a structured environment that converts a frontier LLM into an autonomous research agent. The system is LLM-agnostic — any model that supports tool use and multi-modal input can be dropped in with no changes to the infrastructure. We evaluate with GPT-5.2 and Claude Opus 4.6 in the paper; all differences in outcomes are attributable to the model, not the harness.

The pipeline runs in four phases:

- **Phase 0 — Adapter Resolution.** All domain-specific behavior is parameterized by a *domain adapter*: 11 files comprising prompt templates (one per agent role), metric definitions, experiment structure, and a `domain_knowledge.md` document that is injected into every agent's context for the entire campaign. Phase 0 generates or customizes this adapter by examining the actual data and searching the web for prior work. The key idea is that *prompt engineering is performed by the model*, grounded in the data.

- **Phase 1 — Data Exploration.** A single Explorer agent operates autonomously for 1-2 hours: it generates a plan, then works through it — writing and running Python scripts, generating plots, searching the web for relevant papers, and updating its notes after each finding. It produces a human-readable research report and a machine-readable `learnings.md` consumed by later phases.

- **Phase 2 — Adversarial Evaluation Construction.** Evaluation correctness is critical — if the metric is wrong, every experiment optimizes the wrong objective. Phase 2 addresses this through a multi-agent adversarial loop: a **Builder** writes the evaluation framework, a **Critic** (a fresh agent with no shared context) audits for data leakage, lookahead bias, and metric errors, and a **Tester** writes and runs an automated test suite. The loop repeats until all tests pass.

- **Phase 3 — GPU-Scale Experimentation.** The core of the system: a sustained experimental campaign where a **Strategist** proposes experiments, **Workers** implement and analyze them, and a **Dispatcher** (pure Python, no LLM) orchestrates GPU resources. A persistent **Playbook** accumulates domain knowledge across experiments, creating a feedback loop that functions as online prompt optimization — the system literally gets better at its task as the campaign progresses. The campaign runs until convergence (no improvement for 20 consecutive experiments) or budget exhaustion.

A **Supervisor** meta-agent monitors health across the pipeline, intervening when error rates spike — it can diagnose systemic issues and patch the adapter's domain knowledge on the fly.

### Tools

Every agent has access to the same core tool set:

| Tool | Description | Usage |
|------|-------------|-------|
| `shell_exec` | Full Unix shell access — write code, install packages, run training, manage jobs | ~50% of all tool calls |
| `web_search` | Search the web for papers, documentation, and prior work | Heavy in early phases |
| `spawn_agent` | Launch a sub-agent with its own context window and full tool access — enables recursive delegation | Used for complex subtasks |
| `view_image` | Read plots and visualizations the agent generates | Used to inspect results |

Additional tools include `read_file`, `grep_file`, `propose_experiment`, `report_to_user`, and adapter-specific tools for reading/patching domain configuration. The full tool set is defined in [`src/alpha_lab/tools.py`](src/alpha_lab/tools.py).

### Key Results (from the paper)

| Domain | Task | Best Result | vs. Baseline |
|--------|------|-------------|-------------|
| **CUDA Kernels** | Write optimized GPU kernels vs. `torch.compile` | **4.4x** mean speedup (up to **91x**) | Outperforms compiled PyTorch on 83% of tasks |
| **LLM Pretraining** | Minimize val BPB under 20-min budget, <100M params | **0.7578** BPB (Opus) | 22% lower loss than single-shot baseline |
| **Traffic Forecasting** | 24h ahead road occupancy (862 sensors) | **0.0214** RMSE (Opus) | 25% better than seasonal baseline |

Each campaign costs $150-200 in LLM API calls and completes in 12-48 hours on 4x H100 hardware. The two models (GPT-5.2 and Claude Opus 4.6) discover qualitatively different solutions in every domain — neither dominates uniformly — suggesting that multi-model campaigns provide complementary search coverage.

For detailed architecture documentation, see [DETAILS.md](DETAILS.md).

---

## 🚨🚨🚨 SAFETY WARNING 🚨🚨🚨

> **⚠️ READ THIS BEFORE RUNNING ⚠️**
>
> AlphaLab runs an LLM **in a loop** as **you** on your machine.
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

### 1. Clone and install

```bash
git clone https://github.com/morganstanley/MSML.git
cd MSML/projects/alpha-lab

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install PyTorch for your hardware (see https://pytorch.org/get-started/locally/)
pip install torch  # CPU only
# pip install torch --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — add your API key:
#   OPENAI_API_KEY=sk-...        (for OpenAI)
#   or ANTHROPIC_API_KEY       (for Claude)
source .env
```

### 3. Run the demo

A working demo is included using synthetic exchange rate data. Pick the config that matches your hardware:

**CPU only** (no GPU needed — runs tree-based models like XGBoost, LightGBM, Random Forest):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_cpu_only.json --workspace ./workspace_demo
```

**Single GPU** (one GPU, experiments run sequentially):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_single_gpu.json --workspace ./workspace_demo
```

**Multi-GPU** (4x GPUs in parallel — the full experience):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_config.json --workspace ./workspace_demo
```

All three run the full pipeline autonomously:
- **Phase 0**: Customize the domain adapter for your data (~5 min)
- **Phase 1**: Explore the dataset, write scripts, generate plots, build a report (~30-90 min)
- **Phase 2**: Build an evaluation framework with tests (~20-60 min)
- **Phase 3**: Run experiments with different ML models (~1-3 hours depending on hardware)

### 4. Watch it work — Web Dashboard

**In a separate terminal:**

```bash
# First time only — build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the dashboard
python serve.py --workspace ./workspace_demo --port 8000
# Open http://localhost:8000
```

The dashboard is a passive viewer — it streams live events from the running pipeline. You can start it before, during, or after a run.

---

## Reproducing Paper Results

The paper evaluates AlphaLab on four datasets. Each has a config and download instructions.

### Download datasets

```bash
pip install huggingface_hub  # needed for Traffic and LLM Speedrun

# Download all datasets
python data/download_datasets.py

# Or download individually
python data/download_datasets.py --dataset traffic
python data/download_datasets.py --dataset llm_speedrun
python data/download_datasets.py --dataset kernelbench
python data/download_datasets.py --dataset exchange  # already included
```

### Run experiments

Each dataset has a pre-configured JSON. Adjust `gpu_ids` and `worker_count` for your hardware.

| Dataset | Config | Domain | Metric | GPUs |
|---------|--------|--------|--------|------|
| Exchange Rates | `data/demo_exchange_config.json` | time_series | Sharpe ratio (maximize) | 4x |
| Traffic | `data/paper_traffic_gpt.json` | time_series | RMSE (minimize) | 4x |
| LLM Speedrun | `data/paper_llm_speedrun_gpt.json` | llm_speedrun | val BPB (minimize) | 4x |
| CUDA KernelBench | `data/paper_cuda_kernelbench_gpt.json` | cuda_kernel | GFLOPS (maximize) | 4x |

```bash
# Exchange rates (synthetic, included)
python run.py --config data/demo_exchange_config.json --workspace ./results/exchange

# Traffic forecasting
python run.py --config data/paper_traffic_gpt.json --workspace ./results/traffic

# LLM pretraining speedrun
python run.py --config data/paper_llm_speedrun_gpt.json --workspace ./results/llm_speedrun

# CUDA kernel optimization
python run.py --config data/paper_cuda_kernelbench_gpt.json --workspace ./results/cuda
```

### Using Claude instead of GPT

Change `provider` and `model` in any config:

```json
{
  "provider": "anthropic",
  "model": "us.anthropic.claude-sonnet-4-20250514",
  "reasoning_effort": "medium"
}
```

Requires ANTHROPIC_API_KEY in your `.env`.

---

## Running Your Own Experiments

### Use an agent to set up your config

This codebase was largely built with AI coding agents, and while it aims to be plug-and-play, it may need some tweaking for your setup. We recommend using an AI coding agent:

1. Open [Claude Code](https://claude.ai/code) (or your preferred agent) in this repo
2. Prompt it to **explore the repository and become an expert in it**
3. Tell it: **where your data is**, **what you want to do**, **which model**, **what GPUs you have**
4. Ask it to **write a config and run script** for you

### Config format

```json
{
  "data_path": "path/to/your/data.csv",
  "description": "What this dataset is...",
  "target": "What to predict/optimize...",
  "provider": "openai",
  "model": "gpt-5.2",
  "domain": "",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "phase3": {
      "executor": "local",
      "max_experiments": 50,
      "gpu_ids": [0, 1, 2, 3],
      "max_per_gpu": 1,
      "worker_count": 4,
      "time_limit_seconds": 21600,
      "python_executable": ""
    }
  }
}
```

### Hardware configuration

The key settings to adjust for your hardware:

| Setting | What it does | Examples |
|---------|-------------|---------|
| `gpu_ids` | Which GPUs to use. Set `[]` for CPU-only mode | `[0]` (one GPU), `[0,1,2,3]` (four), `[]` (CPU only) |
| `max_per_gpu` | Experiments per GPU (1 = exclusive, 2+ = packing) | `1` for large models, `2` if models fit |
| `worker_count` | Number of LLM worker agents running in parallel | Usually matches number of GPUs |
| `max_experiments` | Total experiments before stopping | `10` for quick test, `50` for full run |
| `time_limit_seconds` | Kill experiments exceeding this | `3600` (1h), `21600` (6h) |
| `cpu_enabled` | Run tree-based models on CPU in parallel with GPU | `true` (default) |
| `cpu_max_parallel` | Max concurrent CPU experiments | `4` |
| `python_executable` | Python binary for experiment subprocesses | Empty = uses `ALPHALAB_PYTHON` env var or `sys.executable` |

**Domain options:** `""` for time series (default), `"cuda_kernel"`, `"nanogpt"`, `"llm_speedrun"`, or any free-text description to generate a custom adapter from scratch.

**Provider options:** `"openai"` (GPT-5.2) or `"anthropic"` (Claude). Set the corresponding API key in `.env`.

---

## Citation

If you use AlphaLab in your research, please cite:

```bibtex
@article{alphalab2026,
  title={AlphaLab: Autonomous Research Agent for End-to-End Machine Learning Experimentation},
  author={Morgan Stanley Machine Learning Research},
  year={2026},
  note={Technical Report}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
