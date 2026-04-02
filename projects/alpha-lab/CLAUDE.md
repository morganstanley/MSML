# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpha Lab is an autonomous research agent that explores datasets, generates analysis scripts, builds evaluation frameworks, and runs GPU-scale experiments — all without human intervention.

**Current state:** Full 4-phase pipeline (Phase 0 → 1 → 2 → 3) with local GPU executor. Supports OpenAI and Anthropic Claude (via Anthropic API). Domain-agnostic via adapter system — ships with built-in adapters for time series prediction, CUDA kernel optimization, and NanoGPT speed competition.

## Commands

```bash
# Run the full pipeline (headless) — OpenAI, default time_series domain
python run.py --config data/exchange_config.json --workspace ./workspace

# Run the full pipeline (headless) — Claude via Anthropic API
python run.py --config data/demo_exchange_config.json --workspace ./workspace_claude

# Run with a specific domain (copies built-in, then customizes via LLM)
# Set "domain": "cuda_kernel" or "domain": "nanogpt" in config JSON

# Run with a novel domain (triggers Phase 0 agent to generate adapter from scratch)
# Set "domain": "your task description here" in config JSON

# Run the web dashboard
python serve.py --config data/exchange_config.json --workspace ./workspace

# Generate synthetic test data
python data/generate_synthetic.py
```

Requires Python 3.11+ with dependencies from requirements.txt.

## Architecture

### Domain Adapter System (`adapter.py`, `adapter_loader.py`, `adapters/`)
Every domain-specific aspect of the pipeline is parameterized by a `DomainAdapter`:
- **Prompts**: 9 prompt `.md` files (one per phase/role), plus `domain_knowledge.md`
- **Metrics**: `MetricConfig` — primary_metric, direction (maximize/minimize), display_name
- **Experiment structure**: `ExperimentStructure` — required_files, entry_point, framework_dir

Three built-in adapters ship under `src/alpha_lab/adapters/`:
- **time_series** — Sharpe ratio (maximize), walk-forward backtesting
- **cuda_kernel** — throughput_gflops (maximize), benchmark framework
- **nanogpt** — wall_clock_seconds (minimize), training framework

Adapter resolution priority: workspace adapter > built-in matching domain > time_series fallback.

### Phase 0: Adapter Resolution & Customization (`phase0.py`)
Runs before Phase 1 to resolve, customize, or generate the domain adapter:
1. **Resume path**: `{workspace}/adapter/manifest.json` exists → load and return (already customized)
2. **Built-in match**: domain matches built-in name → copy template → run customization agent → return
3. **Default path**: no domain specified → copy time_series template → run customization agent → return
4. **Generation path**: free-text domain → run full generation agent with `write_adapter_file` + `read_reference_adapter` tools

The customization agent (`_run_customization_agent`) examines the actual data/task and patches adapter files to be task-specific. Uses `read_adapter`, `shell_exec`, `read_file`, and `patch_adapter_file`. Highest-value target is `domain_knowledge.md` (injected into every phase). Runs at `reasoning_effort="medium"`.

### Supervisory Agent (`supervisor.py`)
Meta-agent that monitors pipeline phases and can patch the adapter:
- `validate_adapter()` — after Phase 0 (always, for all domains): checks completeness, validity, prompt quality
- `review_phase1()` — after Phase 1: checks learnings, data report, scripts
- `review_phase2()` — after Phase 2: checks framework, tests, review verdict
- `phase3_health_check()` — during Phase 3: triggered when error rate > 40%, diagnoses systemic issues and patches adapter files (with git checkpoint)

### Provider System (`provider.py`, `provider_openai.py`, `provider_anthropic.py`)
All LLM calls go through the `Provider` protocol. Two implementations:
- **OpenAIProvider**: Wraps OpenAI Responses API. Built-in `web_search_preview`.
- **AnthropicProvider: Wraps Anthropic Messages API (Claude). Translates tool schemas from OpenAI format. Web search proxied through GPT.

The `get_provider()` factory in `client.py` returns the right provider based on config.

### Agent Loop (`agent.py`)
Provider-agnostic iterative loop: call `provider.stream_response()` → collect text + tool calls → dispatch tools → feed results back via `provider.build_tool_result_items()`. Conversation history tracked locally in `_input_history`. Accepts optional `adapter` param, passed to `execute_tool()`. Continues until `report_to_user` is called or `ask_user` returns control.

### Client (`client.py`)
Factory for providers:
- **OpenAI**: Standard OpenAI API with `OPENAI_API_KEY`
- **Anthropic: Anthropic Messages API with ANTHROPIC_API_KEY

### Local GPU Executor (`local_gpu.py`)
Replaces SLURM for single multi-GPU boxes. Same 5-method interface as `SlurmManager`:
- Spawns experiments as subprocesses with `CUDA_VISIBLE_DEVICES` pinning
- Polls `proc.poll()` for status
- Enforces time limits

### Tool System (`tools.py`)
Function tools + web search. Tool dispatch is a flat if/elif in `execute_tool()`. Each tool returns a dict with `"output"` (string for API), optional `"image"` (base64 tuple for injection), and optional `"done"` flag. The `web_search` tool proxies queries through GPT when using Anthropic (since Anthropic API does not have built-in web search).

Four adapter tools added for Phase 0 and Supervisor:
- `write_adapter_file` — write a file to `{workspace}/adapter/`
- `read_reference_adapter` — read a built-in adapter for format reference
- `read_adapter` — read current workspace adapter files
- `patch_adapter_file` — overwrite an adapter file (with git checkpoint)

### Context Management (`context.py`)
Local conversation history tracking with character-based token estimation (tiktoken fallback for offline). Uses `provider.complete()` for summarization. Accepts `domain_description` to parameterize the summarization prompt.

### System Prompt (`prompts.py`)
Instructs the agent to work autonomously through structured phases. When an adapter is provided, uses adapter-specific prompts and injects domain knowledge. Falls back to built-in `PROMPT_REGISTRY` for backward compatibility. Workspace path and accumulated learnings are injected dynamically. Uses `python` directly (not uv).

### Key Design Patterns
- Provider protocol abstracts OpenAI vs Anthropic — tool schemas written in OpenAI format, Anthropic provider translates
- Domain adapter abstracts time_series vs cuda_kernel vs nanogpt vs custom — prompts, metrics, and file structure all parameterized
- ZDR mode: local history tracking, no server-side conversation storage
- All shell commands run in workspace directory via `subprocess.run`
- Default model: `gpt-5.2` (OpenAI) or `us.anthropic.claude-sonnet-4-20250514` (Anthropic)
- Python executable configurable via `ALPHALAB_PYTHON` env var or `python_executable` config field
- Convergence direction: `maximize` domains track `> best`, `minimize` domains track `< best`

## Four-Phase Pipeline

0. **Phase 0**: Resolve and customize domain adapter (built-ins are customized for the actual task, novel domains generated from scratch)
1. **Phase 1**: Single agent explores dataset, writes scripts, generates plots, builds research report
2. **Phase 2**: Multi-agent pipeline (Builder/Critic/Tester) creates domain-appropriate evaluation framework
3. **Phase 3**: Dispatcher orchestrates Strategist + Workers to run dozens of GPU experiments

Supervisor reviews output between each phase and monitors Phase 3 health.

## Config

JSON config (YAML also supported if pyyaml installed):

```json
{
  "data_path": "data/exchange_rates.csv",
  "description": "...",
  "target": "...",
  "provider": "openai",
  "model": "gpt-5.2",
  "reasoning_effort": "low",
  "domain": "",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "phase3": {
      "executor": "local",
      "gpu_ids": [0, 1, 2, 3],
      "max_per_gpu": 1,
      "time_limit_seconds": 21600,
      "convergence_metric": ""
    }
  }
}
```

### Domain field values
- `""` (empty) — uses time_series template, customized for the actual dataset
- `"time_series"` — built-in template, customized for actual data (Sharpe ratio, walk-forward backtesting)
- `"cuda_kernel"` — built-in template, customized for actual benchmark (throughput GFLOPS)
- `"nanogpt"` — built-in template, customized for actual task (wall clock seconds, minimize)
- `"free text description"` — triggers Phase 0 agent to generate a custom adapter from scratch

### Convergence metric
- `""` (empty) — uses adapter's primary_metric (recommended)
- Any string — overrides the adapter's metric for convergence tracking

Set `"provider": "anthropic"` and `"model": "us.anthropic.claude-sonnet-4-20250514"` for Claude.
