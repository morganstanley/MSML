# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Alpha Lab is an autonomous multi-agent system for machine-learning research. Given a dataset and a task, it explores the data, builds an evaluation framework, and runs dozens of experiments on GPUs — all without human intervention. A run proceeds through four phases (0 → 1 → 2 → 3) with a Supervisor reviewing output between them:

- **Phase 0** — resolve the domain [adapter](docs/03_adapters.md) (customize a built-in or generate one from scratch).
- **Phase 1** — one Explorer agent profiles the raw dataset (`learnings.md`, `data_report/`).
- **Phase 2** — a Builder / Critic / Tester loop builds the domain evaluation framework.
- **Phase 3** — a Strategist + Workers propose, run, and analyze experiments until the metric converges or `max_experiments` is hit.

The system is domain-agnostic via the adapter system, supports OpenAI (default), Anthropic Claude (natively or via AWS Bedrock), xAI grok, and local vLLM models, runs on-prem through the MS AI Gateway (no API key; token/Kerberos auth), and keeps no server-side conversation history (ZDR — history is tracked locally).

**The `README.md` and `docs/` set are the canonical, verified documentation.** Read them before answering conceptual questions or making design decisions; this file is the code-navigation and conventions layer on top. Keep the two consistent — if you change behavior documented in `docs/`, update the doc in the same change.

| Topic | Doc |
|-------|-----|
| Phases, executors, run lifecycle | [docs/01_overview.md](docs/01_overview.md) |
| Config fields & defaults | [docs/02_configuration.md](docs/02_configuration.md) |
| Domain adapters | [docs/03_adapters.md](docs/03_adapters.md) |
| Agent definitions, providers, sandboxing | [docs/04_agents.md](docs/04_agents.md) |
| Tool definitions & granting | [docs/05_tools.md](docs/05_tools.md) |
| Memory store | [docs/06_memory.md](docs/06_memory.md) |
| Workspace on-disk layout | [docs/07_workspace.md](docs/07_workspace.md) |
| Benchmarks / evaluation / examples / tracking | [docs/08–11](docs/) |

## Commands

```bash
# Environment (sets ALPHALAB_PYTHON; builds the venv at the path if absent).
export ON_PREM=1
source scripts/setup-venv --venv-path .venv
python -m pytest tests                 # run the test suite

# Demo: generate synthetic exchange-rate data + a ready-to-use config.json.
./scripts/generate-exchange-test-data --output-dir ./demo

# Run the full pipeline. MLflow is ON by default and the run exits at startup
# if the connection env vars are missing — source mlflow.env, or pass --no-mlflow.
source mlflow.env
# Experiment name is optional (defaults to the workspace basename); set to group runs.
export MLFLOW_EXPERIMENT_NAME="alpha-lab-demo"
python run.py --config ./demo/config.json --workspace ./demo

# Watch a run (passive; start before/during/after). Also: `alpha-lab-serve`.
python serve.py --workspace ./demo --port 8000

# Interactive intake (user-proxy interview before Phase 0).
python run.py --config ./demo/config.json --workspace ./demo --enable-intake
```

`run.py` and `serve.py` at the repo root are thin wrappers that add `src/` to `sys.path` and call into the package. Installed console scripts (see `pyproject.toml [project.scripts]`): `alpha-lab` (chat REPL + intake), `alpha-lab-run`, `alpha-lab-serve`, `alpha-lab-memory`, `alpha-lab-evaluate`, `generate-synthetic-data`, and the `alpha-lab-*-benchmark(s)` / suite scripts.

`$ALPHALAB_PYTHON` (set by `setup-venv`) is the Python used for **Phase 3 experiment subprocesses** — it must have PyTorch/NumPy/Pandas and the ML deps. It is independent of whatever `python` runs the orchestrator; scripts under `scripts/` use `${ALPHALAB_PYTHON:-python3}`.

## Module map (`src/alpha_lab/`)

**Orchestration**
- `run.py` — `run_main`: top-level pipeline. Validates MLflow env, runs Phase 0, then the selected phases, invoking the Supervisor between them. (Root `run.py` wraps this.)
- `pipeline.py` — `Pipeline`: the Phase 2 Builder → Critic → Tester loop; `detect_phase1_complete`.
- `cli.py` — the `alpha-lab` chat REPL and the interactive intake session (`run_intake`).
- `server.py` / `serve.py` — FastAPI web dashboard.
- `config.py` — `TaskConfig`, `PipelineConfig`, `Phase3Config` (all config fields + defaults).
- `constants.py` — `Phase` enum and shared constants.
- `deps.py` — `RunDeps`: the run-scoped container (config, workspace, executors, memory store) published as a module global for the run's duration and read via `deps.get()`; `utils.py` holds the capacity readers over it. Tool implementations reach live run state through this seam.

**Phase 0 & adapters**
- `phase0.py` — `run_phase0`: resolve / customize / generate the adapter.
- `adapter.py` — `DomainAdapter`, `MetricConfig`, `ExperimentStructure`, `PROMPT_KEYS` (the 9 required prompt files).
- `adapter_loader.py` — load and validate an adapter directory from disk.
- `adapters/` — the seven built-in adapters (`time_series`, `cuda_kernel`, `nanogpt`, `llm_speedrun`, `tabular_classification`, `tabular_regression`, `blackbox`), each a dir of `manifest.json` + prompt `.md` files.
- `supervisor.py` — cross-phase reviews (`validate_adapter`, `review_phase1`, `review_phase2`) and the Phase 3 health check (fires when the experiment error rate exceeds 40%).

**Agents, tools, prompts**
- `agent.py` — `AgentLoop`: provider-agnostic think → call tools → observe loop.
- `agents/` — `load_agent` → `AgentDefinition`; `factory.py`; `registry/` holds the markdown agent definitions (Agents.md/Skills.md frontmatter).
- `tools/` — `load_tool` / `load_tools` → `ToolDefinition`; `execution.py` (`execute_tool` dispatch); `registry/` holds the markdown tool definitions. (There is **no** top-level `tools.py`.)
- `prompts.py` — system-prompt assembly; injects adapter prompts, workspace path, and live `learnings.md`.
- `context.py` — `ContextManager`: local history tracking + summarization.
- `sandboxing/` — `sandbox.py` (bwrap confinement), `runner.py`, `db_proxy.py` (sandboxed children reach the SQLite stores through the parent).

**Providers & auth** (`providers/` package — import from `alpha_lab.providers`, not the submodules)
- `providers/types.py` — the `Provider` protocol (structural; backends don't subclass it) + the `ToolCall` / `StreamEvent` / `Response` records.
- `providers/{openai,anthropic,bedrock,grok,local}.py` — the five backends (`OpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `GrokProvider`, `LocalProvider`). Each exposes a `from_config(api_key, model)` classmethod that builds its own client. `GrokProvider` subclasses `OpenAIProvider` (xAI is OpenAI-dialect); `LocalProvider` serves local vLLM endpoints (Kimi/GLM).
- `providers/__init__.py` — the facade: exports the provider classes and `get_provider(provider_name, api_key, model)`, which maps `provider_name` through `PROVIDER_CLASSES` and calls the selected class's `from_config`. `provider_name` is `"openai"` (default), `"anthropic"`, `"grok"`, `"bedrock"`, or `"local"`; for `"local"` the `model` name selects the dialect via `local.py:_sniff_dialect` (must contain `"glm"` or `"kimi"`).
- `providers/utils/auth.py` — 3-tier bearer-token fallback (live SCV/PingFed → cached token → off-prem via `USE_OFFPREM=1` + `OPENAI_API_KEY`), on-prem detection, MS CA bundle.
- `providers/utils/clients.py` — all client construction: `get_openai_client(is_async=...)` / `get_anthropic_client` / `get_bedrock_client` / `get_grok_client` / `get_local_client` + the shared on-prem gateway builder.
- `providers/utils/scalar_2_sample_setup.py` — token setup.
- The PydanticAI model builder `get_pydantic_ai_model` (OpenAI/Bedrock only) lives in `evaluations.py` — it's used solely by the LLM judge.

**Phase 3 execution**
- `dispatcher.py` — `Dispatcher`: assigns Workers, submits jobs, polls status, handles failures, and routes each experiment CPU vs GPU (`_is_cpu_experiment`).
- `strategist.py`, `worker.py` — the Phase 3 agents.
- `local_gpu.py`, `local_cpu.py`, `slurm.py` — the executors (same small interface).
- `experiment_db.py` — the SQLite experiment board and its status lifecycle (`KANBAN_COLUMNS`).
- `experiment_validation.py`, `validation.py`, `process_control.py` — reality checks and subprocess control.

**Memory**
- `memory.py` — `Memory` (a `SQLiteModel` record) and `MemoryStore` (read/write API).
- `memory_cli.py` — the `alpha-lab-memory` CLI.
- `databases.py` — `ModelDB` (records + SQLite index + embeddings + git + locking).
- `models/` — `SQLiteModel` base, codecs. `embeddings.py` — embedding vectors and the shared vector cache. `git.py` — record versioning.

**Tracking, output, benchmarks**
- `mlflow_logger.py` — MLflow logging (the default; `missing_required_env` drives the startup gate; mTLS auth).
- `tracing.py` — OpenTelemetry fallback, active only under `--no-mlflow`; no-op with no exporter.
- `events.py`, `metrics.py` — the structured event stream and `MetricsCollector`.
- `output_generator.py` — `OutputGenerator`: deterministic `output/` documents (no LLM).
- `benchmarks/` — benchmark suite creation/running (the `alpha-lab-*-benchmark(s)` scripts).
- `evaluations.py` — `alpha-lab-evaluate` (scores how well the run was tailored).
- `generate_synthetic_data.py` — synthetic dataset generation.

## Things to get right when editing

- **Adapters don't control experiment launch.** All three executors launch every experiment by running `run_experiment.py` and mark it done only when `results/metrics.json` exists — regardless of the manifest's `entry_point` / `results_dir` / `results_file` (those feed only the CPU/GPU routing scan and metric extraction). See [docs/03_adapters.md](docs/03_adapters.md). Don't "fix" an adapter by renaming its entry file to match the manifest.
- **CPU/GPU routing** is by the experiment's `resource` field (`"cpu"`/`"gpu"`) if set, else a source scan for GPU markers, defaulting to GPU when unsure (`dispatcher.py:_is_cpu_experiment`).
- **Config defaults** worth knowing: `jit=True` (just-in-time proposals) and `handoff=True` (each analyzed experiment gets a user-proxy handoff turn). Keep `config.py` and [docs/02_configuration.md](docs/02_configuration.md) in sync, and pin flags in `tests/` fixtures that depend on them.
- **Agent and tool definitions are data, not code** — markdown files under `agents/registry/` and `tools/registry/`. An agent's `allowed-tools` list is what bounds it; tool implementations live in `tools/execution.py`. Validate agent frontmatter with `scripts/validate_agents_frontmatter.py`.
- **Bedrock reasoning** (`providers/bedrock.py`) has two paths keyed on the Claude family version (`_supports_adaptive`): Claude `>= 4.7` passes `reasoning_effort` to `output_config.effort` with adaptive thinking (the model decides how much to reason); older Claudes (`< 4.7`) map each tier to a fixed `budget_tokens`. The default model is `gpt-5.2` (OpenAI); set `provider: "bedrock"` (or `"anthropic"` for the native Messages API) + a Claude model id for Claude.
- **Local models** use `provider: "local"` (the old `"kimi"` / `"glm"` provider values now raise `ValueError`). Set `LOCAL_BASE_URL` to the vLLM endpoint; the config `model` is sent verbatim and its `"glm"` / `"kimi"` substring picks the dialect (`providers/local.py:_sniff_dialect`). `GLM_NATIVE_TOOLS=1` selects native tool-calling for GLM-5.2 (default off keeps GLM-5.1's text-tool workaround). Bedrock's endpoint env var is `BEDROCK_BASE_URL` (was `AIGW_BEDROCK_ENDPOINT`).
- **Convergence direction** follows the adapter metric: `maximize` domains track `> best`, `minimize` domains track `< best`.
- The `benchmarking/` directory at the repo root is **deprecated**; the live benchmark code is `src/alpha_lab/benchmarks/`.
