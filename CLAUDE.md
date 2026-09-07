# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpha Lab is an autonomous research agent that explores datasets, generates analysis scripts, builds evaluation frameworks, and runs GPU-scale experiments — all without human intervention.

**Current state:** Full 4-phase pipeline (Phase 0 → 1 → 2 → 3) working on-prem with local GPU + parallel CPU executors. Providers: OpenAI, Anthropic Claude (via Bedrock), xAI grok, and lab-hosted Chat Completions endpoints (Kimi, GLM). ZDR-compatible (no server-side conversation storage). Domain-agnostic via adapter system — ships with built-in adapters for time series prediction, CUDA kernel optimization, NanoGPT speed competition, and LLM pretraining speedrun.

**Safety:** the pipeline runs an LLM in a loop with full shell access as the invoking user. Run it in an isolated workspace, never a home directory (see the README warning). `workspace*/` directories are runtime-generated and gitignored — never commit them.

Architecture deep-dive lives in `DETAILS.md`; observability guide in `MLFLOW.md`.

## Invariants

(From `.github/copilot-instructions.md` — preserve these in every change.) The core pipeline knows how to *do research*; it must not know about any specific problem. When in doubt: the **pipeline** gains capabilities; the **adapter** gains domain knowledge.

- Never leak problem-specific logic (finance, CUDA, NanoGPT, forecasting) into the core pipeline, agent loop, prompts, providers, or tools. If it describes "what good looks like" for a task, it belongs in an adapter.
- Never couple to a single LLM provider — tool schemas are authored in OpenAI format; the other providers translate. New features must work through all providers.
- Never couple to a single executor — `local_gpu.py`, `slurm.py`, `local_cpu.py` share a 5-method interface; honor it.
- Preserve resumability: Phase 0 manifests, the experiment DB, and the workspace layout are designed so a run can be stopped and resumed.

## Commands

Python 3.11+ required. Install in a venv:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .                         # exposes alpha-lab / alpha-lab-run / alpha-lab-serve console scripts

cp .env.example .env                     # set ALPHALAB_PYTHON to a python with torch/numpy/pandas
source .env
```

```bash
# Generate synthetic test data (one-off, no internet)
python data/generate_synthetic.py

# Run the full pipeline (headless)
python run.py --config data/demo_exchange_config.json --workspace ./workspace_demo

# Bedrock/Claude: set "provider": "bedrock" in the config JSON, then run as above.
# Domain: set "domain" in config to "" (default: time_series), "cuda_kernel", "nanogpt",
# "llm_speedrun", or any free-text description (triggers Phase 0 generation agent).

# Web dashboard (passive viewer — start before/during/after a run, points at same workspace)
cd frontend && npm install && npm run build && cd ..   # one-off
python serve.py --workspace ./workspace_demo --port 8000

# Tests
pytest tests/                            # whole suite (needs `pip install -e .` first)
pytest tests/test_dispatcher.py          # one file
pytest tests/test_dispatcher.py::TestX   # one class/test
```

CI on PR builds runs only `python -m compileall -q src/` — there is no test execution gate, so run pytest locally (~25s, green on main; do not pass `--timeout`, the plugin isn't installed). The merge gate (`.github/pull_request_requirements.yml`) additionally requires the PR title to carry a Jira key `PROJ-<n>` whose status is To Do/In Progress, plus ≥1 successful build-type check.

`requirements.txt` is the full pinned dependency set (incl. torch+CUDA); `pyproject.toml` lists only the minimal runtime core. Frontend needs Node 18+ (React 19, Vite 6, TypeScript; `npm run build` runs `tsc && vite build`).

`$ALPHALAB_PYTHON` controls the python used for Phase 3 experiment subprocesses (falls back to `sys.executable`); scripts under `scripts/` read `${ALPHALAB_PYTHON:-python3}`. The Phase-3 python needs torch/numpy/pandas; the orchestrator python only needs the `requirements.txt` deps.

### On-prem auth (AI Gateway)

Token bootstrap before any run. Two terminals:

```bash
# Terminal A (your account) — create the cache file the daemon will write to
touch .token_cache.json && chmod 666 .token_cache.json

# Terminal B (proid, separate tmux) — long-running token refresh daemon
tmux new -s token
suu -tr "<ticket>" <proid>
cd /full/path/to/repo                    # absolute path, not ~
export ALPHALAB_PYTHON=<your python with httpx>
./scripts/token_refresh.sh               # refreshes every 45 min; detach with Ctrl+b d
```

Off-prem: skip the daemon, set `USE_OFFPREM=1` and `OPENAI_API_KEY=...`.

The 3-tier auth chain in `client.py` is: in-memory cache → `.token_cache.json` on disk → live SCV/PingFed (needs proid Kerberos).

### Post-hoc framework comparison (`src/alpha_lab/benchmarks/runcmp/`)

Compares finished runs from any number of harnesses (in-house frameworks and
external evidence-contract submissions), working strictly off post-run logs
(never writes into an input run). **The manual is
`src/alpha_lab/benchmarks/runcmp/README.md`** (pipeline, viewer, evidence
contract, adding domains/harnesses, LLM reviewers);
`src/alpha_lab/benchmarks/runcmp/DESIGN.md` keeps the design rationale.

```bash
OUT=comparison_out
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp index    --root <corpus_dir> --out $OUT/corpus.json
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp extract  --corpus $OUT/corpus.json --out $OUT/packs --workers 4
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp tabulate --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT \
    [--pair "name=LEFT_LABEL:RIGHT_LABEL"]      # extra cross-era pairs
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp referee  --corpus $OUT/corpus.json --packs $OUT/packs \
    --out $OUT/referee.json --pair "name=L:R"   # one-evaluator re-scoring of preserved predictions
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp bench    --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT \
    --referee $OUT/referee.json                 # deterministic metric registry -> bench.md/json
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp investigate --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT \
    --provider openai --model gpt-5.6-sol [--mission @file]   # optional agentic layer; needs LLM auth
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp recompose --out $OUT/review --corpus $OUT/corpus.json \
    --packs $OUT/packs --provider <p> --model <m>   # re-write ONLY the report from a review's frozen findings
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp factcheck --out $OUT --packs $OUT/packs   # + findings.md; exit 0
                                                    # only when every hard audit is clean, else exit 1
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp publish --corpus $OUT/corpus.json --packs $OUT/packs \
    --bench $OUT/bench.json --referee $OUT/referee.json --out $OUT \
    --store <showcase_dir>                      # export cells + lens charts to the showcase MLflow store
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp lineup --list        # adopted benchmark domains
PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp lineup --out <campaign_dir> \
    --domains d5_rfq,d6_cuda --models "glm=glm:glm-5.2,o48=bedrock:claude-opus-4-8"   # emit per-cell configs
```

The benchmark domain set lives in `runcmp/lineup.json` (detection tokens,
dataset path, task text with evidence contract, per-framework domain field,
referee spec). Adding a dataset = one JSON entry; new code only for a new
referee kind. Adopted beyond d2/d4: `d5_rfq` (ETF RFQ win/loss, log-loss on
a frozen 2025-10-01+ holdout, referee kind `classification_table`) and
`d6_cuda` (fused GEMM+bias+GELU TFLOP/s, correctness-gated, referee kind
`kernel_bench`; task contract in `runcmp/tasks/d6_cuda/TASK.md`).

The primary path is deterministic end to end: `corpus.py` (registry;
row-level DB completeness — a present-but-empty `experiments.db` is not a
usable run, and a newer zero-byte DB must not shadow the populated one) →
`extract.py` (streamed evidence packs; `api_request`/`agent_text` event lines
are counted but never JSON-parsed; reads cond `meta/` Conductor artifacts msml
tooling ignores; per-tool failure-signature histograms) → `tabulate.py` (pair +
corpus tables) → `referee.py` (aligns evaluation origins across runs, builds a
consistency-verified truth pool so forecast-only artifacts are scorable, and
re-scores every experiment under one explicit metric — the only
cross-framework model-quality evidence) → `bench.py` (versioned metric
registry, ~46 metrics × runs, framework rollups, and failure signatures
counted + classified via `rules.json` into bug / design / external / policy).
`investigate.py`/`factcheck.py` are the optional judgment layer: a single LLM
investigator whose findings carry machine-checkable evidence references,
validated at record time and re-verified deterministically (`findings.md` is
the readable rendering; factcheck's exit status is the verdict — 0 only when
every hard audit is clean). `investigate_team.py` adds planner/executor/critic
seats; the critic gates findings AND scores every report draft 0–100; the
draft with the fewest hard audit violations (best score among those) publishes
even if the writer dies mid-loop. `recompose.py` re-writes only the report
from a finished review's frozen findings — probe files are append-only and
fingerprint-checked at exit, so the cited evidence provably survives.
Tests: `pytest tests/test_runcmp.py`.

## Architecture

### Benchmark suites (`src/alpha_lab/benchmarks/`)
`run_benchmarks.py` runs import-resolved benchmark generators against import-resolved runners (`module:object` paths). `runners.py` provides `LocalRunner` and `MLflowRunner` (groups a suite's pipeline runs under a single MLflow Suite Run).

### Observability (`mlflow_logger.py`, `tracing.py`)
Two mutually exclusive, strictly opt-in backends — the pipeline runs without either:
- **MLflow** (preferred): `--mlflow` flag + `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME`. Logs one Pipeline Run per invocation (phase artifacts at each boundary), nested per-experiment sub-runs, and per-agent LLM traces via `mlflow.start_span`. See `MLFLOW.md`.
- **OTel/Tempo** (legacy): used when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and `--mlflow` is not passed.

### Domain Adapter System (`adapter.py`, `adapter_loader.py`, `adapters/`)
Every domain-specific aspect of the pipeline is parameterized by a `DomainAdapter`:
- **Prompts**: 9 prompt `.md` files (one per phase/role), plus `domain_knowledge.md`
- **Metrics**: `MetricConfig` — primary_metric, direction (maximize/minimize), display_name
- **Experiment structure**: `ExperimentStructure` — required_files, entry_point, framework_dir

Four built-in adapters ship under `src/alpha_lab/adapters/`:
- **time_series** — Sharpe ratio (maximize), walk-forward backtesting
- **cuda_kernel** — throughput_gflops (maximize), benchmark framework
- **nanogpt** — wall_clock_seconds (minimize), training framework
- **llm_speedrun** — val BPB (minimize), LLM pretraining harness with wall-clock + param budgets

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

### Verifier (`verifier.py`, `scripts/verify_workspace.py`)
Standalone post-run audit of a finished workspace's claimed findings. Three cooperating agents — **User-Rep** (selects candidates + arbitrates), **Worker** (independently re-implements the finding), **Critic** (attacks it) — all applying one test: *would a skeptical human be persuaded?* Ground rule: see everything in the workspace, import nothing from it — the value is an independent reconstruction from raw data. Output unit is an executed Jupyter notebook built via `scripts/nb_run.py`. Read-only on the workspace except its `verify/` subtree. Not yet folded into the pipeline; drive it standalone:

```bash
PYTHONPATH=src python scripts/verify_workspace.py --config data/verify_etfflow_o48_config.json
```

### Conductor (`conductor.py`, `conductor_tools.py`, `meta_layout.py`)
Top-level meta-agent that represents the user and steers the whole pipeline with a light hand. Runs after each Phase 0/1/2 completes, after each Phase 3 milestone, and on a slow timer between milestones. Reads everything (proposals, code, debriefs, milestones, its own past decisions, user instructions) and intervenes through three channels: directives written to `meta/directives.md`, leaderboard annotations in `meta/annotations.json`, and queue mutations (`park`/`unpark`/`set_priority`/`set_throttle`). Plus, with Python-verified evidence, phase rewinds.

The Conductor's authority is bounded — it does not propose experiments, write framework code, or edit prompts. It influences other agents by writing files they read at the top of their next turn. A `phase3.no_conductor=true` flag is the NOOP fallback: the dispatcher skips the conductor hook and the system reverts to its pre-Conductor behavior. The strategist's `cancel_experiments` tool is removed in default mode (Conductor administers parking) and restored in NOOP mode.

`meta/` filesystem layout (auto-created when conductor is enabled):
- `meta/directives.md` — current Conductor-issued directives, read by every other agent
- `meta/annotations.json` — `{exp_id: label}` map (champion / control / quarantined / exploration / exploitation / ensemble-candidate / home-run-attempt)
- `meta/instructions/from_user.md` — write here to instruct the Conductor; it diffs against `.last_seen` each turn
- `meta/instructions/ack.md` — Conductor acknowledgements of user instructions
- `meta/notes_to_user.md` — Conductor's running record (user reads when they choose; system never alerts)
- `meta/notes_inbox.md` — strategist / workers leaving notes for the Conductor (`note_to_conductor` tool)
- `meta/meta_log.jsonl` + `meta_log.md` — append-only audit trail of every Conductor decision (reason + evidence)
- `meta/throttle.json` — `{"gpu": "none|slow|halt-new", "cpu": ...}`; dispatcher consults this in `_submit_checked`
- `meta/scratch/` — Python analysis scripts the Conductor writes; required for kill / phase-rewind evidence
- `meta/backups/<ts>/` — pre-overwrite snapshots from `delete_path` and phase rewinds
- `meta/taxonomy.json` — cached mechanism-class taxonomy (Conductor maintains incrementally)

DB schema additions: `priority INTEGER DEFAULT 0` (queue ordering: priority DESC, created_at ASC) and `parked_at REAL` (NULL = active; non-NULL = soft-cancelled, dispatcher skips). Migration is idempotent; existing DBs upgrade silently on first open.

### Provider System (`provider.py`, `provider_openai.py`, `provider_bedrock.py`, `provider_grok.py`, `provider_chat.py`)
All LLM calls go through the `Provider` protocol. The `get_provider()` factory in `client.py` accepts `"openai"`, `"bedrock"`, `"grok"`, `"kimi"`, or `"glm"`:
- **OpenAIProvider** (`"openai"`): Wraps OpenAI Responses API. Built-in `web_search_preview`.
- **BedrockProvider** (`"bedrock"`): Wraps AWS Bedrock Converse API (Claude). Translates tool schemas from OpenAI format to Bedrock `toolSpec`.
- **GrokProvider** (`"grok"`): xAI's OpenAI-compatible gateway; subclasses OpenAIProvider. Drops reasoning items from next-turn input (xAI rejects modified "compaction blobs" with HTTP 400) and clamps reasoning effort into grok's `minimal`..`xhigh` vocabulary (`none`→`minimal`, `max`→`xhigh`).
- **ChatProvider** (`"kimi"` / `"glm"`): any OpenAI-compatible `POST /v1/chat/completions` lab endpoint (base URLs via `KIMI_BASE_URL` / `GLM_BASE_URL`). Per-model thinking dialects (Kimi: graded `thinking_budget`; GLM: binary `enable_thinking`). Every chat model replays its reasoning into the next request's history by default (`CHAT_REPLAY_REASONING=0` opts out; kimi-k3 replays unconditionally per its model card) — note the endpoint's chat template must also render the field: kimi's does, deepseek's currently drops it. GLM-5.1 requires text-based tool calling (native `tools` degenerates it); `GLM_NATIVE_TOOLS=1` selects the native path for GLM-5.2, which also routes image turns to opus (GLM is text-only) — via the native Anthropic gateway by default; `GLM_VISION_PROVIDER` selects `anthropic`/`bedrock`/`none`, with the gpt-4o proxy as the final fallback.

System policy: web search always flows through OpenAI — every non-OpenAI provider proxies the `web_search` tool through a real OpenAI client.

### Agent Loop (`agent.py`)
Provider-agnostic iterative loop: call `provider.stream_response()` → collect text + tool calls → dispatch tools → feed results back via `provider.build_tool_result_items()`. Conversation history tracked locally in `_input_history`. Accepts optional `adapter` param, passed to `execute_tool()`. Continues until `report_to_user` is called or `ask_user` returns control.

### Client (`client.py`)
Factory for providers. Auto-detects on-prem vs off-prem:
- **On-prem OpenAI**: Uses the internal AI Dev Platform via `scalar_2_sample_setup.py` (PingFed/SCV/LDAP auth)
- **On-prem Bedrock**: Bearer token auth via PingFed through the internal AI Gateway
- **Off-prem**: Standard OpenAI API with `OPENAI_API_KEY`

Set `USE_OFFPREM=1` to force off-prem mode.

### Local GPU Executor (`local_gpu.py`) and CPU Executor (`local_cpu.py`)
`LocalGPUManager` replaces SLURM for single multi-GPU boxes. Same 5-method interface as `SlurmManager` (`submit_experiment` / `poll_jobs` / `cancel` / `can_submit` / `running_gpu_count`):
- Spawns experiments as subprocesses with `CUDA_VISIBLE_DEVICES` pinning, polls `proc.poll()`, enforces time limits
- Supports GPU packing (`max_per_gpu` > 1)

`LocalCPUManager` runs tree/linear models in parallel with the GPU pool. The dispatcher auto-routes by model type (XGBoost, LightGBM, CatBoost, RandomForest, sklearn, etc. → CPU); experiments can also set `resource: "cpu"|"gpu"` explicitly. Configured via `phase3.cpu_enabled` / `cpu_max_parallel` / `cpu_time_limit_seconds`. Swap `executor: local` ↔ `executor: slurm` and the dispatcher behaves the same — it doesn't know which is running.

### Tool System (`tools.py`)
Function tools + web search. Tool dispatch is a flat if/elif in `execute_tool()`. Each tool returns a dict with `"output"` (string for API), optional `"image"` (base64 tuple for injection), and optional `"done"` flag. The `web_search` tool proxies queries through GPT when using Bedrock (since Bedrock blocks built-in Anthropic tools).

Four adapter tools added for Phase 0 and Supervisor:
- `write_adapter_file` — write a file to `{workspace}/adapter/`
- `read_reference_adapter` — read a built-in adapter for format reference
- `read_adapter` — read current workspace adapter files
- `patch_adapter_file` — overwrite an adapter file (with git checkpoint)

### Context Management (`context.py`)
Local conversation history tracking with character-based token estimation (tiktoken fallback for offline). Uses `provider.complete()` for summarization. Accepts `domain_description` to parameterize the summarization prompt.

### System Prompt (`prompts.py`)
Instructs the agent to work autonomously through structured phases. When an adapter is provided, uses adapter-specific prompts and injects domain knowledge. Falls back to built-in `PROMPT_REGISTRY` for backward compatibility. Workspace path and accumulated learnings are injected dynamically. Uses `python` directly.

### Key Design Patterns
- Provider protocol abstracts OpenAI vs Bedrock — tool schemas written in OpenAI format, Bedrock provider translates
- Domain adapter abstracts time_series vs cuda_kernel vs nanogpt vs llm_speedrun vs custom — prompts, metrics, and file structure all parameterized
- ZDR mode: local history tracking, no server-side conversation storage
- All shell commands run in workspace directory via `subprocess.run`
- Models are config-driven, not hardcoded — tracked configs use `gpt-5.2`/`gpt-5.4` (OpenAI) and `claude-opus-4-6-v1`/`claude-opus-4-7` (Bedrock); the CLI's argparse default is `gpt-5.2`
- Python executable defaults to `sys.executable`; override via `ALPHALAB_PYTHON` env var or `Phase3Config.python_executable`
- Convergence direction: `maximize` domains track `> best`, `minimize` domains track `< best`

## Four-Phase Pipeline

0. **Phase 0**: Resolve and customize domain adapter (built-ins are customized for the actual task, novel domains generated from scratch)
1. **Phase 1**: Single agent explores dataset, writes scripts, generates plots, builds research report
2. **Phase 2**: Multi-agent pipeline (Builder/Critic/Tester) creates domain-appropriate evaluation framework
3. **Phase 3**: Dispatcher orchestrates Strategist + Workers to run dozens of GPU experiments

Supervisor reviews output between each phase and monitors Phase 3 health. The **Conductor** runs in parallel: triggered after each phase boundary and after each Phase 3 milestone, plus on a slow timer between milestones. It represents the user across the whole run, parks unpromising experiments, reorders priorities, annotates the leaderboard, issues directives that other agents read, and (rarely) requests phase rewinds. Set `phase3.no_conductor=true` to disable.

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
- `"llm_speedrun"` — built-in template for LLM pretraining quality optimization (val BPB, minimize)
- `"free text description"` — triggers Phase 0 agent to generate a custom adapter from scratch

### Convergence metric
- `""` (empty) — uses adapter's primary_metric (recommended)
- Any string — overrides the adapter's metric for convergence tracking

### Other knobs worth knowing about
- `phase3.cpu_enabled` / `cpu_max_parallel` / `cpu_time_limit_seconds` — parallel CPU executor for tree/linear models
- `phase3.no_strategist` / `no_playbook` — ablations (random proposals, no playbook accumulation)
- `phase3.no_conductor` (default false) — NOOP fallback that disables the Conductor and reverts the dispatcher to its pre-Conductor behavior
- `phase3.conductor_interval` (default 1800 = 30 min) — slow-timer interval between Conductor turns when no milestone has fired
- `conductor_reasoning_effort` (top-level, default "high") — AgentLoop reasoning effort for Conductor turns. Parallel to the main `reasoning_effort` knob; same value semantics (`none` / `low` / `medium` / `high` / `max` for Bedrock).
- `conductor_provider` (top-level, default "bedrock") and `conductor_model` (top-level, default "claude-opus-4-7") — the Conductor's deep cross-experiment audit and retrospective evaluation benefit from the strongest available model, so by default it routes to opus regardless of what the rest of the pipeline uses. Set either to `""` to inherit the main `provider` / `model`. If the secondary provider can't be built (auth, network), the dispatcher falls back to the main one with a warning rather than crashing.
- User instructions for the Conductor live in `<workspace>/meta/instructions/from_user.md` — the only place to put run-level directives. Edit before launch for baseline guidance, or edit mid-run to nudge in real time. The Conductor reads it at the top of every turn and treats whatever's there as authoritative.
- `phase3.convergence_threshold` (default 20) — stop after N experiments with no improvement
- `reasoning_effort` for Bedrock controls Claude's thinking budget: `none`/`low`/`medium`/`high` → 0/5k/16k/32k tokens

Set `"provider": "bedrock"` and pick a Claude model (e.g. `"claude-opus-4-6-v1"`) for Claude. `"grok"` routes to xAI; `"kimi"` / `"glm"` route to lab-hosted Chat Completions endpoints.

Tracked example configs live under `data/`: `demo_exchange_config.json`, `llm_speedrun_config.json`, `paper_llm_speedrun_gpt.json`, `paper_traffic_gpt.json`. The `.gitignore` blocks ad-hoc `data/*.json` configs by default — allowlist new tracked configs explicitly there.
