<div align="center">

<img src="header.png" alt="Alpha Lab" width="600">

[![Python](https://github.com/your-org/alpha-lab/actions/workflows/pr-build-python.yml/badge.svg?branch=main)](https://github.com/your-org/alpha-lab/actions/workflows/pr-build-python.yml)
[![Frontend](https://github.com/your-org/alpha-lab/actions/workflows/pr-build-frontend.yml/badge.svg?branch=main)](https://github.com/your-org/alpha-lab/actions/workflows/pr-build-frontend.yml)

</div>

---

> [!WARNING]
> **Alpha Lab runs an LLM in a loop as _you_.** Anything your user account can do,
> it can do — delete or overwrite files, execute arbitrary code, install
> packages, make network calls. It is not malicious, but it is autonomous and
> will make mistakes.
>
> Before running: use an **isolated workspace** (not your home directory) and
> **back up anything that matters**.


## What is Alpha Lab?
Alpha Lab is a multi-agent system for autonomous machine-learning research. Give it a
dataset and a task, and it takes over from there
— exploring the data, building an evaluation framework, and running dozens of experiments 
on your machines. 

## Quick start

To help you hit the ground running, we've included a lightweight toy problem based on synthetic exchange-rate data. 
Follow the steps shown below to setup and launch Alpha Lab, then watch as it does the rest!

### Setup

Fork and clone the repo, then create the virtual environment.

```bash
gh repo fork https://github.com/your-org/alpha-lab --clone
cd alpha-lab

export ON_PREM=1
source scripts/setup-venv --venv-path .venv   # builds the venv at this path if absent
python -m pytest tests                        # optional: confirm the install
```

`setup-venv` sets the `ALPHALAB_PYTHON` variable, which tells Alpha Lab which
Python to use for running GPU experiments (Phase 3). It should point to a Python
environment that has PyTorch, NumPy, Pandas, and other ML dependencies installed.
All shell scripts and configs read from this variable automatically.

Check core runtime prerequisites before starting a long run:

```bash
alpha-lab-doctor --workspace ./demo
```

The command checks Python, Pydantic compatibility, SQLite FTS5, Git, optional
bwrap sandboxing, workspace writability, and MLflow configuration. Its MLflow
check logs and reads back a temporary metric, then deletes the temporary run.
It exits non-zero when a required check fails; pass `--json` for machine-readable
output or `--skip-network` to skip the live MLflow check.

First-time GitHub CLI setup: [gh quickstart](https://docs.github.com/en/github-cli/github-cli/quickstart).

### Run the demo

> [!IMPORTANT]
> **MLflow tracking is ON by default** — a run exits at startup if the MLflow env vars are missing.
> Either pass `--no-mlflow` or source mlflow.env and go. For details, see [Tracking](docs/11_tracking.md).


```bash
# Generate the synthetic demo dataset + config (no internet needed).
./scripts/generate-exchange-test-data --output-dir ./demo

# Configure MLflow and (optionally) name the experiment.
source mlflow.env
export MLFLOW_EXPERIMENT_NAME="alpha-lab-demo"  # defaults to the workspace name

# Run.
python run.py --config ./demo/config.json --workspace ./demo
```

> **Config resolution.** The run materializes a canonical config at
> `{workspace}/.alpha_lab/config.json` and treats it as the single source of truth
> (intake may edit it in place). `--config` **seeds** that canonical on the first run and
> is therefore **required only when the workspace has no config yet**; on a resume you can
> omit `--config` and the workspace's canonical is used. Passing `--config` when a canonical
> already exists is a hard error **unless** it matches the existing config — pass
> **`--overwrite-config`** to deliberately replace it.

This will run all four phases autonomously:
- **Phase 0**: Tailor the system to your data
- **Phase 1**: Explore the dataset, write scripts, generate plots, build a report (~30-90 min)
- **Phase 2**: Build an evaluation framework with tests (~20-60 min)
- **Phase 3**: Run 10 GPU experiments with different ML models (~1-3 hours)

> [!TIP]
> Pass `--enable-intake` to begin a run with an interactive session for improved discovery and alignment.
> See [Configuration](docs/02_configuration.md) for additional flags and settings.

### Providers

Alpha Lab supports `openai` (default), `anthropic`, `grok`, `bedrock`, and
`local` providers. Set `provider` and the provider-specific `model` in your
config; the rest of the agent loop stays provider-agnostic. See
[Agents › Providers](docs/04_agents.md) for details.

### To run local GLM 5.2 model

Update the provider in the config to "local" and the model to the exact
model id served by the endpoint — the `id` from its `GET /v1/models` (e.g.
"glm-5.2"). The value is sent to the endpoint verbatim, and its "glm"/"kimi"
substring selects the dialect, so it must contain one of those (e.g. "glm-5.2",
"zai-org/GLM-5.1", "kimi-k2"). A bare "glm" is not a real model id and will 404.
```
export GLM_NATIVE_TOOLS=1
export LOCAL_BASE_URL=http://gpu-host-1.example.com:8000
```
> bf16 setup: gpu-host-2.example.com:8000
> nvfp4 setup: gpu-host-1.example.com:8000

See `examples/local/` for a ready-made `config.json` (`"provider": "local"`).
GLM-5.2 is text-only and routes image turns to Bedrock — set `BEDROCK_BASE_URL` if your
gateway differs from the default.

## Essentials

How Alpha Lab works internally.


| | Topic | Covers |
|---|-------|--------|
| 1 | [Overview](docs/01_overview.md) | The phases and executors that drive a run. |
| 2 | [Configuration](docs/02_configuration.md) | Task settings, agent settings, intake, and experiment-phase options. |
| 3 | [Adapters](docs/03_adapters.md) | How the system is tailored to a domain — built-in domains, customization, and generation. |
| 4 | [Agents](docs/04_agents.md) | The LLM-driven roles, their definitions, LLM providers, and sandboxing. |
| 5 | [Tools](docs/05_tools.md) | The tools agents call, and how they're granted per agent. |
| 6 | [Memory](docs/06_memory.md) | The persistent, portable store agents read from and write to. |
| 7 | [Workspace](docs/07_workspace.md) | The run's on-disk layout — what each phase writes, and where. |

## Add-ons

Tooling and integrations that sit alongside a run rather than inside it.

| | Topic | Covers |
|---|-------|--------|
| 8 | [Benchmarks](docs/08_benchmarks.md) | Creating, running, and reporting on benchmark suites. |
| 9 | [Evaluation](docs/09_evaluation.md) | Scoring how well the system was tailored to a task. |
| 10 | [Examples](docs/10_examples.md) | Bundled example configs to run as-is or adapt. |
| 11 | [Tracking](docs/11_tracking.md) | MLflow, token-usage metrics, events, and the web dashboard. |
