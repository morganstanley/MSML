# Tracking

How a run is tracked, and how to watch it live. Tracking has two independent
paths — **MLflow** (Runs, metrics, artifacts, traces; on by default) and an
always-on **token-usage metric** — and the web dashboard is a separate, passive
viewer you can point at any run.

## MLflow

**MLflow is on by default.** Every run logs Runs, metrics, artifacts, and traces
to the shared MLflow tracking server over mTLS, so `run.py` validates the required
env vars at startup and errors out if any are missing. Set MLflow up (`source
mlflow.env`) or pass `--no-mlflow` to disable it — a fallback for when the
tracking server is unavailable.

The shared MLflow tracking server is hosted at `https://mlflow.example.com`.
**Authentication is always mTLS**: Alpha Lab presents a cinit client cert and the
proxy derives your identity from it. Each experiment lands in the workspace of the
user who ran it. The client is pinned to `mlflow==3.14.0` to match the server.

### Required env vars

| Env var | Purpose |
|---------|---------|
| `MLFLOW_TRACKING_URI` | Tracking server URL (`https://mlflow.example.com`). |
| `MLFLOW_TRACKING_CLIENT_CERT_PATH` | Single PEM with cert + key (the mTLS client cert). |
| `MLFLOW_TRACKING_SERVER_CERT_PATH` | CA bundle to verify the server. |

`source mlflow.env` sets all three. The **experiment is optional**:
`MLFLOW_EXPERIMENT_NAME` (or `MLFLOW_EXPERIMENT_ID`) selects it — reused if it
already exists, created if not — and if you set neither it auto-defaults to the
workspace basename.

```bash
source mlflow.env
# Optional — defaults to the workspace basename; set to group runs together:
export MLFLOW_EXPERIMENT_NAME="alpha-lab-demo"
python run.py --config ./demo/config.json --workspace ./demo
```

Before a long run, verify the SDK version, mTLS files, server connection, and
metric logging path:

```bash
source mlflow.env
alpha-lab-doctor --workspace ./demo
```

The doctor creates a temporary run, logs and reads back one metric, then deletes
the temporary run. If the selected experiment does not exist, the check creates
it just as the pipeline would. Use `--skip-network` to validate configuration
without contacting the server.

### Run hierarchy

The shape depends on how a run is launched. A single pipeline is the top-level
Run, with each Phase 3 experiment nested under it as a sub-run:

```text
Experiment  "<MLFLOW_EXPERIMENT_NAME>"
│
└── Pipeline Run "<run_id>"                    ← top-level Run
    tags:      alpha_lab.run_kind = "pipeline", mlflow.user = "$USER", …
    params:    task.description, task.target, data_path, domain, provider,
               model, phase0.adapter_domain, …
    metrics:   phase{0,1,2,3}.duration_seconds
    artifacts: config.json, phase0/adapter/, phase1/learnings.md,
               phase2/<framework_dir>/, phase3/leaderboard.md, phase3/reports/, …
    traces:    one "invoke_agent <role>" span per agent invocation
               (phase1_explorer, phase2_builder, strategist, supervisor_*, …)
    │
    └── Sub-Run "<experiment_name>"            ← one per Phase 3 experiment
        tags:      alpha_lab.run_kind = "experiment", mlflow.parentRunId,
                   alpha_lab.parent_run_id, alpha_lab.parent_run_name
        params:    description, hypothesis, config, experiment_id
        metrics:   numeric keys from the results file (accuracy, sharpe, …)
        artifacts: experiments/<name>/, debrief/…
        traces:    worker_<n>_implement/_analyze/_fix_<name>
```

A [Benchmark](08_benchmarks.md) suite (`alpha-lab-run-benchmarks --runner mlflow`)
adds a Suite Run on top, with each workspace's pipeline run parented under it:

```text
Experiment  "<MLFLOW_EXPERIMENT_NAME>"
│
└── Suite Run "<suite_name>"                   ← benchmark suites only
    tags:    alpha_lab.run_kind = "suite", alpha_lab.suite = "<suite_name>"
    metrics: alpha_lab.suite.benchmarks_total, …benchmarks_completed
    │
    └── Pipeline Run "<run_id>"                ← one per workspace
        tags: alpha_lab.run_kind = "pipeline"; parent tags point to the Suite Run
        (params / metrics / artifacts / traces — same as above)
        │
        └── Sub-Run "<experiment_name>"        ← same shape as above
```

Nesting is by tag: a child carries `mlflow.parentRunId` plus the dot-free
`alpha_lab.parent_run_id` / `alpha_lab.parent_run_name`. In suite mode the parent
tags are set after each child exits, so a pipeline run briefly appears at the top
level and then re-nests under its Suite Run once its child finishes.

Filter the UI on these tags:

| Goal | Filter |
|------|--------|
| All experiment sub-runs | `tags.alpha_lab.run_kind = "experiment"` |
| All pipeline runs | `tags.alpha_lab.run_kind = "pipeline"` |
| Everything under one suite | `tags.alpha_lab.parent_run_name = "<suite_name>"` |
| Sub-runs of one pipeline | `tags.alpha_lab.parent_run_name = "<pipeline run_id>"` |

### Resuming a run

By default each invocation gets a new MLflow Run, even from the same workspace. To
resume an existing one, pass `--run-id <prior-run-id>` (or export `ALPHALAB_RUN_ID`).

## Token-usage metrics

Independent of MLflow and **on by default**, every run emits a token-usage metric:
a single Prometheus counter (`alpha_lab_token_usage`, broken down by
`gen_ai.token.type`) pushed to the firm Cortex/Mimir store via remote-write, so
team-wide LLM token spend is queryable in Grafana by `user.id`,
`gen_ai.request.model`, and `experiment`.

**Dashboard:** Alpha-Lab Token Usage (internal Grafana — see your observability team for the link)

It no-ops only when it can't or shouldn't run: `ALPHALAB_TOKEN_METRICS_DISABLED=1`
(explicit off-switch), or no mTLS client cert is available (`$CINITCCNAME` /
`cert.pem` missing).

The `experiment` label reuses the resolved MLflow experiment name (or the
workspace basename when unset), so token spend still lands under a meaningful
experiment even under `--no-mlflow`.

## Events & metrics

As it runs, the system emits structured events for real-time monitoring, used
by both the CLI and the web dashboard below.

| Event | Emitted when |
|-------|--------------|
| `StatusEvent` | An agent changes status (starting, thinking, tool_executing, done, error) |
| `PhaseEvent` | A phase transitions |
| `ExperimentEvent` | An experiment changes status, with metrics |
| `BoardSummaryEvent` | Periodic experiment-board snapshots |
| `ToolCallEvent` / `ToolResultEvent` | A tool is called / returns |
| `FileChangedEvent` | A workspace file changes |
| `ErrorEvent` | An error is logged |

`MetricsCollector` provides thread-safe, in-memory tracking with no external
dependencies: token accounting (input/output per API call), API call counts and
error rates, experiment throughput (count, average duration, experiments/hour),
and session uptime. Call `metrics.snapshot()` for a JSON-serializable summary at
any point.

## Web dashboard

The dashboard lets you watch the system in real-time. It's a passive viewer — it
doesn't control the system, and you can start it before, during, or after a run.

```bash
# First time only — build the frontend (configure npm for your registry first).
module load node/24.10.0
cd frontend && npm install && npm run build && cd ..

# Start it against a workspace.
python serve.py --workspace ./workspace --port 8000
# open http://localhost:8000
```

It will:
- **Stream live events** — see the LLM thinking, writing code, running experiments
- **Browse workspace files** — scripts, plots, reports, experiment code
- **Show the experiment board** — the status lifecycle from proposed → running → done
- **Display the leaderboard** — experiments ranked by metric
- **Chat** — ask questions about system state ("What's the best model?", "Any errors?")
