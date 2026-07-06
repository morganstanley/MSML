# MLflow Integration

This doc describes how alpha-lab integrates with MLflow today and how to run
it. The integration lives entirely in `src/alpha_lab/mlflow_logger.py` as a
self-contained side-by-side module. The pre-existing OTel/Tempo path in
`src/alpha_lab/tracing.py` is untouched; the two backends are mutually
exclusive at runtime.

## 0. MLflow Tracking server

Point `MLFLOW_TRACKING_URI` at any MLflow tracking server you have access to.

To run a local server:

```bash
mlflow server --host 0.0.0.0 --port 8081
export MLFLOW_TRACKING_URI="http://localhost:8081"
```

If the server runs on a remote host, forward the port over SSH:

```bash
ssh -fN -L 8081:localhost:8081 <your-mlflow-host>   # tunnel in background
export MLFLOW_TRACKING_URI="http://localhost:8081"
```

- `-f` daemonizes after auth; `-N` skips the remote shell.

Kill the tunnel later:

```bash
pkill -f "ssh.*-L 8081:localhost:8081"
```

## 1. Setup & Usage

### 1.1. Prerequisites

```bash
pip install mlflow
```

Already pinned in `requirements.txt` / `pyproject.toml`; only needed if you
have an older install.

### 1.2. Required env vars (only when `--mlflow` is on)

| Env var | Required? | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | **yes** | MLflow tracking server URL (e.g. `http://localhost:8081`). `--mlflow` exits with an error if unset. |
| `MLFLOW_EXPERIMENT_NAME` | **yes** (or `_ID`) | Experiment to land Runs in. Auto-created if missing. |
| `MLFLOW_EXPERIMENT_ID` | optional | Skip the name→id resolution if you already have the handle. |
| `USER` | implicit | The login user is sent as both the ADC `remote-user` header and the value passed to `mlflow.set_workspace(...)`. |
| `ALPHALAB_MLFLOW` | optional | `--mlflow` flips this to `1` automatically; exporting directly has the same effect (useful in CI). |

### 1.3. Running a single pipeline

```bash
export MLFLOW_TRACKING_URI="http://localhost:8081"
export MLFLOW_EXPERIMENT_NAME="alpha-lab-demo"

python run.py \
    --config data/exchange_config.json \
    --workspace ./workspace_demo \
    --mlflow
```

Expected output on startup:

```
INFO MLflow SDK configured
INFO MLflow pipeline run: <run_id> (workspace: ..., mlflow_run_uuid: <uuid>, artifact_uri: ...)
```

### 1.4. Multiple invocations per experiment

By default each invocation gets a **new** MLflow Run, even from the same
workspace. 

To **resume** an existing Run (same UUID, same name, status reset to
`RUNNING` on entry), pass the prior run id explicitly:

```bash
python run.py --config ... --workspace ... \
              --run-id <prior-run-id> --mlflow
```

Equivalent: export `ALPHALAB_RUN_ID=<prior-run-id>`. 

### 1.5. Running a benchmark suite

```bash
export MLFLOW_TRACKING_URI="http://localhost:8081"
export MLFLOW_EXPERIMENT_NAME="alpha-lab-scm-smoke"

# 1. Materialize the suite catalog (once)
alpha-lab-create-suite \
    --suite scm_classification/smoke_test \
    --dest /var/tmp/${USER}/alpha-lab/scm_smoke \
    --overwrite

# 2. Run it — note --save is a separate path. Use a timestamp so each
#    invocation gets fresh outputs.
alpha-lab-run-benchmarks \
    --db /var/tmp/${USER}/alpha-lab/scm_smoke/suite.db \
    --save /var/tmp/${USER}/alpha-lab/scm_smoke_runs/$(date +%s) \
    --num-workers 2 \
    --runner mlflow
```



## 2. The MLflow hierarchy

The shape depends on how you launch alpha-lab. There are two cases.

### 2.1. Single pipeline mode (`python run.py … --mlflow`)

No Suite Run — the Pipeline Run is the top-level Run. Phase 3 experiment
sub-runs (if any) nest underneath it.

```
Experiment "<MLFLOW_EXPERIMENT_NAME>"
│
└── Pipeline Run "<run_id>"                       ← top-level Run
    tags:    alpha_lab.run_kind = "pipeline"
             mlflow.runName = "<run_id>"
             mlflow.user = "$USER"
    params:  task.description, task.target, data_path,
             domain, provider, model, reasoning_effort,
             config_path, workspace,
             phase0.adapter_domain, phase0.primary_metric, phase0.metric_direction
    metrics: phase{0,1,2,3}.duration_seconds
    artifacts:
        config.json
        phase0/adapter/
        phase1/learnings.md, phase1/data_report/, phase1/plots/, phase1/scripts/
        phase2/<framework_dir>/, phase2/framework_review.md, phase2/framework_critique.md
        phase3/leaderboard.md, phase3/playbook.md, phase3/final_report.md, phase3/reports/
    traces: one per agent invocation
        invoke_agent phase0_customize
        invoke_agent phase1_explorer
        invoke_agent phase2_builder_0 / phase2_critic / phase2_tester
        invoke_agent strategist                   (one per strategist tick)
        invoke_agent supervisor_validate_adapter / supervisor_review_phase{1,2} / …
    │
    ├── Sub-Run "<experiment_name>"               ← per Phase 3 propose_experiment
    │   tags:    alpha_lab.run_kind = "experiment"
    │            mlflow.parentRunId = "<pipeline run uuid>"
    │            alpha_lab.parent_run_id = "<pipeline run uuid>"
    │            alpha_lab.parent_run_name = "<run_id of pipeline>"
    │   params:  description, hypothesis, config, experiment_id
    │   metrics: numeric keys from results.json (e.g. accuracy, sharpe, …)
    │   artifacts: experiments/<name>/, debrief/<file>
    │   traces: worker_<n>_implement_<name>, _analyze_<name>, _fix_<name>
    └── …
```

### 2.2. Benchmark suite mode (`alpha-lab-run-benchmarks --runner mlflow`)

`MLflowRunner` creates a Suite Run at the top; each benchmarked workspace's
Pipeline Run is parented under it.

```
Experiment "<MLFLOW_EXPERIMENT_NAME>"
│
└── Suite Run "<suite_name>"                      ← MLflowRunner only
    tags:    alpha_lab.run_kind = "suite"
             alpha_lab.suite = "<suite_name>"
             mlflow.user = "$USER"
    metrics: alpha_lab.suite.benchmarks_total
             alpha_lab.suite.benchmarks_completed
    │
    ├── Pipeline Run "<run_id>"                   ← one per workspace
    │   tags:   alpha_lab.run_kind = "pipeline"
    │           mlflow.parentRunId = "<suite run uuid>"   (set post-hoc)
    │           alpha_lab.parent_run_id = "<suite run uuid>"
    │           alpha_lab.parent_run_name = "<suite_name>"
    │   (params / metrics / artifacts / traces — same as §2.1)
    │   │
    │   ├── Sub-Run "<experiment_name>"           ← Phase 3 propose_experiment
    │   │   (same shape as §2.1's sub-run)
    │   └── …
    └── …
```

The Pipeline Run gets the three parent tags set in
`MLflowRunner._on_workspace_done` *after* the child subprocess exits, so
during execution the Pipeline Run momentarily appears at the top level of
the experiment view and then re-nests under the Suite Run when its child
finishes. 

## 3. UI filters

| Goal | Filter |
|---|---|
| All experiment sub-runs across the experiment | `tags.alpha_lab.run_kind = "experiment"` |
| All pipeline runs | `tags.alpha_lab.run_kind = "pipeline"` |
| Everything under a specific suite | `tags.alpha_lab.parent_run_name = "alpha-lab-scm-smoke-20260528-..."` |
| Sub-runs of one pipeline | `tags.alpha_lab.parent_run_name = "<pipeline_run_id>"` |

## 4. Code map

| File | Role |
|---|---|
| `src/alpha_lab/mlflow_logger.py` | All MLflow integration (new). Public API: `is_active()` / `mlflow_enabled()`, `configure_sdk()`, `pipeline_run()`, `agent_trace()`, `child_span()`, `set_inputs/outputs`, `SpanType`, `create_experiment_run`, `terminate_run`, `log_run_*`, `log_pipeline_*`. Uses MLflow native tracing SDK; no OTel imports. |
| `src/alpha_lab/tracing.py` | OTel/Tempo path. **Untouched** by the MLflow integration. |
| `src/alpha_lab/run.py` | Owns the `--mlflow` CLI flag. Picks the backend (mutually exclusive); calls `mlflow_logger.configure_sdk()` and wraps the pipeline in `mlflow_logger.pipeline_run` in MLflow mode, otherwise the existing `pipeline_span`. Sprinkles `mlflow_logger.log_pipeline_*` at phase boundaries (all no-op when MLflow off). |
| `src/alpha_lab/agent.py` | `AgentLoop` accepts `mlflow_run_target`. Layers `mlflow_logger.agent_trace` around the OTel root span; adds `child_span` + `set_inputs/outputs` on chat + tool spans. All MLflow-mode branches are no-op without MLflow. |
| `src/alpha_lab/worker.py` | Passes `mlflow_run_target=experiment.mlflow_run_uuid` so Phase 3 worker traces attach to the experiment's sub-run. |
| `src/alpha_lab/tools.py` | `_attach_mlflow_run_to_experiment` (called from `propose_experiment`) and `_log_experiment_results_to_mlflow` (called from `update_experiment`). |
| `src/alpha_lab/experiment_db.py` | Two columns `mlflow_run_uuid` + `mlflow_artifact_uri` (NULL in Tempo mode), `set_mlflow_run()` setter, idempotent ALTER TABLE migration. |
| `src/alpha_lab/benchmarks/runners.py` | `MLflowRunner(LocalRunner)`. Resolves `MLFLOW_EXPERIMENT_NAME` from env, creates the Suite Run, injects `ALPHALAB_MLFLOW=1` + experiment env into child subprocesses, re-parents pipeline Runs under the Suite Run via tags. Falls back to `LocalRunner.run_many` when `MLFLOW_TRACKING_URI` is unset. |
| `src/alpha_lab/benchmarks/scripts/run_benchmarks.py` | `--runner mlflow` registered in `_RUNNER_REGISTRY`. |