# MLflow Integration

This doc describes how alpha-lab integrates with MLflow today and how to run
it. The integration lives entirely in `src/alpha_lab/mlflow_logger.py` as a
self-contained side-by-side module. The pre-existing OTel/Tempo path in
`src/alpha_lab/tracing.py` is untouched; the two backends are mutually
exclusive at runtime.

## Quick start

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
export MLFLOW_EXPERIMENT_NAME=alpha-lab-runs   # or set MLFLOW_EXPERIMENT_ID
python run.py --config data/demo_exchange_config.json \
              --workspace ./workspace_demo \
              --mlflow
```

The `--mlflow` flag sets `ALPHALAB_MLFLOW=1` in the process environment.
Without it (or without `MLFLOW_TRACKING_URI`) every call in
`mlflow_logger.py` is a no-op and no MLflow SDK code runs.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | yes | HTTP(S) URL of the MLflow tracking server |
| `ALPHALAB_MLFLOW` | yes | Set to `1` / `true` / `yes` to activate (set by `--mlflow`) |
| `MLFLOW_EXPERIMENT_ID` | one of these | Numeric experiment ID; takes priority |
| `MLFLOW_EXPERIMENT_NAME` | one of these | Human-readable name; auto-created if missing |
| `USER` | no | Forwarded as `remote-user` ADC header (set automatically on Linux) |

## Auth (MS-internal tracking server)

The MS-internal MLflow tracking server uses `remote-user` trust-the-header
auth (the `msml_mlflow_adc_auth` plugin). `mlflow_logger.py` registers an
`_AlphaLabHeaderProvider` with the MLflow SDK's
`_request_header_provider_registry` so every outgoing SDK HTTP call carries
`remote-user: $USER`. No extra setup needed beyond having a valid Kerberos
ticket (same requirement as the rest of alpha-lab on-prem).

## Run hierarchy

```
MLflow Experiment
└── Pipeline Run  (one per alpha-lab workspace; name = run_id)
    ├── Agent Span  (Phase 1 agent loop)
    │   ├── chat span
    │   └── tool span
    ├── Agent Span  (Phase 2 builder)
    ├── Agent Span  (Phase 2 critic)
    ├── Agent Span  (Phase 3 strategist)
    └── Experiment Sub-Run  (one per Phase 3 GPU experiment)
        └── Agent Span  (worker loop)
```

The Pipeline Run is created (or resumed) by `pipeline_run()` in
`mlflow_logger.py`. It is tagged `alpha_lab.run_kind = "pipeline"`.

Phase 3 GPU experiments each get a child sub-run tagged
`alpha_lab.run_kind = "experiment"` and `mlflow.parentRunId = <pipeline uuid>`
so the MLflow UI tree view works.

## Span wiring

`agent_trace(name)` opens a top-level MLflow Span scoped to the active
pipeline Run (via `mlflow.tracing.context`). `child_span(name)` opens a
child span under whatever `agent_trace` is active. Both are context managers
that yield the span (or `None` when MLflow is off).

The `mlflow.tracing.context(metadata={TraceMetadataKey.SOURCE_RUN: uuid})`
call pins the source run for each worker thread, which is needed in Phase 3
where many worker threads run concurrently and MLflow's default
`_get_latest_active_run()` would pick the wrong run.

## Autolog

`configure_sdk()` calls `mlflow.openai.autolog()` and `mlflow.bedrock.autolog()`
so every LLM API call is automatically captured as a child span without any
instrumentation in `agent.py`. Failures (missing transitive deps, version
mismatches) are caught per-SDK and logged as warnings — they don't abort the
`--mlflow` path.

## Artifacts

Three levels of artifact upload are supported:

1. **Single file**: `log_run_artifact(run_uuid, local_path, artifact_path)`
2. **Directory tree**: `log_run_artifacts_dir(run_uuid, local_dir, prefix)`
3. **Pipeline-Run shortcuts**: `log_pipeline_artifact(...)` / `log_pipeline_artifacts_dir(...)`
   route to the active pipeline Run without passing `run_uuid` explicitly.

The directory uploader skips `__pycache__`, `.pyc`, `.git`, `.lock` files
and anything over 100 MB by default.

## Resume behavior

If alpha-lab is restarted against the same workspace (same `run_id`),
`pipeline_run()` calls `_get_or_create_run()` which searches for an existing
MLflow Run by name. If found it reuses the same Run UUID and calls
`client.update_run(..., status="RUNNING")` to reset the stale `FAILED` status
in the UI. New spans and metrics are appended to the existing Run.

## Params are immutable

MLflow params are write-once. If the same param key is logged twice (e.g.
on restart), the second write silently fails (logged at DEBUG level). This is
by design — MLflow's API doesn't support param updates.

## OTel / Tempo co-existence

The two tracing backends are mutually exclusive:

- `ALPHALAB_MLFLOW=1` + `MLFLOW_TRACKING_URI` set → MLflow active, OTel dormant
  (`init_tracing()` in `tracing.py` skips installing a TracerProvider when no
  `OTEL_EXPORTER_OTLP_ENDPOINT` is configured)
- `OTEL_EXPORTER_OTLP_ENDPOINT` set, `ALPHALAB_MLFLOW` unset → OTel/Tempo active,
  `mlflow_logger.py` is a no-op
- Neither set → no tracing at all; pipeline runs fine

## Benchmarks integration

When running alpha-lab as part of a benchmark suite (via
`src/alpha_lab/benchmarks/`), the `MLflowRunner` creates a parent Suite Run,
runs many pipeline Runs in parallel (each in its own process with
`ALPHALAB_MLFLOW=1` injected), then re-parents each pipeline Run under the
Suite Run using `mlflow.parentRunId` tags. The `trace_info.json` written by
`_write_trace_info()` is how `MLflowRunner` discovers each pipeline Run's UUID
after the subprocess exits.

## Code map

| File | What it does |
|---|---|
| `src/alpha_lab/mlflow_logger.py` | All MLflow logic. `is_active()` gate, `configure_sdk()`, `pipeline_run()` CM, `agent_trace()` / `child_span()` CMs, `create_experiment_run()`, logging helpers. |
| `src/alpha_lab/run.py` | `--mlflow` flag → sets `ALPHALAB_MLFLOW=1`, calls `configure_sdk()`, wraps `_run_pipeline()` in `pipeline_run()`. |
| `src/alpha_lab/agent.py` | `run()` wraps each agent loop in `agent_trace(phase_name)` and each LLM call / tool dispatch in `child_span()`. |
| `src/alpha_lab/dispatcher.py` | `_submit_experiment()` calls `create_experiment_run()` and stores `mlflow_run_uuid` on the experiment record; `_on_experiment_done()` calls `terminate_run()`. |
| `src/alpha_lab/worker.py` | Worker loop calls `agent_trace(exp_id, target_run_uuid=mlflow_run_uuid)` so worker spans land in the experiment sub-run, not the pipeline run. |
| `src/alpha_lab/benchmarks/runners.py` | `MLflowRunner(LocalRunner)`. Resolves `MLFLOW_EXPERIMENT_NAME` from env, creates the Suite Run, injects `ALPHALAB_MLFLOW=1` + experiment env into child subprocesses, re-parents pipeline Runs under the Suite Run via tags. Falls back to `LocalRunner.run_many` when `MLFLOW_TRACKING_URI` is unset. |
| `src/alpha_lab/benchmarks/run_benchmarks.py` | `--runner mlflow` registered in `_RUNNER_REGISTRY`. |
