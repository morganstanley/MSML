# Benchmarks

A framework for generating, registering, and running benchmark suites, and
reporting results across runs. Each user-facing `alpha-lab-*` command wraps a
generic raw script; the raw scripts (with the full flag set) remain available via
`python -m alpha_lab.benchmarks.scripts.<name>`.

## Getting started

Install, materialize a suite, run it.

```bash
# The `benchmarks` extra pulls optional generator deps (e.g. tabicl for SCM).
pip install -e ".[benchmarks]"

# 1. Create the suite (writes suite.db + workspaces/ under --dest).
alpha-lab-create-suite --suite gp_regression/smoke_test --dest smoke_test

# 2. Run it (materializes per-problem workspaces under --save).
alpha-lab-run-benchmarks --db smoke_test/suite.db --save smoke_test/run_001 --num-workers 2
```

Two paths are involved on the run side:
- The **suite directory** (`--dest` for create) — contains `suite.db` and `workspaces/`.
- The **run output directory** (`--save` for run) — where this run's materialized
  workspaces and artifacts land. Omit `--save` to use a tempdir auto-cleaned on exit.

## Concepts

**Suite** — a self-contained directory of benchmark problems:

```
<suite-dir>/
    suite.db          # SQLite registry: one row per problem
    workspaces/<id>/  # .alpha_lab/config.json, data/, private/ (held-out), adapter/, benchmark_manifest.json
```

**Category** controls the source of problems:
- `msml/` — internal problems registered from real workspaces (default for `register_workspaces`).
- `scm_classification/` — synthetic tabular classification.
- `gp_regression/` — synthetic tabular regression.
- `gp_blackbox/` — synthetic black-box optimization.

**Run** — the result of executing a suite through the system; one workspace
output per problem, persisted under `--save` (or discarded in a tempdir).

Internal suites live at `/v/campus/vi/appl/msml/qa/data/alpha-labs-benchmarks/suites/`.

## Creating a suite

### Synthetic (built-in generators)

Suite definitions live in `suites.yaml`; built-in groups are `scm_classification`,
`gp_regression`, `gp_blackbox`, each with `smoke_test`/`easy`/`medium`/`hard` tiers.

```bash
alpha-lab-create-suite --suite <group>/<tier> --dest <suite-dir> [--overwrite] [--owner <str>]
```

`--suite` accepts `<group>/<tier>` (a key into the bundled `suites.yaml`) or
`<path>:<key>` (load `<key>` from an external YAML following the same structure).

### From existing workspaces

```bash
alpha-lab-register-workspaces --src path/to/ws1 path/to/ws2 --dest <suite-dir> \
  [--config <JSON_OR_PATH>] [--symlink-data] [--overwrite] [--owner <str>]
```

Each source workspace is registered via `copy_workspace` (a full copy by default;
`data/` and `workspace_includes` entries can be symlinked instead).

### Copying a workspace standalone

```bash
alpha-lab-copy-workspace --src <source_ws> --dest <dst_parent> \
  [--name <dst_name>] [--symlink-data] [--symlink-includes]
```

## Running a suite

```bash
alpha-lab-run-benchmarks --db <suite-dir>/suite.db \
  [--save <run-output-dir>]   # default: tempdir, auto-cleaned on exit
  [--num-workers <N>] [--runner local|mlflow] \
  [--filter ID ID2 ... | N]   # subset: ids, OR a single int = first N
  [--flags '<json>']          # escape hatch for raw-script flags
```

- `--runner` — `local` (default) or `mlflow`. The `mlflow` runner parents each
  workspace's pipeline run under a Suite Run; see [Tracking](11_tracking.md).
- `--num-workers` — parallelism across problems; workspaces materialize lazily.
- `--filter` — a list of benchmark ids, or a single positive integer N (first N).
- `--flags` — JSON dict forwarded to the raw script (snake_case keys become
  `--kebab-case` flags), e.g. `'{"config_overrides": {"provider": "bedrock"}}'`.

## Removing benchmarks

```bash
alpha-lab-remove-benchmark --dest <suite-dir> --id <bench_id_1> <bench_id_2> ...
```

Deletes the matching rows from `suite.db` and their `workspaces/<id>/`. Missing
ids warn but don't fail unless none are found.

## Reporting

After runs complete, summarize each workspace, then produce a cross-run report.
We haven't given these two simplified CLIs yet — invoke via `python -m`.

```bash
# 1. Summarize each workspace (writes bench_summary.json to each).
python -m alpha_lab.benchmarks.scripts.summarize_workspace \
  --workspaces runs/my_run_001/ws_a runs/my_run_001/ws_b [--model gpt-5.4] [--top-k 5] [--overwrite]

# 2. Generate the cross-run markdown report.
python -m alpha_lab.benchmarks.scripts.create_report \
  --runs runs/my_run_001 runs/my_run_002 --output reports/comparison.md \
  [--metrics sharpe_ratio] [--model gpt-5.4] [--no-prose]
```

Step 1 must run for every workspace before step 2; step 2 fails loudly if a
`bench_summary.json` is missing. The report covers resource consumption per run,
per-metric rank-based tables, and per-run narratives. Runs needn't cover
identical workspace sets — the report uses the intersection and warns about the rest.

## Available suites

All live under `/v/campus/vi/appl/msml/qa/data/alpha-labs-benchmarks/suites/`.
Each synthetic group has `smoke_test` (2–4 problems, quick validation) plus
`easy`/`medium`/`hard` tiers (32 each) of increasing difficulty:

| Group | Problem type |
|-------|--------------|
| `scm_classification` | Synthetic tabular classification (tabicl SCM prior) |
| `gp_regression` | Synthetic regression from a GP prior |
| `gp_blackbox` | Synthetic black-box optimization with a GP-prior objective |
| `msml` | Internal problems registered from real workspaces (empty by default) |

Each problem is a self-contained workspace; held-out test data lives under
`private/` and is quarantined from the agent.

## Adding suite definitions

Edit `suites.yaml`. Each top-level group declares a `generator` (import path, or
absolute file path with `:ClassName`) and one or more named tiers; tiers inherit
the group's generator unless they override it. Suite paths map to nested YAML
keys (e.g. `gp_regression/hard`):

```yaml
gp_regression:
  generator: alpha_lab.benchmarks.generators.gp_regression:GPRegressionGenerator
  hard:
    generator_kwargs: { seed: 3000, count: 32 }
    config_overrides: { reasoning_effort: high }
```

For generators outside the package, `generator` accepts a path-form import (e.g.
`/scratch/me/my_gen.py:MyGenerator`).
