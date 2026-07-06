# alpha_lab.benchmarks

A benchmarking framework for Alpha Lab. Generates, registers, and runs benchmark
suites; records results for cross-run comparison and reporting.

Each user-facing `alpha-lab-*` command wraps a generic raw script. The simple
CLI surface is what's documented here. The raw scripts (with the full flag set)
remain available via ``python -m alpha_lab.benchmarks.scripts.<name>``.

---

## Getting started

End-to-end: install, materialize a suite, run it.

```bash
# 0. Install (the `benchmarks` extra pulls optional generator deps, e.g. tabicl for SCM)
pip install -e ".[benchmarks]"

# 1. Create the suite (writes suite.db + workspaces/ under --dest)
alpha-lab-create-suite \
  --suite gp_regression/smoke_test \
  --dest smoke_test

# 2. Run it (pulls from the suite's suite.db; materializes per-problem
#    workspaces under --save)
alpha-lab-run-benchmarks \
  --db smoke_test/suite.db \
  --save smoke_test/run_001 \
  --num-workers 2
```

Two paths are involved on the run side:
- The **suite directory** (`--dest` for create) — contains `suite.db` and `workspaces/`.
- The **run output directory** (`--save` for run) — where this run's
  materialized workspaces (and their artifacts) land. Omit `--save` to use a
  tempdir auto-cleaned on exit.

The rest of this README is reference material for each piece.

---

## Concepts

**Suite** — a self-contained directory of benchmark problems:

```
<suite-dir>/
    suite.db          # SQLite registry: one row per problem
    workspaces/
        <id>/
            config.json
            data/
            private/              # held-out test data + audit artifacts
            adapter/              # optional domain adapter
            benchmark_manifest.json
```

**Category** controls the source of problems:
- `custom/` — benchmark problems registered from real workspaces. Default suite for `register_workspaces`.
- `scm_classification/` — synthetic tabular classification problems.
- `gp_regression/` — synthetic tabular regression problems.
- `gp_blackbox/` — synthetic black-box optimization problems.

Registered suites are stored under your configured suite directory (see `DEFAULT_SUITE_DIR` in `paths.py`).

**Run** — the result of executing a suite through the Alpha Lab pipeline; one
workspace output per problem, persisted under `--save <dir>` (or thrown away
into a tempdir if `--save` is omitted).

**`workspace_includes`** — list of workspace-relative paths (e.g. `["private"]`)
that the generator's `bootstrap` method and `copy_workspace` carry over
alongside `data/`. Each declared include must exist or bootstrap raises.

---

## Creating a suite

### Synthetic (built-in generators)

Suite definitions live in `suites.yaml`. Each top-level group declares a
`generator` (import path) plus per-tier `generator_kwargs` and
`config_overrides`. Built-in groups: `scm_classification`, `gp_regression`,
`gp_blackbox` — each with `smoke_test`, `easy`, `medium`, `hard` tiers.

```bash
alpha-lab-create-suite \
  --suite <group>/<tier> \
  --dest <suite-dir> \
  [--overwrite] [--owner <str>]
```

`--suite` accepts two forms:

- `<group>/<tier>` — a slash-separated key into the bundled `suites.yaml`
  (e.g. `gp_blackbox/smoke_test`).
- `<path>:<key>` — load `<key>` from an external YAML file at `<path>`.
  The external YAML follows the same nested structure as `suites.yaml`
  (top-level group with `generator` + nested tier blocks containing
  `generator_kwargs` and `config_overrides`).

### From existing workspaces

```bash
alpha-lab-register-workspaces \
  --src path/to/ws1 path/to/ws2 \
  --dest <suite-dir> \
  [--config <JSON_OR_PATH>]   # external config: inline JSON dict OR path to a JSON file
  [--symlink-data]            # symlink data instead of copying
  [--overwrite] [--owner <str>]
```

Pass your suite directory (any path) as `--dest`.

Each source workspace is registered via `copy_workspace` — a full copy of
the source directory by default; `data/` and `workspace_includes` entries can
be symlinked instead via flags below.

### Copying a workspace standalone

```bash
alpha-lab-copy-workspace \
  --src <source_ws> \
  --dest <dst_parent> \
  [--name <dst_name>] \
  [--symlink-data] \
  [--symlink-includes]
```

- Default behavior copies everything from the source workspace.
- `--symlink-data` flips `data/` from copy to symlink.
- `--symlink-includes` flips each entry declared in the source's
  `workspace_includes` from copy to symlink.

---

## Running a suite

```bash
alpha-lab-run-benchmarks \
  --db <suite-dir>/suite.db \
  [--save <run-output-dir>]     # default: tempdir, auto-cleaned on exit
  [--num-workers <N>] \
  [--runner local]              # only "local" supported today
  [--filter ID ID2 ... | N]     # subset: one or more ids, OR a single int = first N
  [--flags '<json>']            # escape hatch for raw-script flags
```

Key flags:
- `--db` — path to a `suite.db` (input). Required.
- `--save <dir>` — persist materialized workspaces under this directory. If
  omitted, runs in a temp directory that's deleted on exit.
- `--num-workers` — parallelism across problems. Workspaces are pulled lazily
  from the generator, so at most ~`num_workers` materialize ahead of the pool.
- `--runner` — runner backend; only `local` is implemented (mlflow planned).
- `--filter` — either a list of benchmark ids to run, OR a single positive
  integer N which selects the first N entries in the registry.
- `--flags` — JSON dict forwarded to the raw `run_benchmarks` script. Keys
  use snake_case and become `--kebab-case` flags. Example:
  `'{"config_overrides": {"provider": "bedrock"}, "agent_config": {"model": "gpt-5.4"}, "overwrite": true}'`.

`--save` and the implicit tempdir mode are mutually exclusive (omit `--save`
to opt into the tempdir).

---

## Removing benchmarks

```bash
alpha-lab-remove-benchmark \
  --dest <suite-dir> \
  --id <bench_id_1> <bench_id_2> ...
```

Deletes the matching rows from `suite.db` and removes `workspaces/<bench_id>/`
from disk. Missing IDs log a warning but don't fail the call unless none of the
requested IDs are found.

---

## Reporting

After one or more runs complete, generate `bench_summary.json` for each
workspace, then produce a cross-run markdown report. These two tools don't
have simplified CLIs yet; invoke via `python -m`.

### Step 1 — summarize each workspace

```bash
python -m alpha_lab.benchmarks.scripts.summarize_workspace \
  --workspaces runs/my_run_001/ws_a runs/my_run_001/ws_b \
  [--model gpt-5.4] \
  [--top-k 5] \
  [--num-workers 4] \
  [--overwrite]
```

Writes `bench_summary.json` to each workspace root. Reads `experiments.db`,
`adapter/manifest.json`, milestone reports, and top-K experiment debriefs; calls
the model to produce `narrative`, `errors_summary`, and `noteworthy_observations`.
Degrades gracefully if artifacts are missing. Glob patterns are supported.
Must be run for every workspace in every run before generating a report.

### Step 2 — generate the report

```bash
python -m alpha_lab.benchmarks.scripts.create_report \
  --runs runs/my_run_001 runs/my_run_002 \
  --output reports/comparison.md \
  [--metrics sharpe_ratio]   # default: all metrics found in summaries
  [--model gpt-5.4]          # omit to skip per-run narrative generation
  [--no-prose]               # explicit skip even if --model is set
```

Collects `bench_summary.json` from each workspace across all runs. Fails loudly
if any is missing. Produces a markdown report with:
- resource consumption per run (experiment counts by status)
- per-metric rank-based relative performance tables
- per-run narrative (from `bench_summary.json` narratives, or LLM-generated if
  `--model` is set)

Runs do not need to cover identical workspace sets; the report uses the
intersection and warns about omitted workspaces.

---

## Available suites

All suites live under your configured suite directory (see `DEFAULT_SUITE_DIR` in `paths.py`).

| Suite | Problems | Description |
|-------|----------|-------------|
| `custom` | 0 (empty) | Benchmark problems registered from real workspaces. Default target for `register_workspaces`. |
| `scm_classification/smoke_test` | 4 | Quick validation. Synthetic tabular classification (tabicl SCM prior). |
| `scm_classification/easy` | 32 | 2–8 features, ≤3 classes. |
| `scm_classification/medium` | 32 | 4–16 features, ≤4 classes. |
| `scm_classification/hard` | 32 | 8–32 features, ≤8 classes. |
| `gp_regression/smoke_test` | 2 | Quick validation. Synthetic 1-D regression from a GP prior. |
| `gp_regression/easy` | 32 | 1-D inputs, low noise. |
| `gp_regression/medium` | 32 | 4-D inputs, moderate noise. |
| `gp_regression/hard` | 32 | 8-D inputs, high noise, short lengthscale. |
| `gp_blackbox/smoke_test` | 2 | Quick validation. Synthetic black-box optimization with a GP-prior objective. |
| `gp_blackbox/easy` | 32 | 2-D inputs, eval budget 32. |
| `gp_blackbox/medium` | 32 | 3-D inputs, eval budget 48. |
| `gp_blackbox/hard` | 32 | 4-D inputs, eval budget 64. |

Synthetic suites are generated from packaged generators. Each problem is a
self-contained workspace; held-out test data lives under `private/` and is
quarantined from the agent.

---

## Module layout

```
benchmarks/
  suites.yaml              # built-in suite definitions
  runners.py               # LocalRunner
  manifest.py              # benchmark + run manifest writers
  paths.py                 # repo-root discovery, git_commit helper
  agents.py                # AgentConfig override fragment
  scripts/
    create_suite.py        # main() + cli() — materialize a suite from suites.yaml
    register_workspaces.py # main() + cli() — register existing workspaces
    run_benchmarks.py      # main() + cli() — run a suite through the pipeline
    copy_workspace.py      # main() + cli() — copy/symlink a workspace
    remove_benchmark.py    # main() + cli() — remove rows + workspaces from a suite
    create_report.py       # main()        — generate cross-run markdown report
    summarize_workspace.py # main()        — write bench_summary.json for one workspace
  generators/
    base.py                # WorkspaceGenerator ABC
    database.py            # RegistryGenerator (from suite.db)
    structural_causal.py   # StructuralCausalGenerator (tabicl SCM prior)
    gp_regression/         # GPRegressionGenerator (Matern GP prior, regression)
    gp_blackbox/           # GPBlackboxGenerator (Matern GP prior, blackbox)
  registry/
    schema.sql             # SQLite schema
    models.py              # Benchmark dataclass
    store.py               # connect/load helpers
    seed.py                # insert_benchmark_row
```

Each script that has both `main()` and `cli()`: `main()` is the verbose,
generic interface (invoked via `python -m alpha_lab.benchmarks.scripts.<name>`);
`cli()` is the simplified user-facing CLI bound to the `alpha-lab-*` entry
points in `pyproject.toml`.

---

## Adding suite definitions

Edit `suites.yaml`. Each top-level group declares a `generator` (import path or
absolute file path with `:ClassName`) and one or more named tiers; tiers
inherit the group's `generator` unless they override it. Suite paths are
slash-separated and map to nested YAML keys, e.g. `gp_regression/hard`:

```yaml
gp_regression:
  generator: alpha_lab.benchmarks.generators.gp_regression:GPRegressionGenerator
  hard:
    generator_kwargs:
      seed: 3000
      count: 32
      ...
    config_overrides:
      reasoning_effort: high
      ...
```

YAML anchors (`&name`) can be used for shared defaults across tiers.

For ad-hoc generators that live outside the package, `generator` accepts a
path-form import (e.g. `/scratch/wilsjame/my_gen.py:MyGenerator`) — `utils.py`'s
`resolve_import` handles either form.
