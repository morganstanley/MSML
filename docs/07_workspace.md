# Workspace

A run operates inside one workspace directory (`--workspace`). Each phase writes
its outputs there, so by the end the workspace holds everything the run produced.
(The dataset and config live wherever you point `--config` and `data_path`, not in
the workspace.)

```
<workspace>/
├── adapter/             # the resolved domain adapter (Phase 0)
├── learnings.md         # accumulated learnings, updated continuously (Phase 1)
├── agenda.md            # user-proxy record: intake framing + handoff feedback
├── scripts/             # Phase 1 analysis scripts
├── plots/               # generated plots (Phase 1)
├── data_report/         # Phase 1 structured report
├── <framework_dir>/     # Phase 2 evaluation framework (e.g. backtest/, harness/); name set by the adapter
├── experiments/         # Phase 3 per-experiment working directories
├── playbook.md          # the strategist's accumulated playbook (Phase 3)
├── reports/             # Phase 3 milestone reports
├── output/              # deterministic generated documents (see below)
├── logs/                # agent logs (JSONL)
├── trace_info.json      # run + trace identity (run_id, trace_id, attempt)
└── .alpha_lab/          # run-local state
    ├── config.json      # the task config for this run
    ├── experiments.db   # the experiment board (SQLite)
    └── memory/          # the persistent memory store (see Memory)
```

The `<framework_dir>` name and the experiment file layout come from the active
[Adapter](03_adapters.md); the memory store is described in [Memory](06_memory.md).

## Generated output

`output/` holds the polished, human-facing documents produced after each phase by
`OutputGenerator` — purely deterministic extraction from the workspace, no LLM
calls:

| Document | Content |
|----------|---------|
| `01_data_exploration.md` | Phase 1 findings, schema, learnings, plots |
| `02_backtest_methodology.md` | Phase 2 framework design and baselines |
| `03_baseline_results.md` | Baseline metric tables |
| `index.md` | Table of contents for the above |

(Phase 3 milestone reports land in `reports/`.)

## Benchmark runs

When a workspace is materialized as part of a [Benchmark](08_benchmarks.md) suite,
it also carries a `benchmark_manifest.json` and a `private/` directory holding
held-out test data that's quarantined from the agent.
