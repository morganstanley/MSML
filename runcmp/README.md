# runcmp — the Alpha Lab benchmarking suite

Given a pile of **finished** autonomous-research runs — produced by any
number of harnesses, on any of the adopted benchmark tasks — this suite
answers, with evidence that survives fact-checking:

- **Which run did best on each task?** (one independent referee, one
  leaderboard per task)
- **Which harness is better overall, and at what?** (one face-off page
  across every task; written investigation reports)
- **Which model should drive a run?** (same-task, same-harness pairs)
- **What should each harness change first?** (ranked, evidence-cited)

It works **strictly off post-run artifacts** — it never launches runs and
never writes into an input run directory. Everything it produces is either
deterministic (re-running reproduces it) or LLM-written **and then
deterministically fact-checked**.

Design rationale and the critique of the predecessor tooling live in
[DESIGN.md](DESIGN.md). This file is the manual.

---

## 1. The trust model (read this first)

Three ledgers, never silently mixed. Every number the suite publishes
belongs to exactly one:

| ledger | produced by | may be compared across harnesses? |
|---|---|---|
| `referee.*` | ONE evaluator re-scoring every run's **preserved predictions** against frozen truth | **Yes — this is the only cross-harness quality evidence.** |
| `det.*` | mechanical extraction from each run's own logs/databases | Process facts (cost, timing, tool failures): yes, where both sides recorded them. Quality self-reports: **no.** |
| `inv.*` | LLM investigators | Only after `factcheck` verified the finding's citations. |

Two rules the suite enforces so you don't have to:

- **Rankability.** A run appears on a task leaderboard only if the referee
  could score it on the same frozen truth as the other runs. A run that
  measured itself on different validation data is *verified* (its
  self-reports are recomputed) but *unranked*, and its page says why.
- **Absence is a fact, not a zero.** A harness that recorded no token
  ledger has no cost bars — charts show nothing rather than zero, and
  parameters say "not recorded".

## 2. Vocabulary

- **Harness** (or *framework*): the autonomous-research system under test
  (e.g. `cond`, `msml`, or any external system submitting runs).
- **Run / cell**: one harness attacking one task once. "Cell" emphasizes
  its position in a campaign grid (harness x model x task).
- **Campaign**: one comparison batch — a set of runs indexed, scored, and
  published together under one name.
- **Domain / task**: one benchmark problem (dataset + task text + metric +
  referee spec), declared in `lineup.json`.
- **Evidence pack**: one JSON file per run distilling its artifacts —
  experiments, trajectories, transcripts statistics, seat assignments,
  failures. Everything downstream reads packs, not raw runs.
- **Referee**: the deterministic re-scorer. Re-computes every experiment's
  score from preserved prediction artifacts against frozen truth.
- **Seats**: which model actually performed each agent role in a run
  (observed from the run's own records — declared config is tracked
  separately as intent).
- **Validation identity**: the identifier of the frozen data a score was
  measured on (`val_slice_id`, content hashes). Same identity = same ruler.

## 3. Quickstart: finished runs → leaderboards

Requirements: Python 3.11+ with this repo's `requirements.txt` (the
`mlflow` and `matplotlib` extras are needed only by `publish`; `pandas` +
`pyarrow` only by table-based referees).

```bash
export PYTHONPATH=.
OUT=comparison_out/my_campaign

# 0) See the adopted benchmark tasks
python -m runcmp lineup --list

# 1) Discover finished runs under one or more roots -> the registry
python -m runcmp index \
    --root /data/runs_batch_A --root /data/external_submissions \
    --framework-alias alphalab=alphalab2 \
    --out $OUT/corpus.json
# (optionally filter $OUT/corpus.json down to the runs you mean to compare)

# 2..5) Evidence packs, tables, referee, metric registry, token ledger —
#        one script, with post-condition checks (pairs are optional 2-way
#        verdict overlays; the referee scores every run either way):
bash scripts/runcmp_chain.sh "$OUT/corpus.json" "$OUT" \
    "d4_glm=eraA/domain4/cond/run1:eraA/domain4/msml/run2"

# 6) Publish to the viewer (a standalone MLflow instance)
python -m runcmp publish \
    --corpus $OUT/corpus.json --packs $OUT/packs \
    --bench $OUT/bench.json --referee $OUT/referee.json \
    --out $OUT --store showcase_mlflow --campaign my_campaign

# 7) Serve and browse
mlflow server --backend-store-uri sqlite:///showcase_mlflow/mlflow.db \
    --default-artifact-root "$PWD/showcase_mlflow/artifacts" \
    --host 127.0.0.1 --port 5601
# open http://localhost:5601
```

The written scorecards are already on disk after step 5: `bench.md`
(opens with one referee leaderboard per task), `tables.md`,
`token_accounting.md`.

## 4. The pipeline, stage by stage

Run any stage as `python -m runcmp <stage> --help`.
The bare command prints the full stage list.

### `lineup` — the benchmark task registry
`lineup.json` (next to the code) declares every adopted task: detection
tokens, dataset path, task text with its **evidence contract**, metric +
direction, per-harness `domain` config value, and the referee spec.
`lineup --list` prints them; `lineup --out <dir> --domains ... --models
short=provider:model,...` generates per-cell run configs (one model in
every seat by design). **`data_path` values are deployment-local** —
point them at your copies of the datasets.

### `index` — discover runs → `corpus.json`
Walks `--root` (repeatable) and registers every run it recognizes:

- **in-house workspaces** — a directory with `experiments.db` /
  `.alpha_lab` / `adapter` markers; completeness = the database holds
  terminal experiment rows (a present-but-empty database is not a usable
  run, and a newer zero-byte database never shadows the populated one);
- **evidence-contract submissions** (any external harness) — a directory
  shaped `experiments/<name>/results/metrics.json`; completeness = every
  experiment directory parses. `--framework-alias OLD=NEW` unifies
  inconsistently-stamped run-directory names.

The printed report shows every run with its completeness; incomplete runs
stay in the registry (a failed run is evidence, not a gap).

Each run also records a `run_state` (`finished` / `in_flight` / `unknown`,
read from the launcher log's exit line). A run still writing rows is marked
`[IN FLIGHT]` in the printout and is never paired — comparing a live run
against a finished one measures elapsed time, not research quality.
`--finished-only` drops such runs from the registry entirely.

### `extract` — one evidence pack per run
Streams each run's artifacts once into `packs/<label>.json`: experiment
rollup (totals, best, trajectory, validation census), transcript
statistics (per-seat models, request composition, latency samples), token
ledger seats, conductor governance record (where the harness has one),
code metrics (AST-level, per experiment), failure signatures, timings.
The request-composition census understands all three payload dialects —
Anthropic content blocks, OpenAI Responses items, and lab-endpoint
chat-completions messages (role="tool" results, `tool_calls` arguments,
`reasoning_content`, `image_url` parts) — so images/tool/thinking fields
are populated for kimi/glm runs, not silently zero. ZDR reasoning items
(empty `summary`/`content` arrays + an `encrypted_content` blob) count
into `thinking` by serialized size: the text is unreadable by design,
but the bytes are re-sent and billed on every deep request. Packs also
carry `agent_logs.reasoning_roundtrip` (requests/responses totals and
how many carried a reasoning trace) — surfaced as the bench metrics
`context.reasoning_produced_share` / `context.reasoning_returned_share`
and a standing check in every auto-mission: a run that produces
reasoning but never returns it is running blind to its own traces.
Gzip-compressed artifacts are read transparently. Packs are **schema
versioned**: after an extractor upgrade, stale or truncated packs
re-extract themselves (no `--force` needed; `--force` re-extracts
everything). `--all` includes incomplete runs (the chain script uses it).

### `tabulate` — deterministic pair tables
`tables.md`/`tables.json`: side-by-side pair tables (`--pair
NAME=LEFT_LABEL:RIGHT_LABEL`, repeatable) plus corpus-level aggregates.
Arithmetic only; no winners are declared. `MODEL_RATES` at the top of
`tabulate.py` is the **deployment-local price table** (USD per 1M tokens)
— edit it for your models; unknown models get the most expensive rate so
naming drift inflates costs loudly instead of zeroing them.

### `referee` — the only cross-harness quality evidence
Re-scores **every run in every domain** against that domain's truth:

- shared-origin forecast domains: one truth pool per domain built from
  all runs' preserved arrays, cross-verified (any conflict is counted and
  reported), every prediction re-scored against it;
- frozen-table domains (classification/regression): every prediction
  table scored against the dataset's frozen holdout;
- kernel domains: correctness-gated recomputation from raw timing samples;
- self-report domains (bits-per-byte, returns): each run's claims are
  recomputed from its preserved counts, with **rankability** decided by
  validation identity (multiple identities within one run, or an identity
  differing from one that several runs share exactly → verified but
  unranked, with the reason recorded).

`--pair NAME=LEFT:RIGHT` (repeatable) adds 2-way verdict records on top;
the N-way scoring happens regardless. Output: `referee.json`.

### `bench` — the metric registry
~130 versioned metrics per run (`metrics_manifest.json` is append-only —
bench refuses to run if a registered metric disappears), failure
signatures classified by `rules.json` into bug / design / policy /
external, and `bench.md` — the written scorecard, opening with one
referee leaderboard per task (unrankable runs listed beneath with the
referee's reason).

### `token-accounting` — tokens and cost per run, per model
Merges the run's own event telemetry with optional MLflow Bedrock traces;
absent ledgers stay absent. Costs come from `MODEL_RATES`.

### `publish` — the viewer
Exports everything into a standalone MLflow store (`--store DIR`):

- **one experiment per task** — the home page is the benchmark index; a
  task page is ONE leaderboard: every run of every campaign, one row per
  run (a later republish of the same run replaces its row);
- **metric names carry the chart layout** — the charts tab auto-draws one
  chart per metric, grouped into named sections (`1 verdict`, `2 race`,
  `3 economy`, `governance`, `0 seats`, ...). No hand-built charts; new
  metrics slot in automatically;
- **`harness face-off — every task, one page`** — one row per harness:
  mean-percent-behind-winners (the single-glance answer), per-task gaps
  and best scores in native units, per-attempt race series (x-axis
  switchable to wall-clock), recorded cost/hours, runs fielded;
- **`campaign reports`** — one row per campaign holding `bench.md`,
  grids, and the investigation reports with their fact-check counters;
- **seats everywhere** — `seat.<role>` parameters (observed models per
  agent role) and a "0 seats" chart section; mixed-seat runs state their
  lineup in the description. Seat swaps are lineup data, not alarms.

Republish is idempotent. `--retire-campaign NAME` (repeatable) removes a
superseded campaign's leftover task rows (recoverable mark-deletes);
campaign report pages are never touched.

### Day-2 helpers — `status`, `watch`, `preflight`, `validate-submission`, `mission --check`, `rereview`

Small stages for the questions otherwise answered with one-off scripts:
`status` (one look at a review directory: critic rounds with scores and
defect counts, report and stub check, verification), `watch` (every run
under a root: state, board counts, best-so-far), `preflight` (pre-launch
checks — corpus roster, pack coverage/staleness, one tiny credential call
per `--provider`, store writable; exit 1 on any blocker), and
`validate-submission` (evidence-contract layout check for external
harnesses). `mission --check FILE` lints a hand-written mission: numbered
`N. **Theme**` sections must parse (the stub check keys off them) and any
run labels it names must exist in the corpus. `rereview` re-writes a
report with the review's OWN writer, read from `sessions.jsonl` — every
investigator stage appends its identity there, so a rewrite can no longer
silently switch a review to a different model.

## 5. Reading the viewer

- **Home page** lists `task ...` experiments (one per benchmark task); the
  description preview is the champion tagline.
- On a task page: the **table** is the leaderboard (the description's
  "Current standings" link opens it pre-sorted by referee score with the
  right columns); the **middle icon above the table** switches to charts —
  every quantity drawn, sections in story order, `Search metric charts`
  finds anything by word.
- If a page opens looking empty with "Traces"/"Sessions" tabs, click the
  **Model training** toggle (top-left) — some MLflow versions land on a
  GenAI view these runs don't use.
- Each run row: description = headline facts in plain words; Artifacts tab
  = the run's rollout timeline picture; metrics table = every number.

## 6. Submitting runs from any harness — the evidence contract

Any system can be benchmarked by dropping a directory per run:

```
<your_run_dir>/
  README.md                            # how it was produced (prose)
  experiments/
    <experiment_name>/
      results/metrics.json             # REQUIRED, see below
      results/<referee artifact>       # per-domain, see below
      <code files>                     # optional but strongly encouraged
```

`metrics.json` must carry the domain metric under any reasonable spelling
(`rmse` / `overall_rmse` / `holdout_log_loss` are all matched by
normalized name). Strongly recommended fields:

- `val_slice_id` (or a content hash): the frozen-data identity — this is
  what makes scores rankable;
- `started` / `finished` timestamps (otherwise timing falls back to file
  times);
- `approach`: one paragraph on what the experiment tried (becomes the
  run's hypothesis text);
- `wall_clock_seconds`, `param_count` where meaningful.

Per-domain referee artifacts (see each `lineup.json` entry's `referee`
spec for the authoritative globs):

- forecast domains: `results/referee_predictions.npz` with `predictions`,
  `origins`, `series_ids`, and (if you have it) `truth` — your truth is
  cross-verified against every other run's;
- classification/regression tables:
  `results/referee_predictions.parquet` with the id column + prediction
  column named in the spec, covering the frozen holdout;
- kernel domains: `results/kernel_report.json` per the task contract.

What you get without transcripts/token ledgers: full quality scoring and
ranking, code metrics, coarse timing. What you forgo: cost/economy
comparisons, seat observation, race-over-wall-clock detail. To join those
panels, also preserve per-request records (model name + token counts +
timestamps, any JSONL) — see `extract.py`'s seat ledger reader.

## 7. Adding a benchmark task

One entry in `lineup.json`: id, title, `detect` tokens (how run paths map
to the task), `resource`, `metric` (`{"name", "lower_is_better"}`),
`data_path` (your dataset), `task` text **including the evidence
contract**, `framework_domains` (what each harness calls the domain), and
a `referee` spec. If the spec's `kind` already exists
(`classification_table`, `regression_table`, `kernel_bench`, plus the
builtin forecast/bpb/returns kinds) no code is needed. A new kind means
one verification function in `referee.py`.

## 8. The investigation layer (LLM reviewers)

Deterministic tables say *what*; investigators argue *why it matters* —
and every claim they record must carry machine-checkable citations that
are validated at record time AND re-verified afterwards.

```bash
# single writer (mission is generated automatically — see below)
python -m runcmp investigate \
    --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT/review_solo \
    --provider <provider> --model <model>

# planner/executor/critic team (per-role model assignment)
python -m runcmp investigate-team \
    --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT/review_team \
    --provider <provider> --model <model> --reasoning-effort high \
    [--role executor=<provider>:<model>] [--role critic=none]

# deterministic fact-check of a finished report
python -m runcmp factcheck \
    --out $OUT/review_solo --packs $OUT/packs

# re-write ONLY the report of a finished review, from its frozen findings
python -m runcmp recompose \
    --out $OUT/review_team --corpus $OUT/corpus.json --packs $OUT/packs \
    --provider <writer-provider> --model <writer-model> \
    [--role critic=<provider>:<model> | --role critic=none]
```

- **Missions** scope the investigation (the decisions to deliver + what
  the corpus contains) and are **generated by default** from the
  deterministic artifacts — corpus registry, referee rankability,
  replication groups, available scorecards — so the corpus description
  can never go stale. `--mission auto:<focus>` picks the decision preset:
  `harness` (default: harness/model/combination/capability + what to
  change first), `variability` (run-to-run variance over the corpus's
  replication groups), `gate` (does one specific change clear a bar),
  `treatment` (before/after one deliberate change), or `union` (the
  all-encompassing report: section themes are the union of what the
  per-battery reports actually carried, at full depth). Facts the registry cannot know (a config knob one
  wave carried, a serving incident) go in repeatable `--note` flags and
  are printed as operator-declared. A note of the form `--note @path`
  reads the file instead (one note per non-blank line) — use this from
  shell wrappers: an apostrophe inside a quoted `--note` string has
  silently killed reviewer launches through nested `bash -c` quoting. Whatever text scoped a report is
  written to `<out>/mission.md` for the record. Each successful
  investigation also refreshes `REPORTS.md` at the campaign's top level —
  an index of every written report beneath it, with its verification
  counters — so the reports are findable without knowing the layout.
  Preview or reuse one without launching anything:

  ```bash
  python -m runcmp mission \
      --corpus $OUT/corpus.json --focus variability \
      --note "wave-2 cond runs carry a max-pending-proposals cap of 4"
  ```

  Hand-written missions remain supported (`--mission @file.md` /
  `--mission none` for the bare standing orders) — see
  [missions/TEMPLATE.md](missions/TEMPLATE.md); never splice an old
  mission's corpus description forward.
- The investigators' tools are read-only (packs, SQL over experiment
  databases, bounded log search, python probes); probes importing
  `probe_std` get the deterministic layer's own metric definitions so
  numbers never fork.
- Reports are refused at write time until they meet chart floors and pass
  the built-in audits (registry echo, progression-series fidelity,
  referee-attribution verbatim). In team mode a **report critic** also
  reads every submitted draft against the mission and scores it 0–100:
  drafts that don't deliver bounce back with required changes while
  iteration and wall-clock budget remain, every draft is kept on disk
  under `report_drafts/` with its score AND its count of hard audit
  violations, and when the gate stops bouncing the winner is the draft
  with the **fewest violations, best score among those** — never the
  longest, never merely the last, and never a higher-scored stale draft
  over one whose defects the writer already fixed. If the writer becomes
  unable to submit at all (context exhausted, API dead, iteration cap),
  the same winner still publishes; a review cannot end report-less by
  construction.
- `factcheck` re-verifies every finding and renders `findings.md` +
  verification counters. Its **exit status is the verdict**: `0` only when
  every hard audit is clean (all findings re-verify, no unsourced chart
  series, no referee-attribution violations, every chart renders);
  anything flagged exits `1` with a `FACTCHECK FLAGS` line naming why —
  command success and report success are the same thing, by construction.
- Every published report ends with its **production cost** (dollars and
  tokens by seat, from the session's own usage ledger `usage.jsonl` at the
  shared list prices); the fact-checker exempts that footer from the body
  audit. Exploration is measured, not just timed: packs carry per-seat
  web-search queries and phase-0/1 product sizes (`inventory.exploration`),
  the `exploration.*` bench family puts them on the cross-run boards, and a
  standing mission check makes reviewers grade whether the early phases
  earned their hours.
- The runs' own writing is measured too: a per-run report census
  (`inventory.reports`: bytes/tables/numbers/images over report documents
  and debriefs) and a readership ledger (`agent_logs.artifact_reads`:
  which seats opened which written products) feed the `reporting.*`
  metric family; a standing mission check makes reviewers flag written
  products nobody read.
- `recompose` re-writes only the report of a finished review: findings and
  ledger are frozen (`record_finding` refused), the writer re-reads probes
  for exact numbers, and the same report-critic gate applies. The replaced
  report is kept as `REPORT.superseded_<ts>.md`, a previous session's
  drafts are archived (never overwritten), and a rewrite that fails with
  nothing to salvage puts the previous report back. Two integrity guards
  hold throughout: probe files are append-only (a new session continues
  numbering after everything on disk, and the write site refuses to reuse
  an existing name), and recompose fingerprints every frozen-evidence file
  at start (`recompose_preflight_<ts>.json`) and fails loudly at exit if
  any of them changed.
- `meta-eval` scores whole investigations against key-fact rubrics — for
  comparing investigator configurations, not for everyday use.

#### Four things that decide whether a batch of reports is worth reading

Learned the hard way: an earlier batch passed every mechanical gate and was
still useless. What separates a good batch from that one:

1. **Put the deliverable's requirements in the mission, not in `--note`.**
   Notes are advisory context; the mission is the contract the investigator
   works to. If the report must cover a specific cube of models × domains ×
   harnesses or a specific section list, it belongs in the mission — derive
   it (`--mission auto:union` derives the section list from the reports that
   already exist) rather than describing it.
2. **Give it enough turns.** Budgets scale with the number of questions and
   with the corpus; a review whose iteration budget or critic-refund pool is
   sized for four questions produces a thin report on twenty and still
   verifies clean. Check `plan.md`'s question count against the run's
   budget before trusting a short report.
3. **Never let a dead run into the corpus.** `index` records
   `archived_attempts_skipped`; read it. A run that was resumed, degraded by
   interference, or superseded by a relaunch must be archived by rename
   *before* indexing, and a pair emitter must refuse duplicate labels rather
   than overwrite. Reviews spent on dead runs are reviews wasted twice: the
   report is wrong and the reader trusts it.
4. **Read `<out>/verification.json`, not just the report.** `total` must
   equal `verified` with `failed: 0`. Note what that proves: every citation
   re-resolves against the packs — not that the interpretation is right.

Providers/models are whatever your `runcmp.llm.client.get_provider`
supports; investigations need LLM credentials and real time (hours) and
money — the deterministic layers never do.

## 9. The change gate — proposing a harness change (the PR path)

You changed the harness and want it adopted. The paved path:

```bash
# 1) run the benchmark campaign twice: once with the old harness (or reuse
#    an existing baseline campaign), once with yours — runs land in two
#    distinguishable places (e.g. two era directories)

# 2) one command: index, score, screen, review, publish
GATE_STORE=showcase_mlflow \
GATE_REVIEWERS="<provider>:<model> <provider>:<model>" \
scripts/runcmp_gate.sh gate_out \
    era=july_baseline era=myfix_0807 \
    /data/runs_baseline /data/runs_candidate
```

What comes out, and the philosophy behind it:

- **The screen (deterministic).** `runcmp gate` pairs the two sides per
  (task, model) cell and evaluates `gate_policy.json` — guard dimensions
  the change must not hurt (final referee score, wall clock, cost,
  failures, code quality — tolerances all data, deployment-local) and
  improvement dimensions it must move (minimum gains, ditto). Output:
  `gate.md` (the evidence table), `gate.json` (machine-readable), and an
  exit code CI can use. **The screen is not the decision**: the change is
  multidimensional and single runs carry wide run-to-run spread, so the
  page presents per-dimension deltas under one sign convention and states
  the mechanical reading for what it is.
- **The reviewers (LLM, opinionated on purpose).** The gate writes
  `mission.md`; each `GATE_REVIEWERS` entry runs an investigator over the
  same corpus with that mission, and the mission REQUIRES the report to
  open with an explicit `Recommendation: PASS` or `Recommendation: FAIL`
  plus a confidence grade and the evidence that would flip it. Several
  reviewers, same mission — positions you can compare. Each report is
  fact-checked deterministically afterwards.
- **The evidence page (native MLflow).** `publish --gate gate.json` adds
  one run to the `change gates — PR evidence` experiment: the name
  carries the flag/improvement counts, the charts are the per-dimension
  deltas per task, the description is the full table plus every
  reviewer's recommendation line, and the full reviews hang under the
  run's Artifacts. The humans deciding the PR read this one page; the
  underlying runs stay browsable on the task pages and the face-off.

Tune the bar in `gate_policy.json` (tolerances, minimum gains, which
metrics are guards at your site); the gate refuses unknown metric ids
loudly. No wrapper script? It only automates the standard stages — run them
individually: `index` over both roots (`--finished-only`), `extract`,
`tabulate`, `referee`, `bench`, then `gate` with `--baseline`/`--candidate`
selectors, then `investigate`/`investigate-team` with
`--mission @<gate_out>/mission.md` per reviewer, then `factcheck` (exit 0
required), and optionally `publish --gate gate.json`. `--allow-model-mismatch` exists for the rare cross-model
comparison and is off by default — a model change is a different
experiment, not a harness change.

## 10. Reproducibility guarantees

- `index`/`extract`/`tabulate`/`referee`/`bench`/`token-accounting` are
  pure functions of the run artifacts: re-running reproduces them
  (timestamps excepted).
- Packs carry `schema` versions; the cache self-invalidates on upgrades.
- Publish is idempotent by run identity; the store is disposable — delete
  it and republish from disk at any time.
- `metrics_manifest.json` guards against silent metric loss; `rules.json`
  versions failure classification; `bench.json` records both versions.

## 11. Deployment-local configuration (edit these for your site)

| what | where |
|---|---|
| benchmark tasks + dataset paths | `lineup.json` (`data_path` per entry) |
| model prices (USD / 1M tokens) | `MODEL_RATES` in `tabulate.py` |
| failure classification | `rules.json` |
| viewer store location / port | `publish --store`, your `mlflow server` flags |
| LLM providers for investigators | your `alpha_lab` provider config / env |

## 12. Troubleshooting

- **Publish fails with `ModuleNotFoundError: mlflow`** — publish is the
  only stage needing mlflow; install it in the python you run publish with.
- **A task page looks empty** — you are on the GenAI/Traces view; click
  the "Model training" toggle (top-left).
- **A run you expected is missing from a leaderboard** — check `index`'s
  printed completeness for it, then its `rankable` flag in `referee.json`
  (the reason string says exactly why).
- **`referee` fails with `PermissionError` on a dataset path** — frozen-
  table referees read the task's dataset itself (the `data_path` in
  `lineup.json`) to rebuild truth; run the chain from an environment that
  can read every listed `data_path` (sandboxed shells often cannot).
- **Metric names**: MLflow rejects `%`, `=`, and em-dashes in metric keys
  — the suite spells them out (`pct`, `:`); do the same in extensions.
- **First viewer start is slow** — MLflow can take minutes to serve on
  networked filesystems; it is not hung.
- **Same harness, differently-stamped run dirs** — `index
  --framework-alias OLD=NEW` unifies the label.

## 13. Package map

| file | role |
|---|---|
| `corpus.py` | run discovery + registry (`index`) |
| `extract.py` | evidence packs |
| `tabulate.py` | pair tables + price table |
| `referee.py` | independent re-scoring + rankability |
| `bench.py` | metric registry + `bench.md` |
| `token_accounting.py` | token/cost ledger |
| `publish.py` | the MLflow viewer exporter |
| `investigate.py` / `investigate_team.py` | LLM reviewers |
| `factcheck.py` | deterministic verification of findings/reports |
| `meta_eval.py` | rubric scoring of whole investigations |
| `probe_std.py` | canonical metric definitions for probes |
| `render_html.py` | report markdown → standalone HTML |
| `gate.py` / `gate_policy.json` | the change gate (PR screen + reviewer mission) |
| `lineup.py` / `lineup.json` | benchmark task registry |
| `rules.json` / `metrics_manifest.json` | versioned judgment data |
| `missions/TEMPLATE.md` | how to write an investigation mission |
| `DESIGN.md` | why the suite is shaped this way |
