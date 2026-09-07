# runcmp — design rationale

> **Start with [README.md](README.md)** — it is the user manual: what the
> suite does, every stage's inputs/outputs, the viewer, and how to add
> domains and harnesses. This file records *why* the suite is shaped the
> way it is, including the critique of the predecessor tooling that drove
> the design. Historical sections are kept verbatim; where the code has
> since moved on, the README is authoritative.

Purpose: given a corpus of **finished** Alpha Lab runs produced by different
harnesses (any number — two in-house variants originally; external
evidence-contract submissions since), decide **which architectural choices
are good and which are bad**, with evidence that survives fact-checking.
Works strictly off post-run logs; never writes into an input run.

## Why a redesign (what the msml machinery gets right and wrong)

The msml side has three tools: `compare_runs.py` (tabular pair monitor),
`posthoc.py` (6.6k-line deterministic two-run comparator + verdict), and
`posthoc_audit.py` (N-run forensic index). Their ingestion and epistemic
discipline (evidence pointers, explicit limitations, ties, read-only inputs)
are excellent and are kept here. Their judgment layer is not:

1. **The causal engine is three hardcoded hypothesis templates** (memory-tool
   failure rate ≥ 0.25, phase-2 window asymmetry, outlier count +2). Every real
   discovery in the historical reports (the SQLite thread-affinity memory bug,
   CPU starvation, validation-set discipline differences, advisory-only phase
   gates) was made by an *agent reading logs against a checklist*, then the
   machinery was patched to recognize that one pattern post-hoc.
2. **Task outcomes are almost always "not decidable"** — the comparator
   requires *exactly one* shared validation identity, and the two frameworks
   each build their own eval harness, so the gate can never open (more shared
   identities counterintuitively make it *less* decidable).
3. **The verdict is a fixed lexicographic gate ladder over process proxies**
   (validity → outliers → artifact completeness → verification-artifact *count*
   → token efficiency), with zero-tolerance axes where one missing file flips
   the winner.
4. **n = 1 per pair** — verdicts are about single runs, but the question is
   about *frameworks*. The corpus now has 5+ usable pairs plus solo runs whose
   failures are themselves evidence (msml died on m5 four times before a
   fix-budget bump; that is reliability data, not a missing pair).
5. **cond's `meta/` (Conductor) is never read** — decision log, directives,
   acks/retirements, annotations, rewinds, verify requests. Half of the
   architectural delta between the repos is invisible to the current tools.

## Architecture: the investigation is the machinery

```
corpus.py           Layer 0  run registry: every run, harness, domain, era,
                             pairing, completeness — messy reality explicit
extract.py          Layer 1  per-run evidence pack, streamed once, cached by
                             schema version; every section carries citations
tabulate.py         Layer 2  deterministic pair tables + corpus aggregates;
                             arithmetic only, no winners
referee.py          Layer 2  ONE evaluator re-scores every run's preserved
                             predictions per domain (the only cross-harness
                             quality evidence); rankability gates included
bench.py            Layer 2  versioned metric registry (~130 metrics x runs)
                             + failure-signature classification (rules.json)
token_accounting.py Layer 2  per-run, per-model token/cost ledger
probe_std.py        Layer 2b shared metric definitions investigator probes
                             import — canonical counting, never re-derived
investigate.py      Layer 3  single-writer LLM investigator, read-only tools;
                             findings.jsonl + REPORT.md (+ REPORT.html)
investigate_team.py Layer 3  planner/executor/critic variant (per-role model
                             assignment; critic gates every finding AND the
                             report — drafts scored 0-100 and counted for
                             hard audit violations; fewest violations then
                             best score publishes, even if the writer dies
                             mid-loop)
recompose.py        Layer 3  re-writes ONLY the report of a finished review
                             from its frozen findings (record_finding
                             refused; same report-critic gate; probes are
                             append-only and fingerprint-verified at exit)
factcheck.py        Layer 4  deterministic verifier: every citation in every
                             finding must resolve; plus report-level audits
                             (exit 0 = every hard audit clean, else exit 1)
meta_eval.py        Layer 4  scores whole investigations against key-fact
                             rubrics (for investigator-architecture trials)
publish.py          viewer   exports corpus+referee+bench into a standalone
                             MLflow instance (task leaderboards, harness
                             face-off, campaign reports)
```

Key inversions vs the msml design:

- **LLM as investigator, not editorial auditor.** posthoc.py caps the LLM at a
  2000-token accept/revise pass over a deterministic packet. Here the agent
  *drives*: it reads evidence packs, greps logs (bounded), queries the
  experiment DBs, runs probes over **all** items (never samples), and records
  findings. No fixed bucket taxonomy is imposed on it — the prompt states the
  mission and the evidence rules, not a rubric.
- **Determinism moves to the boundaries**: extraction below (what happened) and
  fact-checking above (does every cited number resolve). The middle is
  judgment, which is what the question actually requires.
- **Framework-level scope.** Findings carry scope ∈ {run, pair, framework,
  choice}. The final deliverable is the *choice ledger*: each observable
  architectural difference (Conductor layer, memory subsystem, verifier,
  fix-iteration budget, harness-building policy, validation discipline, gate
  wiring, …) judged good / bad / unresolved on corpus-wide evidence with a
  falsifier stated.
- **Ties and non-decidability remain first-class.** "Unresolved with reason"
  beats a forced verdict.

## What is deliberately kept from the msml tools

- Both DB locations (`ws/experiments.db`, `ws/.alpha_lab/experiments.db`),
  sha256 comparison when both exist, newest-wins on disagreement.
- Tool-failure counting only on standardized markers (`[ERROR]` /
  "tool execution failed" / "error executing tool" prefixes).
- OpenAI-shaped usage schema (`input_tokens_details.cached_tokens`,
  `output_tokens_details.reasoning_tokens`) as the normalized token form;
  raw Chat Completions and Anthropic shapes are converted into it.
- events.jsonl streamed with type-prefiltering; its `api_request` payloads
  are never parsed (per-agent transcripts are the parsed source of request
  payloads and model identity).
- Read-only inputs; output dir refused inside any input root.
- Cost model: fresh_input/cache_read/output at explicit prices; relative
  comparisons price-independent.

## What is deliberately dropped

- The lexicographic verdict ladder, the 0.42-similarity experiment matcher,
  the three hypothesis templates, closed failure taxonomies (raw normalized
  signatures are kept; categories are convenience labels, open set).
- Charts/HTML (not load-bearing for verdicts; can be added later).
- The exactly-one-validation-identity gate. Metric comparability is reported
  per validation identity; where artifacts permit re-scoring under one frozen
  identity a probe does it explicitly and says so.

## Lineup: the benchmark domain registry (`lineup.json` + `lineup.py`)

The set of comparison domains is data, not code. `lineup.json` holds one
entry per adopted benchmark: detection tokens (corpus.py maps run paths to
domain ids through them), the dataset path and task text handed to the
frameworks, the per-framework `domain` config field, the metric, an explicit
**evidence contract** (the files every scored experiment must preserve), and
the referee spec that re-scores those files. Adding a dataset to the lineup
is one JSON entry; code is only involved when the entry needs a referee
`kind` that does not exist yet.

Referee kinds and what "comparable" means for each:

- `bpb_curve`, `returns_parquet` (builtin) — self-report verification only;
  no shared truth exists across runs.
- forecast-npz (builtin, domain4) — cross-run re-scoring on a
  consistency-verified truth pool built from the runs' own preserved arrays.
- `classification_table` (d5_rfq) — cross-run by construction: the dataset
  itself is the frozen truth (labels + a holdout boundary declared in the
  spec), every run is scored on the identical slice; coverage below
  `min_coverage` of the holdout rejects the experiment.
- `kernel_bench` (d6_cuda) — cross-run by construction: one pinned task
  fingerprint; correctness is a hard gate (a report may tighten its
  tolerance but never loosen the spec's), and throughput is recomputed from
  the raw preserved timing samples, never trusted from the summary.

`python -m runcmp lineup --list` prints the lineup;
`lineup --out <campaign_dir> --domains ... --models short=provider:model,...`
emits per-cell run configs (framework-appropriate `domain` field, and the
same model in every seat of a cell — a benchmark run measures one model,
so the generator pins the cond conductor to the cell's own model rather
than letting it default to opus). It writes configs only; launching stays
a human act.

## Output layout (per campaign, outside all input roots)

```
<out>/
  corpus.json                Layer 0 registry
  packs/<label>.json         Layer 1 evidence packs
  tables.md / tables.json    Layer 2 pair + corpus tables
  referee.json               Layer 2 one-evaluator re-scoring + rankability
  bench.md / bench.json      Layer 2 metric registry + failure diagnosis
  token_accounting.md/.json  Layer 2 token/cost ledger
  <report>/                  Layer 3 one dir per investigation:
    probes/probe_NNN.py/.out    citable probe scripts + outputs
    findings.jsonl              raw findings (one JSON object per finding)
    REPORT.md / REPORT.html     the report, markdown + standalone HTML
    verification.json           Layer 4 fact-check results per finding
    findings.md                 Layer 4 human rendering of the checks
```

Invocation: `PYTHONPATH=. python -m runcmp <stage>`
(the bare command prints every stage); `scripts/runcmp_chain.sh` runs the
whole deterministic pipeline with post-condition checks. The investigator
stages are optional: the deterministic layers alone give tables, referee
verdicts, and the viewer, but not the written choice ledger.
