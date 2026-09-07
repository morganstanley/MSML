# Memory

Alpha Lab keeps a lightweight, local memory so what a run learns — findings,
patterns, reusable skills — survives across phases, agents, and even future
harnesses. It is treated as a durable asset rather than prompt-stuffing: canonical
records are kept separate from the rebuildable search index, retrieval works
across runs, provenance is preserved so knowledge stays reusable, and
user-provided institutional knowledge is stored consent-first (propose → review →
store). Two classes carry this: `Memory`, a single record, and `MemoryStore`, the
read/write API over a collection of them.

## Memory

A `Memory` is a single persistent record. Its key is assigned by the store (it
isn't a field on the record), and it carries:

| Field | What it is |
|-------|------------|
| `kind` | Its category — see the table below. |
| `summary` | A one-line summary, used for search and display (indexed). |
| `content` | The full memory body (indexed). |
| `tags` | Normalized, searchable labels. |
| `agent`, `run_id`, `owner`, `sources` | Provenance; `owner` defaults to the current user. |
| `frozen` | Opt-in immutability — a frozen record cannot be edited. |
| `created_at`, `updated_at` | Timestamps. |

`tags` and `sources` are normalized (lower-cased, de-duplicated). Editing a record
returns an updated copy — scalar fields are replaced while `tags`/`sources` union
with the existing values — and version history is just the git log.

Each record has a **kind**:

| Kind | Use for |
|------|---------|
| `fact` | Something established as true — a verified finding or property you can rely on. |
| `idea` | A possible direction or hypothesis worth trying, not yet validated. |
| `pattern` | An observed regularity or trend across results or experiments. |
| `constraint` | A requirement or limit the work must respect. |
| `unknown` | An explicit, known gap — something you've determined you *don't* know. Not a catch-all for uncategorized notes. |

## MemoryStore

A `MemoryStore` is the read/write API over one memory root — a thin,
memory-specific facade over a `ModelDB`, which owns records, the SQLite index,
embeddings, git, and locking. It can also reject a new memory that is too similar
to an existing one, but only when asked: `max_similarity` defaults to `1.0`, and
cosine similarity cannot exceed that, so nothing is rejected unless a caller
passes a lower threshold.

### Search

`MemoryStore.search(query=None, *, mode=None, include=None, exclude=None, start=0, limit=10)`:

- default query search: `embedded` if embeddings are available, otherwise `fulltext`
- `mode="embedded"`: vector similarity against the embedded query; requires a query
- `mode="fulltext"`: bm25 against the query as OR-joined keywords; requires a query
- `mode="raw_fts5"`: bm25 against the query as an FTS5 expression; requires a query
- `mode="recency"`: newest updated memories first; rejects a query
- no query and no mode: recency

`fulltext` quotes each whitespace-separated keyword before it reaches FTS5, so
punctuation is searched rather than parsed, and joins them with `OR` so bm25 ranks
partial matches instead of every keyword being required. That matters most when
embeddings are unavailable and `fulltext` becomes the default for prose queries
written with embedded search in mind. `raw_fts5` passes the query through, making
FTS5's operators, prefix terms and column filters available — and its syntax
errors visible to the caller.

`include` and `exclude` filters apply before ranking. You can filter on any record
field — e.g. `tags`, `kind`, `agent`, `run_id`, `sources`, `owner` — with scalar
fields matched exactly and sequence fields (`tags`, `sources`) matched
contains-any; multiple values OR within a field. Recency is ordered by
`updated_at` descending, with key order as a tiebreaker.

### Storage layout

A memory root lives at `<workspace>/.alpha_lab/memory/`, where `ModelDB` keeps:

```text
records/<key>.json   # canonical, committed records — the source of truth
index.db             # derived SQLite metadata + FTS5 index; rebuildable
embeddings/          # derived vector pages for embedded search; rebuildable
```

Only `records/` is authoritative; `index.db` and `embeddings/` are acceleration
layers that stay out of git and can be rebuilt from the records. `ModelDB.build()`
re-derives them: it rebuilds the SQLite/FTS index and gap-fills embeddings,
re-embedding a record only when its vector is missing or its stored content
fingerprint no longer matches — so a rebuild makes no embedding calls in steady
state. Each vector row carries that fingerprint (column 0), so a reused or
re-seeded workspace can't keep a stale vector. External approved sources may point
at another memory root or a git repository — seeded per run via `memory_spec` in
the [Configuration](02_configuration.md). Tools stage external sources through the
parent process, so agents never read or write `.alpha_lab` directly.

## Tools

Agents access memory through [Tools](05_tools.md), not filesystem paths:

- `memory_store(content, tags, summary, kind, agent?, run_id?, sources?, owner?, frozen?, max_similarity?)`
- `memory_search(query?, mode?, include?, exclude?, src?, start?, limit?)`
- `memory_read(memory_id, src?, start?, stop?)`
- `memory_import(src, memory_id)`

`src` is an approved external memory source — a local memory root, URL, or SSH git
address; append `#ref` to select a branch, tag, or commit for git sources.
`memory_read.start` / `memory_read.stop` are character offsets into the rendered
memory; `memory_search.start` / `memory_search.limit` are result-window controls.

## Capture patterns

Alpha Lab captures memory three ways:

- automatic capture of durable outputs such as `learnings.md` and experiment `debrief.md`
- consented intake capture for reusable institutional knowledge
- retrieval-first prompt recall through compact relevant-memory sections

The default posture is conservative:

- do not save secrets, credentials, private details, or raw conversation logs
- prefer short synthesized notes over verbatim text
- store user-provided institutional facts only after explicit user approval
- keep task-specific findings as normal run memory, not curated facts

## CLI

Manual or backfill saves use the workspace memory root under `.alpha_lab/memory`:

```bash
alpha-lab-memory --workspace ./workspace store \
  --kind fact \
  --tag auth \
  --tag bedrock \
  --summary "Bedrock auth uses PingFed token flow" \
  --content "..."
```

Search examples:

```bash
alpha-lab-memory --workspace ./workspace search "validation leakage"
alpha-lab-memory --workspace ./workspace search "validation leakage" --mode fulltext
```
