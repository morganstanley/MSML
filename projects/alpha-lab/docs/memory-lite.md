# Alpha Lab Memory Lite

This design implements a lightweight local memory layer for Alpha Lab while keeping the records portable enough to outlive the current harness.

## Design principles

- Treat memory as a durable asset, not just prompt stuffing.
- Separate **portable canonical memory artifacts** from optional indexing/runtime acceleration.
- Make retrieval useful across runs, agents, and future harnesses.
- Preserve provenance so learned skills and findings remain reusable.
- Treat user-provided institutional knowledge as consent-first: propose, review, then store.

## What we intentionally avoid

- No external service.
- No embeddings or vector DB.
- No background worker / derivation queue.
- No extra runtime dependencies beyond Python stdlib.

## Canonical on-disk format

Under each workspace:

- `.memory/entries/*.md` — human-readable, self-describing memory records
- `.memory/index.json` — portable metadata index
- `.memory/memory.db` — optional SQLite cache/index, rebuildable from the canonical files

The important design choice is that the **portable artifacts are the source of truth**. In practice that means the self-describing entry files and the JSON index are what matter; the SQLite DB exists only to improve search quality. If Alpha Lab is replaced by another framework, the memory remains usable and the index can be rebuilt.

## Lightweight memory patterns

Alpha Lab uses three small memory patterns that fit the existing local pipeline architecture:

- **automatic capture of durable outputs** — high-value artifacts like `learnings.md`, Phase 2 review failures, and experiment `debrief.md` files are ingested into memory with provenance metadata
- **consented intake capture** — intake is the preferred test bed for reusable institutional knowledge; agents can propose a few `reference` topic notes from user-provided context and save only what the user approves
- **retrieval-first prompt recall** — prompts inject a compact "Relevant Prior Memories" section so agents see matching reference topics, prior findings, failures, decisions, and results before they start work

These are intentionally synchronous and local: no embeddings, no background derivation jobs, and no external memory service.

Search stays lightweight but now uses a small hybrid retrieval pass: SQLite FTS for token search plus a lexical substring fallback for domain strings that tokenizers can miss (for example concatenated error codes or metric names). Duplicate candidates are merged with a tiny reciprocal-rank signal, while the markdown entry files remain the durable source of truth.

Memory kinds are lightly normalized to keep retrieval predictable without adding a heavy schema. Preferred kinds are:

- `finding`
- `decision`
- `failure`
- `result`
- `hypothesis`
- `constraint`
- `reference`

Common aliases are accepted and normalized, e.g. `error`/`issue` → `failure`, `experiment_result`/`debrief` → `result`, `choice`/`conclusion` → `decision`, and `runbook`/`howto`/`topic` → `reference`.

## Curated topic knowledge

Some memory should be more deliberate than incidental agent notes. The main path for shared institutional knowledge should be intake-driven and consented: the intake flow is where users naturally explain access steps, project conventions, known gotchas, and workspace constraints, so it is the right place to suggest a small number of reusable notes and ask whether to save them.

Alpha Lab supports curated topic records for institutional knowledge such as:

- how to access a dataset
- where logs or dashboards live
- environment/authentication gotchas
- runbooks for common failures
- conventions that new agents or humans should know before starting work

Topic records live under `.memory/topics/*.md` as current, human-readable Markdown documents with JSON metadata. Approved intake suggestions and manual CLI saves use the same topic format. Each save is also indexed as a normal memory entry with:

- `kind=reference`
- a general `topic` tag
- a stable topic-specific tag like `topic_data_access_exchange_rates`
- `source_path=.memory/topics/<topic>.md`

This gives us a useful balance:

- `.memory/topics/` holds the current curated version for humans and future tools.
- `.memory/entries/` preserves memory history and makes the topic searchable through the existing memory APIs.
- `.memory/memory.db` remains an optional/rebuildable acceleration layer.

Prompt-time recall runs a small dedicated `kind=reference` search before the usual phase-specific memory searches, so relevant curated topic notes are more likely to appear without adding a new agent tool.

### Intake capture policy

The older assumption that topic knowledge would primarily be hand-entered via CLI is too narrow. The CLI is still useful for backfills and maintenance, but the better first product surface is intake:

1. The user provides config/workspace/task context.
2. The intake agent identifies at most a few reusable candidates, such as data access instructions or stable runbook gotchas.
3. The agent shows concise proposed notes and asks for consent.
4. Only approved notes are saved as `kind=reference` topic records.

Default posture is conservative:

- do not save secrets, tokens, credentials, personal/private details, or raw conversation logs
- prefer short synthesized notes over verbatim intake text
- default to not saving unless the user approves
- keep task-specific findings as normal run memory, not topic knowledge
- include owner / source / last-verified metadata when available

The CLI can still be used without an agent, mainly for manual curation or importing existing docs:

```bash
# Add or update a topic from a Markdown file
alpha-lab-memory --workspace ./workspace topic add data_access.exchange_rates \
  --file docs/exchange_rates_access.md \
  --title "Accessing exchange rate data" \
  --tag data_access \
  --owner research-platform \
  --mark-verified

# Search only curated topic records
alpha-lab-memory --workspace ./workspace topic search "exchange rate access"

# Read the current topic document
alpha-lab-memory --workspace ./workspace topic read data_access.exchange_rates

# Search all memories, including topic records
alpha-lab-memory --workspace ./workspace search "bedrock auth" --kind reference
```

## Why this fits Alpha Lab

Alpha Lab needs durable local research memory across:
- phase transitions
- strategist / worker / supervisor roles
- future framework or harness migrations

That maps better to a local, rebuildable, zero-extra-dependency memory layer than to a full service-oriented memory platform.

## Current tool contract

- `memory_store(content, tags, summary, kind?, phase?, agent?, run_id?, source_path?)`
- `memory_search(query, tags?, kind?, phase?, limit?)`
- `memory_read(memory_id)`

This preserves the existing agent architecture while making memory more portable and more searchable.
