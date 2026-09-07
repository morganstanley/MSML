---
name: memory_search
description: Retrieve persistent memories by query or recency. Returns summaries of matching entries. Use memory_read to inspect a result.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      query:
        description: Optional query text. Required for embedded/fulltext/raw_fts5 search; omit for default recency.
        type: string
      mode:
        description: |
          Search mode. Defaults to recency if query is omitted; otherwise, embedded if embeddings exist else fulltext.
          embedded — sorts by vector similarity against the embedded query
          fulltext — sorts by bm25 scores against the query as OR-joined keywords.
          raw_fts5 — sorts by bm25 scores against the query as an FTS5 expression, rejects malformed syntax
          recency — sorts by last update time, rejected if a query is passed
        type: string
        enum:
          - embedded
          - fulltext
          - raw_fts5
          - recency
      include:
        description: "Optional include filters. Allowed keys are tags, kind, agent, run_id, sources, owner. Tags/sources match contains-any. Kinds: fact (taken as true), idea (possible direction), unknown (known gap), pattern (observed trend), constraint (requirement)."
        type: object
      exclude:
        description: "Optional exclude filters with the same keys and matching semantics as include. Kinds: fact (taken as true), idea (possible direction), unknown (known gap), pattern (observed trend), constraint (requirement)."
        type: object
      src:
        description: Optional approved external memory source. Use a local .memory path, URL, or SSH git address; append #ref to select a branch, tag, or commit for git sources. Only use during intake warm-start after user approval.
        type: string
      start:
        description: Zero-based start index after filtering and ranking. Defaults to 0.
        type: integer
      limit:
        description: Maximum number of results to return. Defaults to 10.
        type: integer
---
