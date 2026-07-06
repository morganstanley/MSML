---
name: memory_search
description: Search persistent memory for relevant knowledge from previous agents/phases. Returns summaries of matching entries. Use memory_read to get full content.
metadata:
  workspace_access:
    # rw, not ro: opening the SQLite memory index (memory.db) writes WAL sidecars
    ".memory": rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      limit:
        description: Max results (default 10).
        type: integer
      query:
        description: Search keywords.
        type: string
      tags:
        description: "Optional: filter by tags."
        items:
          type: string
        type: array
      kind:
        description: Optional filter by memory type. Prefer finding, decision, failure, result, hypothesis, constraint, or reference.
        type: string
      phase:
        description: Optional filter by pipeline phase.
        type: string
    required:
      - query
---
