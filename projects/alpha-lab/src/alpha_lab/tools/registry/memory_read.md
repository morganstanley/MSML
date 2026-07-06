---
name: memory_read
description: Read the full content of a specific memory entry by ID. Use memory_search first to find relevant entry IDs.
metadata:
  workspace_access:
    # rw, not ro: opening the SQLite memory index (memory.db) writes WAL sidecars
    ".memory": rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      memory_id:
        description: The memory entry ID.
        type: integer
    required:
      - memory_id
---
