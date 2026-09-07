---
name: memory_read
description: Read the display content of a specific memory entry by ID. Long entries may be truncated for tool output. Use memory_search first to find relevant entry IDs.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      memory_id:
        description: The memory entry ID.
        type: integer
      src:
        description: Optional approved external memory source. Use a local .memory path, URL, or SSH git address; append #ref to select a branch, tag, or commit for git sources. Only use during intake warm-start after user approval.
        type: string
      start:
        description: Zero-based character offset into the rendered memory display. Defaults to 0.
        type: integer
      stop:
        description: Character offset where the rendered memory display should stop. Defaults to 10000. Use null to read through the end.
        oneOf:
          - type: integer
          - type: "null"
    required:
      - memory_id
---
