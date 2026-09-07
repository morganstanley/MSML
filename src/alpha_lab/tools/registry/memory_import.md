---
name: memory_import
description: Import one approved memory from an external source into the current workspace memory during intake warm-start.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      src:
        description: Approved external memory source. Use a local .memory path, URL, or SSH git address; append #ref to select a branch, tag, or commit for git sources.
        type: string
      memory_id:
        description: Memory entry ID in the external source.
        type: integer
    required:
      - src
      - memory_id
---
