---
name: update_task_config
description: Apply user-approved changes to the run's task config during intake. Pass only the fields to change as a nested object; nested objects are deep-merged into the current config (recursively, at any depth), while scalars and lists replace. Unknown fields are rejected.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      updates:
        description: JSON object of config fields to change, deep-merged into the current config and re-validated. Nested objects are merged recursively at any depth; scalars and lists replace.
        type: object
    required:
      - updates
---
