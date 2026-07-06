---
name: grep_file
description: Search files in the workspace using grep. Returns matching lines with file paths and line numbers.
metadata:
  workspace_access:
    ".": ro
  parameters:
    additionalProperties: false
    type: object
    properties:
      include:
        description: "Glob pattern to filter files (e.g. '*.py')."
        type: string
      path:
        description: "Directory or file to search (relative to workspace, default '.')."
        type: string
      pattern:
        description: The search pattern (regex).
        type: string
    required:
      - pattern
---
