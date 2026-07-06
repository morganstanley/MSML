---
name: read_file
description: Read a file from the workspace. Returns numbered lines. Use offset and limit to read portions of large files.
metadata:
  workspace_access:
    ".": ro
  parameters:
    additionalProperties: false
    type: object
    properties:
      limit:
        description: Max number of lines to return (default 500).
        type: integer
      offset:
        description: Line number to start from (0-based, default 0).
        type: integer
      path:
        description: Path to the file (absolute or relative to workspace).
        type: string
    required:
      - path
---
