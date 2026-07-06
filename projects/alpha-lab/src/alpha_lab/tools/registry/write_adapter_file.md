---
name: write_adapter_file
description: "Write a file to the workspace adapter directory. Valid filenames: manifest.json, domain_knowledge.md, and the 9 prompt files (phase1.md, phase2_builder.md, etc.)."
metadata:
  workspace_access:
    adapter: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      content:
        description: File content to write.
        type: string
      filename:
        description: Filename to write (e.g. 'manifest.json', 'phase1.md').
        type: string
    required:
      - filename
      - content
---
