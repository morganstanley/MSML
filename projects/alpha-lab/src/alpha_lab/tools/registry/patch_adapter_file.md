---
name: patch_adapter_file
description: "Patch (overwrite) a file in the workspace adapter directory. Creates a git checkpoint in the workspace before writing. Valid filenames: manifest.json, domain_knowledge.md, and prompt .md files."
metadata:
  workspace_access:
    adapter: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      content:
        description: New file content.
        type: string
      filename:
        description: Filename to patch (e.g. 'phase3_strategist.md').
        type: string
      reason:
        description: Reason for the patch.
        type: string
    required:
      - filename
      - content
      - reason
---
