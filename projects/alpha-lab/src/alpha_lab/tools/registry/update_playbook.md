---
name: update_playbook
description: "Write or update the playbook.md file in the workspace. The playbook contains accumulated strategic wisdom: what works, what doesn't, and what to try next."
metadata:
  workspace_access:
    playbook.md: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      content:
        description: Full text content for playbook.md.
        type: string
    required:
      - content
---
