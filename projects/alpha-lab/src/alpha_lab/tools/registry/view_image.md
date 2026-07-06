---
name: view_image
description: View a PNG or JPG image file from the workspace. Use this after generating plots to analyze them visually. The image will be displayed in the conversation for you to reason about.
metadata:
  workspace_access:
    ".": ro
  parameters:
    additionalProperties: false
    type: object
    properties:
      path:
        description: Path to the image file (absolute or relative to workspace).
        type: string
    required:
      - path
---
