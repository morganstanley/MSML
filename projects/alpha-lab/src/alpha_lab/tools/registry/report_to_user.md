---
name: report_to_user
description: Call this ONLY when you have fully completed the entire analysis and have written all findings to the workspace files. This returns control to the user. Include a summary of everything you found.
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      summary:
        description: A comprehensive summary of all findings, key insights, data quality issues, and recommended next steps.
        type: string
    required:
      - summary
---
