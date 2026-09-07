---
name: complete_research
description: "End Phase 3 successfully only when the existing evidence shows that no additional scientifically admissible experiment should be run. This is not a substitute for proposing work or reporting progress. The request is rejected while execution work is active or before any real non-smoke experiment has produced results."
metadata:
  parameters:
    additionalProperties: false
    type: object
    properties:
      reason:
        description: Explain why another experiment would be invalid, test-informed, redundant, or otherwise scientifically inadmissible despite remaining budget.
        type: string
      evidence:
        description: Concrete board, data, protocol, or result facts that support ending the search now.
        type: array
        items:
          type: string
        minItems: 1
    required:
      - reason
      - evidence
---
