---
name: cancel_experiments
description: "Cancel one or more queued experiments. Use this to prune experiments that are unlikely to beat current best based on learnings from completed runs. Can only cancel experiments in 'to_implement' status (not yet started). Provide a reason for the cancellation."
metadata:
  workspace_access:
    experiments.db: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      experiment_ids:
        description: List of experiment IDs to cancel.
        items:
          type: integer
        type: array
      reason:
        description: "Why these experiments are being cancelled (e.g. 'Similar approach already failed in experiment #42')."
        type: string
    required:
      - experiment_ids
      - reason
---
