---
name: update_experiment
description: Update an experiment's status, results, or error message. Use this to indicate an experiment's state has transitioned.
metadata:
  workspace_access:
    experiments.db: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      debrief_path:
        description: Path to the debrief markdown file (relative to workspace).
        type: string
      error:
        description: Error message if the experiment failed.
        type: string
      experiment_id:
        description: The experiment ID to update.
        type: integer
      results:
        description: JSON string of result metrics (key-value pairs for the domain's metrics).
        type: string
      status:
        description: "New experiment status. Allowed transitions: to_implement -> to_implement | implemented | cancelled; implemented -> implemented | checked | cancelled; finished -> finished | checked | analyzed | cancelled; analyzed -> analyzed | done | cancelled."
        type: string
    required:
      - experiment_id
---
