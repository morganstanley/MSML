---
name: propose_experiment
description: "Propose a new experiment. Creates an entry in the experiment board with status 'to_implement'. A worker will implement and run it."
metadata:
  workspace_access:
    experiments.db: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      config:
        description: "JSON string with experiment config: {resource, model_type, hyperparams, features, horizon, etc.}. `resource` is REQUIRED and must exactly match an enabled device type (see the submission rules for the allowed values)."
        type: string
      description:
        description: Detailed description of what the experiment should do.
        type: string
      hypothesis:
        description: The hypothesis being tested.
        type: string
      name:
        description: "Short unique name for the experiment (used as directory name). Use snake_case, e.g. 'xgboost_momentum_5d'."
        type: string
    required:
      - name
      - description
      - hypothesis
      - config
---
