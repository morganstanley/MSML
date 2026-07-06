---
name: reality_check
description: "Run validation reality check on a slice of real data BEFORE marking experiment as checked. This catches data leakage, missing data, short OOS windows, and timing issues that smoke tests on synthetic data miss. REQUIRED after smoke test, before updating to 'checked' status."
metadata:
  workspace_access:
    experiments: rw
  parameters:
    additionalProperties: false
    type: object
    properties:
      experiment_name:
        description: "Name of the experiment directory (e.g. 'xgboost_momentum_5d')."
        type: string
    required:
      - experiment_name
---
